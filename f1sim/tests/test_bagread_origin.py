"""`read()` must put every topic on one clock, and the same clock regardless of what was asked for.

The origin used to be `min(first stamp of each SELECTED topic)`. Two consequences, both of which
silently corrupted a lag estimate rather than failing:

* reading a subset moved the origin, so `read(bag, ["/odom"])` and `read(bag, ["/odom", "/drive"])`
  disagreed about when /odom happened -- by seconds, because /drive starts 1.96 s into these
  recordings. Results from separate reads could not be combined.
* taking each topic's *first* stamp assumed the records were written in timestamp order.

Stamps are also normalised in integer nanoseconds now: float64 spacing at a 2025 epoch timestamp in
ns is 256 ns, so converting before subtracting quantised exactly the sub-microsecond differences
these calibrations are quoted at.

The relative `t` this returns is intentionally different from before for any subset read.
"""
import sys
import types

import numpy as np
import pytest

from f1sim.calib import bagread


class _FakeMsg:
    def __init__(self, value):
        self.data = value


def _install_fake_bag(monkeypatch, records, topic_types):
    """Replay `records` = [(topic, value, stamp_ns)] through read()'s rosbag2_py/rclpy imports."""
    class Reader:
        def __init__(self):
            self._i = 0

        def open(self, *a, **k):
            pass

        def get_all_topics_and_types(self):
            return [types.SimpleNamespace(name=n, type=t) for n, t in topic_types.items()]

        def has_next(self):
            return self._i < len(records)

        def read_next(self):
            topic, value, stamp = records[self._i]
            self._i += 1
            return topic, value, stamp

    fake = types.ModuleType("rosbag2_py")
    fake.SequentialReader = Reader
    fake.StorageOptions = lambda **k: None
    fake.ConverterOptions = lambda *a: None
    monkeypatch.setitem(sys.modules, "rosbag2_py", fake)

    ser = types.ModuleType("rclpy.serialization")
    ser.deserialize_message = lambda raw, cls: _FakeMsg(raw)
    rclpy = types.ModuleType("rclpy")
    rclpy.serialization = ser
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.serialization", ser)

    util = types.ModuleType("rosidl_runtime_py.utilities")
    util.get_message = lambda t: _FakeMsg
    pkg = types.ModuleType("rosidl_runtime_py")
    pkg.utilities = util
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py", pkg)
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py.utilities", util)


EPOCH = 1_757_000_000_000_000_000            # a realistic 2025 record stamp, in ns
TYPES = {"/commands/motor/speed": "std_msgs/msg/Float64",
         "/sensors/servo_position_command": "std_msgs/msg/Float64"}
# motor starts at the bag's beginning; servo starts two seconds in, like /drive does in the real bags
RECORDS = ([("/commands/motor/speed", 1.0 + i, EPOCH + i * 25_000_000) for i in range(40)]
           + [("/sensors/servo_position_command", 0.1 * i, EPOCH + 2_000_000_000 + i * 20_000_000)
              for i in range(30)])
RECORDS.sort(key=lambda r: r[2])


def test_subset_and_combined_reads_agree_on_one_clock(monkeypatch):
    _install_fake_bag(monkeypatch, RECORDS, TYPES)
    both = bagread.read("/tmp/fake_bag", topics=list(TYPES))
    _install_fake_bag(monkeypatch, RECORDS, TYPES)
    servo_only = bagread.read("/tmp/fake_bag", topics=["/sensors/servo_position_command"])
    _install_fake_bag(monkeypatch, RECORDS, TYPES)
    motor_only = bagread.read("/tmp/fake_bag", topics=["/commands/motor/speed"])

    k_servo, k_motor = "/sensors/servo_position_command", "/commands/motor/speed"
    np.testing.assert_allclose(servo_only.t[k_servo], both.t[k_servo], atol=1e-12)
    np.testing.assert_allclose(motor_only.t[k_motor], both.t[k_motor], atol=1e-12)
    assert servo_only.origin_record_ns == motor_only.origin_record_ns == both.origin_record_ns == EPOCH
    # the servo topic really does start late; that offset is the signal, not something to zero away
    assert both.t[k_servo][0] == pytest.approx(2.0, abs=1e-9)
    assert both.t[k_motor][0] == pytest.approx(0.0, abs=1e-12)


def test_origin_is_the_smallest_record_even_when_records_are_out_of_order(monkeypatch):
    shuffled = list(reversed(RECORDS))               # writer order is not timestamp order
    _install_fake_bag(monkeypatch, shuffled, TYPES)
    out = bagread.read("/tmp/fake_bag", topics=list(TYPES))
    assert out.origin_record_ns == EPOCH, "origin must be the minimum record stamp, not the first"
    assert out.t["/commands/motor/speed"].min() == pytest.approx(0.0, abs=1e-12)


def test_origin_ignores_which_topics_were_requested(monkeypatch):
    # a bag whose earliest record is on a topic nobody asked for
    records = [("/commands/motor/speed", 1.0, EPOCH)] + \
              [("/sensors/servo_position_command", 0.5, EPOCH + 1_000_000_000)]
    _install_fake_bag(monkeypatch, records, TYPES)
    out = bagread.read("/tmp/fake_bag", topics=["/sensors/servo_position_command"])
    assert out.origin_record_ns == EPOCH
    assert out.t["/sensors/servo_position_command"][0] == pytest.approx(1.0, abs=1e-12)


def test_nanosecond_differences_survive_the_epoch(monkeypatch):
    """float64 spacing at this epoch is 256 ns, so subtracting after the conversion loses them."""
    step = 1_000                                      # 1 us apart
    records = [("/commands/motor/speed", float(i), EPOCH + i * step) for i in range(5)]
    _install_fake_bag(monkeypatch, records, TYPES)
    out = bagread.read("/tmp/fake_bag", topics=["/commands/motor/speed"])
    np.testing.assert_allclose(out.t["/commands/motor/speed"],
                               np.arange(5) * step * 1e-9, rtol=0, atol=1e-15)
    # and the naive order really would have destroyed them
    naive = np.asarray([float(EPOCH + i * step) for i in range(5)]) - float(EPOCH)
    assert not np.allclose(naive, np.arange(5) * step * 1e-9, atol=1e-15)


def test_empty_bag_and_no_matching_topics(monkeypatch):
    _install_fake_bag(monkeypatch, [], TYPES)
    empty = bagread.read("/tmp/fake_bag", topics=list(TYPES))
    assert empty.origin_record_ns is None and empty.t == {} and empty.v == {}
    assert not empty.has("/commands/motor/speed")

    _install_fake_bag(monkeypatch, RECORDS, TYPES)
    none_wanted = bagread.read("/tmp/fake_bag", topics=["/not/in/this/bag"])
    assert none_wanted.origin_record_ns == EPOCH, "the bag still has an origin"
    assert none_wanted.t == {}
