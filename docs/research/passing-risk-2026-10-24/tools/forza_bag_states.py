"""What the ForzaETH stack that drove a real bag believed: its state machine (/state) and how often its perception
reported obstacles (/perception/obstacles, raw). usage (ROS sourced): forza_bag_states.py BAG"""
import collections
import struct
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from std_msgs.msg import String

BAG = sys.argv[1]
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
r.set_filter(rosbag2_py.StorageFilter(topics=["/state", "/perception/detection/debug/scan_decision", "/perception/obstacles",
                                              "/perception/detection/raw_obstacles"]))
states = collections.Counter(); dec = collections.Counter(); nobs = {"/perception/obstacles": collections.Counter(),
                                                                       "/perception/detection/raw_obstacles": collections.Counter()}
t_first = None; state_t = []
while r.has_next():
    t, data, ts = r.read_next()
    if t_first is None:
        t_first = ts
    if t == "/state":
        s = deserialize_message(data, String).data; states[s] += 1; state_t.append(((ts - t_first) * 1e-9, s))
    elif t == "/perception/detection/debug/scan_decision":
        dec[deserialize_message(data, String).data[:60]] += 1
    else:
        # CDR: 4-byte encapsulation, header (sec int32, nanosec uint32, frame_id string), then the obstacle sequence length
        b = data[4:]
        n = struct.unpack_from("<I", b, 8)[0]
        off = 12 + n
        off = (off + 3) // 4 * 4
        nobs[t][struct.unpack_from("<I", b, off)[0]] += 1
print("state:", dict(states.most_common(8)))
print("scan_decision (first 60 chars):", dict(dec.most_common(6)))
for k, c in nobs.items():
    tot = sum(c.values())
    print(f"{k}: messages {tot}, with >= 1 obstacle {100 * sum(v for n, v in c.items() if n > 0) / max(tot, 1):.1f} %, counts {dict(sorted(c.items())[:8])}")
chg = [state_t[0]] + [state_t[i] for i in range(1, len(state_t)) if state_t[i][1] != state_t[i - 1][1]]

