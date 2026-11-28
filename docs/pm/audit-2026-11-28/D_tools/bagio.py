"""Independent rosbag2 (sqlite3, CDR) reader for the audit -- no rclpy, no repo code.
Only the message types the audit needs. Little-endian CDR (encapsulation 0x0001)."""
import glob
import math
import os
import sqlite3
import struct

import numpy as np


class R:
    def __init__(self, b):
        self.b = b
        assert b[1] == 1, "only CDR_LE"
        self.o = 4

    def al(self, n):
        r = (self.o - 4) % n
        if r:
            self.o += n - r

    def f64(self, k=1):
        self.al(8)
        v = struct.unpack_from("<%dd" % k, self.b, self.o)
        self.o += 8 * k
        return v if k > 1 else v[0]

    def f32(self, k=1):
        self.al(4)
        v = struct.unpack_from("<%df" % k, self.b, self.o)
        self.o += 4 * k
        return v if k > 1 else v[0]

    def i32(self):
        self.al(4)
        v = struct.unpack_from("<i", self.b, self.o)[0]
        self.o += 4
        return v

    def u32(self):
        self.al(4)
        v = struct.unpack_from("<I", self.b, self.o)[0]
        self.o += 4
        return v

    def s(self):
        n = self.u32()
        v = self.b[self.o:self.o + n - 1].decode(errors="replace")
        self.o += n
        return v

    def header(self):
        sec = self.i32(); ns = self.u32(); fid = self.s()
        return sec + ns * 1e-9, fid

    def f32arr(self):
        n = self.u32()
        self.al(4)
        a = np.frombuffer(self.b, dtype="<f4", count=n, offset=self.o)
        self.o += 4 * n
        return a

    def i8arr(self):
        n = self.u32()
        a = np.frombuffer(self.b, dtype=np.int8, count=n, offset=self.o)
        self.o += n
        return a


def yaw_q(x, y, z, w):
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def p_float64(b):
    return (R(b).f64(),)


def p_ackermann(b):
    r = R(b); st, _ = r.header()
    sa, sav, sp, acc, jerk = r.f32(5)
    return (st, sa, sp)


def p_vesc(b):
    r = R(b); st, _ = r.header()
    temp_fet, temp_motor, cur_motor, cur_in, avg_id, avg_iq, duty, speed, v_in = r.f64(9)
    return (st, speed, cur_motor, duty, v_in, cur_in)


def p_imu(b):
    r = R(b); st, _ = r.header()
    q = r.f64(4); r.f64(9)
    w = r.f64(3); r.f64(9)
    a = r.f64(3)
    return (st, w[0], w[1], w[2], a[0], a[1], a[2], *q)


def p_odom(b):
    r = R(b); st, _ = r.header(); r.s()
    p = r.f64(3); q = r.f64(4); r.f64(36)
    lin = r.f64(3); ang = r.f64(3)
    return (st, p[0], p[1], yaw_q(*q), lin[0], lin[1], ang[2])


def p_scan(b):
    r = R(b); st, _ = r.header()
    meta = r.f32(7)
    rng = r.f32arr().copy()
    return st, meta, rng


def p_map(b):
    r = R(b); st, _ = r.header()
    # MapMetaData: Time map_load_time, float32 resolution, uint32 width, height, Pose origin
    r.i32(); r.u32()
    res = r.f32(); w = r.u32(); h = r.u32()
    p = r.f64(3); q = r.f64(4)
    data = r.i8arr().copy()
    return dict(res=res, w=w, h=h, ox=p[0], oy=p[1], oyaw=yaw_q(*q), data=data.reshape(h, w))


def p_tf(b):
    r = R(b)
    n = r.u32()
    out = []
    for _ in range(n):
        st, fid = r.header(); child = r.s()
        t = r.f64(3); q = r.f64(4)
        out.append((fid, child, t[0], t[1], t[2], q[0], q[1], q[2], q[3]))
    return out


PARSERS = {
    "/commands/servo/position": p_float64, "/commands/motor/speed": p_float64,
    "/sensors/servo_position_command": p_float64,
    "/drive": p_ackermann, "/teleop": p_ackermann, "/ackermann_cmd": p_ackermann,
    "/sensors/core": p_vesc, "/sensors/imu/raw": p_imu, "/odom": p_odom, "/pf/pose/odom": p_odom,
}


def open_bag(d):
    db = glob.glob(os.path.join(d, "*.db3"))[0]
    c = sqlite3.connect(db)
    topics = {r[1]: r[0] for r in c.execute("select id,name from topics")}
    t0 = c.execute("select min(timestamp) from messages").fetchone()[0]
    return c, topics, t0


def read_topic(c, topics, t0, name):
    if name not in topics:
        return None
    tid = topics[name]
    cur = c.execute("select timestamp,data from messages where topic_id=? order by timestamp", (tid,))
    return [((ts - t0) * 1e-9, bytes(d)) for ts, d in cur]
