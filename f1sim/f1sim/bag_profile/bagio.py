"""Pure-python rosbag2 (sqlite3, CDR little-endian) reader -- no rclpy, no message packages.

Why not `calib/bagread.py`: that one goes through `rosbag2_py` + `rosidl_runtime_py`, which needs a sourced ROS
workspace and the message packages installed (`vesc_msgs` is not on the desk machines), and it reads the whole bag.
This module reads only the topics asked for, streams them out of the sqlite file without touching the other topics'
payloads (a LiDAR blob spills onto overflow pages that a `topic_id` filter never reads), and decodes the handful of
message types a vehicle profile needs straight from the CDR bytes. The decoders were checked against
`docs/pm/audit-2026-11-28/D_tools/bagio.py` (the audit's own independent reader) on the same bags.

Time convention: `t` is seconds from the bag's first record (metadata `starting_time`, which is the minimum record
timestamp), the same origin `calib/bagread.py` uses, so numbers from the two readers line up.
"""
from __future__ import annotations

import glob
import math
import os
import sqlite3
import struct
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np


class CDR:
    """Minimal little-endian CDR cursor (alignment relative to the 4-byte encapsulation header)."""

    def __init__(self, b: bytes):
        if len(b) < 4 or b[1] != 1:
            raise ValueError("only CDR little-endian payloads are supported")
        self.b = b
        self.o = 4

    def _al(self, n: int) -> None:
        r = (self.o - 4) % n
        if r:
            self.o += n - r

    def f64(self, k: int = 1):
        self._al(8)
        v = struct.unpack_from("<%dd" % k, self.b, self.o)
        self.o += 8 * k
        return v if k > 1 else v[0]

    def f32(self, k: int = 1):
        self._al(4)
        v = struct.unpack_from("<%df" % k, self.b, self.o)
        self.o += 4 * k
        return v if k > 1 else v[0]

    def i32(self) -> int:
        self._al(4)
        v = struct.unpack_from("<i", self.b, self.o)[0]
        self.o += 4
        return v

    def u32(self) -> int:
        self._al(4)
        v = struct.unpack_from("<I", self.b, self.o)[0]
        self.o += 4
        return v

    def string(self) -> str:
        n = self.u32()
        v = self.b[self.o:self.o + max(n - 1, 0)].decode(errors="replace")
        self.o += n
        return v

    def header(self) -> Tuple[float, str]:
        sec = self.i32()
        ns = self.u32()
        fid = self.string()
        return sec + ns * 1e-9, fid

    def f32seq(self) -> np.ndarray:
        n = self.u32()
        self._al(4)
        a = np.frombuffer(self.b, dtype="<f4", count=n, offset=self.o)
        self.o += 4 * n
        return a

    def i8seq(self) -> np.ndarray:
        n = self.u32()
        a = np.frombuffer(self.b, dtype=np.int8, count=n, offset=self.o)
        self.o += n
        return a


def yaw_of(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ---- decoders: bytes -> tuple of floats (fixed width per type) ------------------------------------------------------

def dec_float64(b):
    return (CDR(b).f64(),)


def dec_ackermann(b):
    """ackermann_msgs/AckermannDriveStamped -> (stamp, steering_angle, speed, acceleration)."""
    r = CDR(b)
    st, _ = r.header()
    sa, sav, sp, acc, jerk = r.f32(5)
    return (st, sa, sp, acc)


def dec_vesc_state(b):
    """vesc_msgs/VescStateStamped -> (stamp, speed[ERPM], current_motor, duty_cycle, voltage_input, current_input)."""
    r = CDR(b)
    st, _ = r.header()
    temp_fet, temp_motor, cur_motor, cur_in, avg_id, avg_iq, duty, speed, v_in = r.f64(9)
    return (st, speed, cur_motor, duty, v_in, cur_in)


def dec_imu(b):
    """sensor_msgs/Imu -> (stamp, wx, wy, wz, ax, ay, az, qx, qy, qz, qw) in the units the driver published."""
    r = CDR(b)
    st, _ = r.header()
    q = r.f64(4)
    r.f64(9)
    w = r.f64(3)
    r.f64(9)
    a = r.f64(3)
    return (st, w[0], w[1], w[2], a[0], a[1], a[2], q[0], q[1], q[2], q[3])


def dec_odom(b):
    """nav_msgs/Odometry -> (stamp, x, y, yaw, vx, vy, wz)."""
    r = CDR(b)
    st, _ = r.header()
    r.string()                                   # child_frame_id
    p = r.f64(3)
    q = r.f64(4)
    r.f64(36)
    lin = r.f64(3)
    ang = r.f64(3)
    return (st, p[0], p[1], yaw_of(*q), lin[0], lin[1], ang[2])


def dec_pose_stamped(b):
    r = CDR(b)
    st, _ = r.header()
    p = r.f64(3)
    q = r.f64(4)
    return (st, p[0], p[1], yaw_of(*q))


def dec_scan(b):
    """sensor_msgs/LaserScan -> (stamp, meta[7], ranges float32[]) -- variable width, see `read_scans`."""
    r = CDR(b)
    st, _ = r.header()
    meta = r.f32(7)          # angle_min, angle_max, angle_increment, time_increment, scan_time, range_min, range_max
    rng = r.f32seq()
    return st, meta, rng


def dec_map(b):
    r = CDR(b)
    st, fid = r.header()
    r.i32(); r.u32()                             # map_load_time
    res = r.f32()
    w = r.u32()
    h = r.u32()
    p = r.f64(3)
    q = r.f64(4)
    data = r.i8seq()
    return dict(stamp=st, frame=fid, res=float(res), w=int(w), h=int(h), ox=p[0], oy=p[1], oyaw=yaw_of(*q),
                data=data.reshape(h, w).copy())


def dec_tf(b):
    """tf2_msgs/TFMessage -> list of (parent, child, x, y, z, qx, qy, qz, qw)."""
    r = CDR(b)
    n = r.u32()
    out = []
    for _ in range(n):
        st, fid = r.header()
        child = r.string()
        t = r.f64(3)
        q = r.f64(4)
        out.append((fid, child, t[0], t[1], t[2], q[0], q[1], q[2], q[3]))
    return out


def dec_string(b):
    return CDR(b).string()


#: message type -> fixed-width decoder. Types not here are reported as present but not decoded.
DECODERS: Dict[str, Callable] = {
    "std_msgs/msg/Float64": dec_float64,
    "ackermann_msgs/msg/AckermannDriveStamped": dec_ackermann,
    "vesc_msgs/msg/VescStateStamped": dec_vesc_state,
    "sensor_msgs/msg/Imu": dec_imu,
    "nav_msgs/msg/Odometry": dec_odom,
    "geometry_msgs/msg/PoseStamped": dec_pose_stamped,
}


class BagFormatError(Exception):
    """The recording cannot be read by this tool; the message says what is wrong and how to convert it."""


@dataclass
class TopicInfo:
    name: str
    type: str
    count: int
    id: Optional[int] = None


@dataclass
class Bag:
    path: str
    db: str                                   # the first file (kept for callers that want one path)
    topics: Dict[str, TopicInfo] = field(default_factory=dict)
    t0_ns: int = 0
    duration: float = 0.0
    dbs: List[str] = field(default_factory=list)                       # every .db3 of a split bag, in recording order
    db_ids: Dict[str, Dict[str, int]] = field(default_factory=dict)    # per file: topic name -> topic_id

    @property
    def name(self) -> str:
        return os.path.basename(self.path.rstrip("/"))

    def has(self, topic: str) -> bool:
        return topic in self.topics and self.topics[topic].count > 0

    def of_type(self, typ: str) -> List[str]:
        return sorted(n for n, t in self.topics.items() if t.type == typ and t.count > 0)

    # -- streaming --------------------------------------------------------------------------------------------------
    @staticmethod
    def _connect(db: str):
        # read-only, immutable: never takes a lock or writes a journal next to someone's recording
        return sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)

    def iter_raw(self, names: Iterable[str]) -> Iterator[Tuple[str, float, bytes]]:
        """(topic, t [s from bag start], payload) for the named topics, over every file of a split bag, in file order
        (not sorted)."""
        names = list(names)
        for db in self.dbs or [self.db]:
            ids = {self.db_ids.get(db, {}).get(n): n for n in names if self.db_ids.get(db, {}).get(n) is not None}
            if not ids:
                continue
            c = self._connect(db)
            try:
                q = "select topic_id, timestamp, data from messages where topic_id in (%s)" % ",".join("?" * len(ids))
                for tid, ts, data in c.execute(q, tuple(ids)):
                    yield ids[tid], (ts - self.t0_ns) * 1e-9, bytes(data)
            finally:
                c.close()

    def read(self, names: Iterable[str]) -> Dict[str, np.ndarray]:
        """Decode fixed-width topics into (N, 1 + k) float64 arrays: column 0 = record time t, then the decoder's
        fields (for header types the first field is the header stamp, absolute seconds). Sorted by t."""
        names = [n for n in names if self.has(n) and self.topics[n].type in DECODERS]
        acc: Dict[str, list] = {n: [] for n in names}
        for n, t, b in self.iter_raw(names):
            try:
                acc[n].append((t,) + tuple(DECODERS[self.topics[n].type](b)))
            except (struct.error, ValueError):
                continue
        out = {}
        for n, rows in acc.items():
            if not rows:
                continue
            a = np.asarray(rows, dtype=np.float64)
            out[n] = a[np.argsort(a[:, 0], kind="stable")]
        return out

    def read_one(self, name: str, decoder: Callable, which: str = "first"):
        """Decode one message of a topic (first or last by record time), e.g. /map or /tf_static."""
        best = None
        for n, t, b in self.iter_raw([name]):
            if best is None or (t < best[0] if which == "first" else t > best[0]):
                best = (t, b)
        if best is None:
            return None
        return best[0], decoder(best[1])

    def read_scans(self, name: str = "/scan", every: int = 1, t_range=None):
        """Stream LaserScan messages: returns (t[N], stamp[N], meta[7], ranges float32 (N, beams)). `every` keeps one
        message in `every` (after sorting by time) to bound memory on long bags."""
        rows = []
        for n, t, b in self.iter_raw([name]):
            if t_range is not None and not (t_range[0] <= t <= t_range[1]):
                continue
            rows.append((t, b))
        rows.sort(key=lambda x: x[0])
        rows = rows[::max(1, every)]
        if not rows:
            return None
        ts, ss, R, meta = [], [], [], None
        nb = None
        for t, b in rows:
            try:
                st, meta, rng = dec_scan(b)
            except (struct.error, ValueError):
                continue
            if nb is None:
                nb = len(rng)
            if len(rng) != nb:
                continue
            ts.append(t); ss.append(st); R.append(np.asarray(rng, dtype=np.float32))
        return np.asarray(ts), np.asarray(ss), np.asarray(meta, dtype=np.float64), np.stack(R)


def _convert_hint(bag_dir: str) -> str:
    out = bag_dir.rstrip("/") + "_sqlite3"
    return ("sqlite3 로 변환하는 방법 (ROS 2 Humble 이 있는 PC 에서):\n"
            f"  1) 변환 설정 파일 convert.yaml 을 만든다:\n"
            f"       output_bags:\n"
            f"       - uri: {out}\n"
            f"         storage_id: sqlite3\n"
            f"         all: true\n"
            f"  2) ros2 bag convert -i {bag_dir.rstrip('/')} -o convert.yaml\n"
            f"  3) 이 도구를 {out} 에 돌린다.\n"
            "  (앞으로 녹화할 때는 `ros2 bag record -s sqlite3 ...` 로, 저장 형식 sqlite3 를 명시한다.)")


def find_dbs(path: str) -> List[str]:
    """Every .db3 file of the recording, in recording order. Raises BagFormatError with a conversion recipe when the
    recording is in a format this reader does not handle (mcap), and FileNotFoundError when there is nothing."""
    path = path.rstrip("/")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: 없는 경로")
    if os.path.isfile(path):
        if path.endswith(".db3"):
            return [path]
        if path.endswith(".mcap"):
            raise BagFormatError(f"{path}: mcap 파일은 읽지 못합니다 (이 도구는 rosbag2 sqlite3 만 읽음).\n"
                                 + _convert_hint(os.path.dirname(path) or "."))
        raise BagFormatError(f"{path}: bag 폴더(metadata.yaml 이 있는 곳) 또는 .db3 파일을 주십시오")
    meta = os.path.join(path, "metadata.yaml")
    storage = None
    rel: List[str] = []
    if os.path.exists(meta):
        try:
            import yaml
            m = yaml.safe_load(open(meta))["rosbag2_bagfile_information"]
            storage = m.get("storage_identifier")
            rel = [str(x) for x in m.get("relative_file_paths", [])]
        except Exception:
            pass
    dbs = [os.path.join(path, r) for r in rel if r.endswith(".db3") and os.path.exists(os.path.join(path, r))]
    if not dbs:
        dbs = sorted(glob.glob(os.path.join(path, "*.db3")))
    if not dbs:
        mcaps = sorted(glob.glob(os.path.join(path, "*.mcap")))
        if mcaps or storage == "mcap":
            raise BagFormatError(f"{path}: 저장 형식이 mcap 입니다 (파일 {len(mcaps)}개). 이 도구는 rosbag2 sqlite3(.db3)만 읽습니다.\n"
                                 + _convert_hint(path))
        raise BagFormatError(f"{path}: .db3 파일이 없습니다 (rosbag2 sqlite3 bag 이 아님: 저장 형식 {storage or '알 수 없음'}).")
    return dbs


def find_db(path: str) -> str:
    return find_dbs(path)[0]


def open_bag(path: str) -> Bag:
    """Topic table with message counts over every file of the bag. Counts come from metadata.yaml when it exists
    (instant); otherwise from one COUNT pass over the messages tables (slow on a large bag, but only the row headers
    are read)."""
    path = path.rstrip("/")
    dbs = find_dbs(path)
    bag_dir = os.path.dirname(dbs[0]) if os.path.isfile(path) else path
    bag = Bag(path=bag_dir, db=dbs[0], dbs=list(dbs))
    count_from_db: Dict[str, int] = {}
    lo_all, hi_all = None, None
    meta = os.path.join(bag_dir, "metadata.yaml")
    mdata = None
    if os.path.exists(meta):
        try:
            import yaml
            mdata = yaml.safe_load(open(meta))["rosbag2_bagfile_information"]
            mdata["starting_time"]["nanoseconds_since_epoch"]; mdata["duration"]["nanoseconds"]
        except Exception:
            mdata = None
    have_meta = mdata is not None
    for db in dbs:
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
            rows = list(c.execute("select id, name, type, serialization_format from topics"))
        except sqlite3.DatabaseError as e:
            raise BagFormatError(f"{db}: sqlite3 로 열 수 없습니다 ({e}). mcap 등 다른 저장 형식이거나 깨진 파일입니다.\n"
                                 + _convert_hint(bag_dir)) from e
        try:
            bag.db_ids[db] = {}
            for tid, name, typ, ser in rows:
                if ser and ser != "cdr":
                    raise BagFormatError(f"{db}: 토픽 {name} 의 직렬화 형식이 {ser} 입니다 (cdr 만 읽음). "
                                         "`ros2 bag record` 기본값(cdr)으로 다시 녹화하거나 변환하십시오.")
                bag.db_ids[db][name] = tid
                if name not in bag.topics:
                    bag.topics[name] = TopicInfo(name=name, type=typ, count=0, id=tid)
            if not have_meta:
                for tid, n in c.execute("select topic_id, count(*) from messages group by topic_id"):
                    for name, i in bag.db_ids[db].items():
                        if i == tid:
                            count_from_db[name] = count_from_db.get(name, 0) + int(n)
                lo, hi = c.execute("select min(timestamp), max(timestamp) from messages").fetchone()
                if lo is not None:
                    lo_all = lo if lo_all is None else min(lo_all, lo)
                    hi_all = hi if hi_all is None else max(hi_all, hi)
        finally:
            c.close()
    if have_meta:
        m = mdata
        for tw in m.get("topics_with_message_count", []):
            nm = tw["topic_metadata"]["name"]
            if nm in bag.topics:
                bag.topics[nm].count = int(tw["message_count"])
        bag.t0_ns = int(m["starting_time"]["nanoseconds_since_epoch"])
        bag.duration = float(m["duration"]["nanoseconds"]) * 1e-9
    else:
        for name, n in count_from_db.items():
            bag.topics[name].count = n
        bag.t0_ns = int(lo_all or 0)
        bag.duration = ((hi_all or 0) - (lo_all or 0)) * 1e-9
    return bag


# ---- writer (tests only): a minimal rosbag2 sqlite file with CDR payloads --------------------------------------------

class CDRWriter:
    def __init__(self):
        self.b = bytearray(b"\x00\x01\x00\x00")

    def _al(self, n):
        r = (len(self.b) - 4) % n
        if r:
            self.b += b"\x00" * (n - r)

    def f64(self, *v):
        self._al(8); self.b += struct.pack("<%dd" % len(v), *v)

    def f32(self, *v):
        self._al(4); self.b += struct.pack("<%df" % len(v), *v)

    def i32(self, v):
        self._al(4); self.b += struct.pack("<i", v)

    def u32(self, v):
        self._al(4); self.b += struct.pack("<I", v)

    def string(self, s):
        e = s.encode() + b"\x00"
        self.u32(len(e)); self.b += e

    def header(self, t, frame="base_link"):
        sec = int(math.floor(t)); ns = int(round((t - sec) * 1e9))
        self.i32(sec); self.u32(ns); self.string(frame)

    def f32seq(self, a):
        self.u32(len(a)); self._al(4); self.b += np.asarray(a, dtype="<f4").tobytes()

    def i8seq(self, a):
        self.u32(len(a)); self.b += np.asarray(a, dtype=np.int8).tobytes()


def enc_float64(v):
    w = CDRWriter(); w.f64(v); return bytes(w.b)


def enc_string(text):
    w = CDRWriter(); w.string(text); return bytes(w.b)


def enc_ackermann(t, steer, speed):
    w = CDRWriter(); w.header(t); w.f32(steer, 0.0, speed, 0.0, 0.0); return bytes(w.b)


def enc_vesc_state(t, erpm, cur, duty=0.0, vin=16.0):
    w = CDRWriter(); w.header(t); w.f64(30.0, 30.0, cur, 0.0, 0.0, 0.0, duty, erpm, vin, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    w.i32(0); w.f64(0.0); w.i32(0); w.f64(0.0, 0.0, 0.0)
    return bytes(w.b)


def enc_imu(t, w3, a3, q=(0.0, 0.0, 0.0, 1.0)):
    w = CDRWriter(); w.header(t, "imu"); w.f64(*q); w.f64(*([0.0] * 9)); w.f64(*w3); w.f64(*([0.0] * 9)); w.f64(*a3)
    w.f64(*([0.0] * 9))
    return bytes(w.b)


def enc_odom(t, x, y, yaw, vx, vy=0.0, wz=0.0):
    w = CDRWriter(); w.header(t, "map"); w.string("base_link"); w.f64(x, y, 0.0)
    w.f64(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)); w.f64(*([0.0] * 36)); w.f64(vx, vy, 0.0)
    w.f64(0.0, 0.0, wz); w.f64(*([0.0] * 36))
    return bytes(w.b)


def enc_scan(t, ranges, angle_min=-2.35619449, angle_inc=0.00436332313, range_max=30.0):
    w = CDRWriter(); w.header(t, "laser")
    w.f32(angle_min, angle_min + angle_inc * (len(ranges) - 1), angle_inc, 0.0, 0.025, 0.02, range_max)
    w.f32seq(ranges); w.f32seq([])
    return bytes(w.b)


def enc_tf(entries):
    w = CDRWriter(); w.u32(len(entries))
    for parent, child, x, y, z in entries:
        w.header(0.0, parent); w.string(child); w.f64(x, y, z); w.f64(0.0, 0.0, 0.0, 1.0)
    return bytes(w.b)


def write_bag(path: str, topics: Dict[str, str], messages: List[Tuple[str, float, bytes]],
              t0_ns: int = 1_700_000_000_000_000_000, split: int = 1, storage: str = "sqlite3"):
    """Write a minimal rosbag2 sqlite3 bag (plus metadata.yaml) -- for the unit tests' synthetic recordings.
    `split` > 1 cuts the recording by time into that many .db3 files like `ros2 bag record --max-bag-duration`."""
    os.makedirs(path, exist_ok=True)
    base = os.path.basename(path.rstrip("/"))
    tmax = max([int(round(t * 1e9)) for _, t, _ in messages] or [0])
    edges = [tmax * (k + 1) / split for k in range(split)]
    names = []
    counts = {n: 0 for n in topics}
    tmax_ts = 0
    ids = {n: i for i, n in enumerate(topics, 1)}
    for part in range(split):
        db = os.path.join(path, f"{base}_{part}.db3")
        names.append(os.path.basename(db))
        if os.path.exists(db):
            os.remove(db)
        c = sqlite3.connect(db)
        c.execute("CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT NOT NULL,type TEXT NOT NULL,"
                  "serialization_format TEXT NOT NULL,offered_qos_profiles TEXT NOT NULL)")
        c.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER NOT NULL,timestamp INTEGER NOT NULL,"
                  " data BLOB NOT NULL)")
        for n, typ in topics.items():
            c.execute("insert into topics values (?,?,?,?,?)", (ids[n], n, typ, "cdr", ""))
        lo = edges[part - 1] if part else -1
        for n, t, b in messages:
            tn = int(round(t * 1e9))
            if not (lo < tn <= edges[part] or (part == split - 1 and tn > edges[part])):
                continue
            ts = t0_ns + tn
            tmax_ts = max(tmax_ts, ts)
            c.execute("insert into messages(topic_id, timestamp, data) values (?,?,?)", (ids[n], ts, b))
            counts[n] += 1
        c.commit(); c.close()
    import yaml
    meta = {"rosbag2_bagfile_information": {
        "version": 5, "storage_identifier": storage, "relative_file_paths": names,
        "duration": {"nanoseconds": int(tmax_ts - t0_ns)}, "starting_time": {"nanoseconds_since_epoch": int(t0_ns)},
        "message_count": len(messages),
        "topics_with_message_count": [{"topic_metadata": {"name": n, "type": typ, "serialization_format": "cdr",
                                                          "offered_qos_profiles": ""}, "message_count": counts[n]}
                                      for n, typ in topics.items()]}}
    with open(os.path.join(path, "metadata.yaml"), "w") as f:
        yaml.safe_dump(meta, f)
    return path
