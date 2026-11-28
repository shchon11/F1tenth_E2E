"""bag dir -> npz with the audit's signals (record time t, seconds from the bag's first record)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bagio as B

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "npz")


def main(d):
    name = os.path.basename(d.rstrip("/"))
    c, topics, t0 = B.open_bag(d)
    out = {}
    for tp, key in (("/commands/servo/position", "servo_cmd"), ("/commands/motor/speed", "erpm_cmd"),
                    ("/sensors/servo_position_command", "servo_pos"),
                    ("/drive", "drive"), ("/teleop", "teleop"), ("/sensors/core", "core"),
                    ("/sensors/imu/raw", "imu"), ("/odom", "odom"), ("/pf/pose/odom", "pf")):
        rows = B.read_topic(c, topics, t0, tp)
        if not rows:
            continue
        p = B.PARSERS[tp]
        a = np.array([(t,) + tuple(p(b)) for t, b in rows], dtype=np.float64)
        out[key] = a
    rows = B.read_topic(c, topics, t0, "/scan")
    if rows:
        ts, ss, R = [], [], []
        for t, b in rows:
            st, meta, rng = B.p_scan(b)
            ts.append(t); ss.append(st); R.append(rng.astype(np.float32))
        out["scan_t"] = np.array(ts); out["scan_stamp"] = np.array(ss)
        out["scan"] = np.stack(R).astype(np.float32)
        out["scan_meta"] = np.array(meta)
    rows = B.read_topic(c, topics, t0, "/map")
    if rows:
        m = B.p_map(rows[-1][1])         # last /map (pre-competition SLAM grows the map)
        out["map"] = m["data"]; out["map_info"] = np.array([m["res"], m["ox"], m["oy"], m["oyaw"]])
        out["n_map_msgs"] = np.array(len(rows))
    rows = B.read_topic(c, topics, t0, "/tf_static")
    if rows:
        out["tf_static"] = np.array([str(x) for x in B.p_tf(rows[0][1])])
    np.savez_compressed(os.path.join(OUT, name + ".npz"), **out)
    print(name, {k: v.shape for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
