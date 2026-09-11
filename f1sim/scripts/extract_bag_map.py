"""Pull the competition track's occupancy grid out of a recording into a ROS map_server map.

The bags carry the /map the stack was localising against, so the venue the car actually raced on
can be driven in the simulator instead of being approximated by a generated track.
"""
import os, sys, glob
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

OUT = "/home/shchon11/F1tenth/F1tenth_E2E/f1sim/f1sim/assets/maps"

def read_map(path):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    if "/map" not in types:
        return None
    while r.has_next():
        topic, data, stamp = r.read_next()
        if topic == "/map":
            return deserialize_message(data, get_message(types[topic]))
    return None

def write_ros_map(m, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    g = np.asarray(m.data, np.int8).reshape(m.info.height, m.info.width)
    # map_server greyscale: 0 occupied, 254 free, 205 unknown; row 0 of the image is the TOP, while
    # the grid's row 0 is at the origin, so the image is flipped before writing.
    img = np.full(g.shape, 205, np.uint8)
    img[g == 0] = 254
    img[g >= 50] = 0
    img = np.flipud(img)
    pgm = os.path.join(out_dir, f"{name}.pgm")
    with open(pgm, "wb") as f:
        f.write(b"P5\n"); f.write(f"# from an F1TENTH recording, /map as published by the stack\n".encode())
        f.write(f"{g.shape[1]} {g.shape[0]}\n255\n".encode()); f.write(img.tobytes())
    o = m.info.origin.position
    with open(os.path.join(out_dir, f"{name}.yaml"), "w") as f:
        f.write(f"image: {name}.pgm\nresolution: {m.info.resolution}\n"
                f"origin: [{o.x}, {o.y}, 0.0]\nnegate: 0\n"
                f"occupied_thresh: 0.65\nfree_thresh: 0.196\n")
    free = int((g == 0).sum()); occ = int((g >= 50).sum()); unk = int((g < 0).sum())
    print(f"{name}: {g.shape[1]}x{g.shape[0]} @ {m.info.resolution} m  "
          f"({g.shape[1]*m.info.resolution:.1f} x {g.shape[0]*m.info.resolution:.1f} m)")
    print(f"   free {free} ({free/g.size*100:.1f}%)  occupied {occ}  unknown {unk}")
    print(f"   origin ({o.x:.3f}, {o.y:.3f})   -> {pgm}")
    return pgm

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", help="rosbag2 directory holding a /map")
    ap.add_argument("name", help="map name; writes <name>.pgm and <name>.yaml")
    ap.add_argument("--out", default=OUT, help="output directory")
    a = ap.parse_args()
    m = read_map(a.bag)
    if m is None:
        raise SystemExit(f"no /map in {a.bag}")
    print("source:", os.path.basename(a.bag.rstrip("/")))
    write_ros_map(m, a.name, a.out)
    print(f"\nregister it in f1sim/maps.py REAL as e.g.\n"
          f'    "{a.name}": (os.path.join(ASSET_MAPS, "{a.name}.yaml"), "duct", 0.35, "..."),')
