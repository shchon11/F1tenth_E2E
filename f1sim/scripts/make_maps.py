"""Export procedurally generated tracks as ROS map_server maps (+ centerline csv)."""
import sys
from f1sim import Track
out = sys.argv[1] if len(sys.argv) > 1 else "../f1sim_ros/maps"
seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2, 3]
for s in seeds:
    tr = Track.generate_random(s)
    print(tr.save_ros_map(out), tr.shape, f"{tr.centerline.shape[0]} centerline pts")
