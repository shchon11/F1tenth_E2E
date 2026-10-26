"""End to end on the deployed path: the console's worker owns the simulation (Korea, a race of 2, a ForzaETH opponent)
and hands car 0 to ROS (ros2="drive"); graph_console.launch.py runs the policy node and the controller node with
opponent_speed_limit:=true (10-25) and, with OBS=1, obstacle_speed_limit:=true on the training's own prop layouts (10-26).
After a while the controller's diagnostics say how many commands each limit held.
usage (ROS sourced): e2e_ros.py CKPT [SECONDS]"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim/tests")


def main():
    from test_sim_worker_smoke import WorkerProc
    from f1sim.viewer.console import protocol as P
    ck = sys.argv[1]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 90
    here = Path(os.path.dirname(os.path.abspath(__file__)))
    w = WorkerProc(here)
    launch = None
    try:
        cfg = P.SessionConfig(run=ck, map_name="real:korea_2026_competition", races=1, cars_per_race=2, device="cuda:0",
                              compile=False, plan_follower="map", opponent="slots",
                              opponent_slots=[{"kind": "forzaeth", "speed_scale": [0.7, 0.8]}],
                              ros2="drive", controller="legacy", procedural=bool(os.environ.get("OBS")))
        w.send(P.CMD_START, gen=11, config=cfg.to_dict())
        ready = w.wait_for(P.MSG_READY, timeout=600, gen=11)
        print("READY ros2:", ready["facts"].get("ros2"), flush=True)
        log = open(here / "launch.log", "w")
        launch = subprocess.Popen(["ros2", "launch", "f1sim_ros", "graph_console.launch.py", f"checkpoint:={ck}",
                                   "controller:=legacy", "opponent_speed_limit:=true", "device:=cpu", "speed_cap:=9.0",
                                   *(["obstacle_speed_limit:=true"] if os.environ.get("OBS") else [])],
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            w.collect_frames(5)
        out = subprocess.run(["ros2", "topic", "echo", "--once", "/f1sim/controller/diag"], capture_output=True, text=True,
                             timeout=30)
        keep = [l for l in out.stdout.splitlines() if any(k in l for k in ("key:", "value:"))]
        pairs = {}
        for a, b in zip(keep[0::2], keep[1::2]):
            pairs[a.split("key:")[-1].strip().strip("'")] = b.split("value:")[-1].strip().strip("'")
        for k in ("plan_follower", "opponent_speed_limit", "opponent_limited_commands", "opponent_no_reading_commands", "opponent_previous_scan_readings", "obstacle_speed_limit",
                  "obstacle_limited_commands", "speed_cap_mps",
                  "commands", "speed_cmd_mps", "plan_checkpoint"):
            print(f"{k}: {pairs.get(k)}", flush=True)
    finally:
        if launch is not None:
            os.killpg(launch.pid, 2)
            try:
                launch.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(launch.pid, 9)
        w.close()


if __name__ == "__main__":
    main()
