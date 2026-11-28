"""Audit C: the same exam (run_exam.py) on a HOSE VARIANT THE POLICY NEVER TRAINED ON. Training used
real:korea26_scan+duct+smooth and +hose0..15 (state_rl.build_env, warps 16); C_HOSE=k (k >= 16) swaps the exam's track for
real:korea26_scan+duct+smooth+hose<k> (Track.with_hoses: the duct hoses laid another way, 1-4 m sections moved 8-25 cm).
The exam3 box layouts are placed by index at the same world coordinates (they sit on the racing line, away from the hoses).
Args: as run_exam.py"""
import os, sys
sys.path.insert(0, "/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad/C")
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
import f1sim.learn.common as C

K = int(os.environ["C_HOSE"])
_lt = C.load_tracks


def load_tracks(names, *a, **k):
    names = [f"{n}+hose{K}" if n == "real:korea26_scan+duct+smooth" else n for n in names]
    print(f"[C] held-out tracks {names}", flush=True)
    return _lt(names, *a, **k)


C.load_tracks = load_tracks
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
import world_of
BANK_HO = os.environ["C_BANK"]                       # make_heldout_bank.py's directory (the exam3 layouts under the variant's name)
world_of.exam_bank = lambda world: BANK_HO
import runpy
runpy.run_path("/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad/C/run_exam.py", run_name="__main__")
