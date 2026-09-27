"""Build (or confirm cached) racelines for a track split, a few processes at a time, CPU only.

usage: python scripts/prebuild_racelines.py SPLIT [WORKERS] [objective a_lat a_acc a_brake]
Prints one line per track: cached / built in N s / FAILED reason. Safe to re-run: cached lines return at once.
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from multiprocessing import Pool

SPLIT = sys.argv[1]; WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
OBJ = sys.argv[3] if len(sys.argv) > 3 else "min_time"
LIM = dict(a_lat=float(sys.argv[4]), a_acc=float(sys.argv[5]), a_brake=float(sys.argv[6])) if len(sys.argv) > 6 else \
      dict(a_lat=7.0, a_acc=6.5, a_brake=4.0)


def one(name):
    from f1sim.learn import common
    from f1sim import maps
    from f1sim.raceline import Raceline, RacelineCacheMiss
    t0 = time.time()
    try:
        tr = maps.load(name)
        try:
            Raceline.build_cached(tr, cache_only=True, objective=OBJ, **LIM)
            return f"{name:40s} cached"
        except RacelineCacheMiss:
            pass
        Raceline.build_cached(tr, objective=OBJ, **LIM)
        return f"{name:40s} built in {time.time() - t0:.0f} s"
    except Exception as e:
        return f"{name:40s} FAILED after {time.time() - t0:.0f} s: {type(e).__name__}: {str(e)[:120]}"


if __name__ == "__main__":
    from f1sim.learn import common
    names = common.track_names(SPLIT)
    print(f"{len(names)} tracks in {SPLIT}, objective {OBJ} {LIM}, {WORKERS} workers", flush=True)
    with Pool(WORKERS) as p:
        for line in p.imap_unordered(one, names):
            print(line, flush=True)
    print("PREBUILD_DONE", flush=True)
