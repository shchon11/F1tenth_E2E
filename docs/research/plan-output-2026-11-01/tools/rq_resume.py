"""(11-27 22:25, ACE's NAS hung: hs43a at ~u375, nc45a at ~u1010 blocked) a run's own settings again from one of its checkpoints,
under a new name, on ACE's /data1: the run's args.json, minus the bookkeeping (name, runs, dist, init, offset, init_critic), plus
--init RUN/plan_uU.pt --init-critic --update-offset U, the world json's *_d1.json twin (bank paths under /data1). The lr
schedule continues (same --lr/--lr-end/--updates, counted on from U); the optimizer state starts fresh.
usage: python3 rq_resume.py RUN U NEWNAME"""
import json
import os
import sys

RUNS = "/home/shchon11/f1sim_runs"; SRC = "/home/shchon11/F1tenth_E2E/f1sim"
run, u, new = sys.argv[1], int(sys.argv[2]), sys.argv[3]
A = json.load(open(f"{RUNS}/{run}/args.json"))["args"]
SKIP = {"name", "dist_world", "dist_rank", "dist_init", "dist_backend", "dist_same_seed", "runs", "check_args", "init",
        "update_offset", "init_critic", "world"}
out = []
for k, v in A.items():
    if k in SKIP or v is None or v is False or v == "":
        continue
    opt = "--" + k.replace("_", "-")
    out += [opt] if v is True else ([opt] + [str(x) for x in v] if isinstance(v, (list, tuple)) else [opt, str(v)])
w = os.path.basename(A["world"]).replace("_ace.json", "_d1.json")
assert w.endswith("_d1.json"), w
out += ["--world", f"{SRC}/f1sim/learn/configs/{w}", "--init", f"{RUNS}/{run}/plan_u{u}.pt", "--init-critic",
        "--update-offset", str(u), "--name", new]
print(" ".join(out))
