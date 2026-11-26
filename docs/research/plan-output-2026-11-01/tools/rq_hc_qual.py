"""(11-27) Qualifier-world fork of scan31b2 u600 with ONE change: --hard-contact-ends (a hard contact ends the return: GAE
terminal value 0, the car drives on, memory kept). Question (rq_plan_2026-11-26.md section 0 + RQ2, T-maze/nudge results 11-27):
under the parent's reward a hard hit costs ~2-4 s and mild caution at a blind spot ~0.05-0.1 s, so hits below ~2.6 % per pass
are the reward's optimum; on this map a memory-based line change before a hidden box is geometrically impossible (tmaze_room.py:
0 of 7 late-visible spots have room), so the lever for 'never ram / no re-hit' is the price of a hit, not memory.
Prediction vs scan31b2 u625-u750 (same lineage, pooled exam3, box hard 0.17 per box session, empty lap 7.1 s): box hard per box
session <= 0.08 pooled over u700-u900 and empty lap <= 7.2 s. Falsifier: lap > 7.4 s or stopped share > 2x (timid optimum).
Runs on ACE GPU 3 (tools/ace_run.sh), 1536 envs, updates 601..900, checkpoints every 50; same lr bookkeeping as rq_tmaze_v4.py.
Usage: python3 rq_hc_qual.py args   -- print the argv (asus paths) for ace_run.sh launch"""
import json, sys
RUNS = "/home/shchon11/f1sim_runs"; SRC = "/home/shchon11/F1tenth_E2E/f1sim"
NAME, PARENT, OFFSET, N_UPD, EVERY, ENVS = "hc41a", RUNS + "/scan31b2/plan_u600.pt", 600, 300, 50, 1536
LR0, LR1 = 2e-4, 1e-4
A = json.load(open(RUNS + "/scan31b2/args.json"))["args"]
SKIP = {"name", "dist_world", "dist_rank", "dist_init", "dist_backend", "dist_same_seed", "runs", "check_args", "init",
        "update_offset", "envs", "updates", "init_critic", "self_opp_init", "save_every", "lr", "lr_end", "live_tap", "live_taps"}
base = []
for k, v in A.items():
    if k in SKIP or v is None or v is False or v == "":     # an empty string is the default and does not survive a shell
        continue
    opt = "--" + k.replace("_", "-")
    base += [opt] if v is True else ([opt] + [str(x) for x in v] if isinstance(v, (list, tuple)) else [opt, str(v)])
U_LAST = OFFSET + N_UPD
f0 = OFFSET / (U_LAST - 1)
LR_ARG = (LR0 - LR1 * f0) / (1.0 - f0)
base += ["--world", SRC + "/f1sim/learn/configs/plan_world_korea26scan_qual_ace.json",   # the bank paths on ACE
         "--envs", str(ENVS), "--updates", str(U_LAST), "--update-offset", str(OFFSET), "--save-every", str(EVERY),
         "--init", PARENT, "--init-critic", "--lr", f"{LR_ARG:.6g}", "--lr-end", f"{LR1:g}", "--hard-contact-ends", "--name", NAME]
if __name__ == "__main__":
    print(" ".join(base))
