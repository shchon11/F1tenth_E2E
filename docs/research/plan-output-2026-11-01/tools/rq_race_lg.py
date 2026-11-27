"""(11-27 17:10, user: "ㄱㄱㄱ 라그랑주 멋있다") lg44a = rc42a u950 with --car-lagrange 0.1,0.5,30,3: walls and boxes still end the
return, a hard at-fault car contact is priced by an adaptive multiplier (target 0.1 per 1v1 minute). rc42a showed the
termination made passing a loss (0.8x passes 1.78 -> 0.22 with room beside the car 47 % of the time following).
Prediction: within 100-150 updates 0.8x passes >= 1 per exam4 session with hard at-fault car contacts <= 0.15 per session.
Falsifier: lambda falls to its floor and passing does not return, or lambda swings timid <-> dirty without settling.
(Below: the rc42a builder this copies.)
(11-27) Race stage with the hit price that worked in the qualifier: rc42a = hc41a u900 (scan31b2 u600 + --hard-contact-ends,
box hard 0.057 vs 0.17 pooled u650-u750) forked into the race world (solo 70 % half with boxes, 1v1 with self-copy opponents,
race33b's race settings) with --hard-contact-ends kept: a hard wall / box contact, or a car contact this car is responsible for
(--car-fault), ends the return. Left out vs race33b (user 11-26, end-to-end only): the opponent prediction head (--pred-*) and
the plan-forecast car cost (--plan-car-*, gamed by dg35a). Question: does the price that cut box hits 3x also cut at-fault car
contacts in 1v1 without slowing passes? Compare with race33b u975 on exam4 (at-fault 0.97 per 1v1, 0.6x opponent 1.84) and
keep exam3 (solo speed / boxes) against hc41a. ACE GPU 3, 1536 envs, updates 901..1200, checkpoints every 50.
Usage: python3 rq_race_hc.py args"""
import json
RUNS = "/home/shchon11/f1sim_runs"; SRC = "/home/shchon11/F1tenth_E2E/f1sim"
NAME, PARENT, OFFSET, N_UPD, EVERY, ENVS = "lg44a", RUNS + "/rc42a/plan_u950.pt", 950, 300, 50, 1536   # 11-27 10:00: from u750 now on ACE GPU 2 (user: "병렬로", GPU 2 free) instead of waiting for u900
LR0, LR1 = 2e-4, 1e-4
A = json.load(open(RUNS + "/race33b/args.json"))["args"]
SKIP = {"name", "dist_world", "dist_rank", "dist_init", "dist_backend", "dist_same_seed", "runs", "check_args", "init",
        "update_offset", "envs", "updates", "init_critic", "save_every", "lr", "lr_end", "world",
        "pred_opp", "pred_smooth", "pred_lr", "pred_range", "plan_car_cost", "plan_car_margin", "plan_car_w_end"}
base = []
for k, v in A.items():
    if k in SKIP or v is None or v is False or v == "":
        continue
    opt = "--" + k.replace("_", "-")
    base += [opt] if v is True else ([opt] + [str(x) for x in v] if isinstance(v, (list, tuple)) else [opt, str(v)])
U_LAST = OFFSET + N_UPD
f0 = OFFSET / (U_LAST - 1)
LR_ARG = (LR0 - LR1 * f0) / (1.0 - f0)
base += ["--world", SRC + "/f1sim/learn/configs/plan_world_korea26scan_race_ace.json", "--envs", str(ENVS),
         "--updates", str(U_LAST), "--update-offset", str(OFFSET), "--save-every", str(EVERY), "--init", PARENT, "--init-critic",
         "--lr", f"{LR_ARG:.6g}", "--lr-end", f"{LR1:g}", "--hard-contact-ends", "--car-lagrange", "0.1,0.5,30,3", "--name", NAME]
if __name__ == "__main__":
    print(" ".join(base))
