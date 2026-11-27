"""(11-27, user OK "ㅇㅋ") hs43a: the qualifier FROM SCRATCH with the hit price that hc41a showed as a fork (scan31b2 u600 +
--hard-contact-ends: box hard per box session 0.062 pooled u650-u850 vs the scan31b2 line's 0.17, lap 7.1-7.2 s). Question: is
that cut the price's own, or does it need a policy already trained without it (hc41a only fine-tuned scan31b2)? Settings = the
qualifier's final recipe (scan31b2's args: steer std floor 0.3, no memory carry-over) with scan31a's scratch-only schedule
(correlated noise ramp at u200-250) and --hard-contact-ends from update 1; ACE GPU 3, 1536 envs, lr schedule over 1500 updates as
scan31a, checkpoints every 50. Prediction: by u450-u600 (where scan31b/scan31b2 were judged) box hard <= 0.08 per box session,
empty lap <= 7.2 s. Falsifier: box hard >= 0.15 there (the price alone does not carry), or lap > 7.4 s / stopped > 2x.
usage: python3 rq_hs_scratch.py args"""
import json
RUNS = "/home/shchon11/f1sim_runs"; SRC = "/home/shchon11/F1tenth_E2E/f1sim"
NAME, ENVS, UPD, EVERY = "hs43a", 1536, 1500, 50
A = json.load(open(RUNS + "/scan31b2/args.json"))["args"]
SKIP = {"name", "dist_world", "dist_rank", "dist_init", "dist_backend", "dist_same_seed", "runs", "check_args", "init",
        "update_offset", "envs", "updates", "init_critic", "self_opp_init", "save_every", "world", "noise_corr_ramp"}
base = []
for k, v in A.items():
    if k in SKIP or v is None or v is False or v == "":
        continue
    opt = "--" + k.replace("_", "-")
    base += [opt] if v is True else ([opt] + [str(x) for x in v] if isinstance(v, (list, tuple)) else [opt, str(v)])
base += ["--world", SRC + "/f1sim/learn/configs/plan_world_korea26scan_qual_ace.json", "--envs", str(ENVS),
         "--updates", str(UPD), "--save-every", str(EVERY), "--noise-corr-ramp", "200,250", "--hard-contact-ends", "--name", NAME]
if __name__ == "__main__":
    print(" ".join(base))
