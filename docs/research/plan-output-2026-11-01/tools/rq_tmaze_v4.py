"""(11-27) RQ4/RQ5 PPO control arm on the racing T-maze v4 (configs/plan_world_korea26scan_tmaze4.json): does PPO learn a
memory-conditioned approach where anticipation is worth it (the nudge test, tools/tmaze_nudge.py, measured the gain on this world
before any training)? One arm, tm40a: a FORK of scan31b2 u600 with the SAME reward as the parent -- the parent's args as
rq_tmaze_arms2.py builds them (416 envs, scene replay off, the T-maze world) plus --update-offset 600, so every update-indexed
schedule continues from 600 instead of restarting at 1. 11-27: the v3 arms (tm39a/b/c) restarted the count, so --shaping-fade
100,150 put the distance shaping back at weight 1 for their first 100 updates (the parent at u600 runs with 0) -- their reward was
not the parent's.
  updates 601..900 (300), checkpoints every 50 (u650 ... u900), then a tmaze exam (384 envs = 64 sessions per b, ideal, 32 s) +
  tmaze_measure.py of each checkpoint, outputs in out/rq/ (tm40a.log, exam_tmaze4_tm40a_u*_n384.npz, measure_tm40a_u*.txt).
  lr: tm39a's effective schedule, 2e-4 at the first update -> 1e-4 at the last. plan_rl interpolates lr -> lr_end over
  (upd - 1) / (updates - 1), which with the offset starts at frac 600/899 -- so the --lr given is the one that makes upd 601 run
  at 2e-4 (printed below); passing 2e-4 literally would run 1.33e-4 -> 1e-4.
  The critic: as rq_tmaze_arms2.py -- NOT the parent's (init_critic is in its SKIP list and never added back: tm39a/b/c started
  with a fresh critic, only the scan encoder copied). INIT_CRITIC = True makes the fork a true continuation (plan_rl loads the
  parent's critic when its inputs match and says so in the log).
Usage: python3 rq_tmaze_v4.py check      -- plan_rl --check-args with the exact argv (nothing built, no GPU)
       python3 rq_tmaze_v4.py GPU        -- train on CUDA_VISIBLE_DEVICES=GPU, then the exams (launch with setsid)"""
import json, os, subprocess, sys, time

R = "/home/shchon11/F1tenth_E2E"; SRC = R + "/f1sim"; PY = R + "/.venv/bin/python"
T = R + "/docs/research/plan-output-2026-11-01/tools"; OUT = R + "/docs/research/plan-output-2026-11-01/out/rq"
WORLD = SRC + "/f1sim/learn/configs/plan_world_korea26scan_tmaze4.json"; RUNS = "/home/shchon11/f1sim_runs"
NAME, PARENT, OFFSET, N_UPD, EVERY, ENVS = "tm40a", RUNS + "/scan31b2/plan_u600.pt", 600, 300, 50, 416
LR0, LR1 = 2e-4, 1e-4                          # the effective lr at the first / last update (tm39a's)
INIT_CRITIC = False                            # rq_tmaze_arms2.py's choice (see the docstring)
A = json.load(open(RUNS + "/scan31b2/args.json"))["args"]
SKIP = {"name", "dist_world", "dist_rank", "dist_init", "dist_backend", "dist_same_seed", "runs", "check_args", "init",
        "update_offset", "envs", "updates", "init_critic", "self_opp_init", "world", "save_every", "scene_replay", "lr", "lr_end"}
base = []
for k, v in A.items():
    if k in SKIP or v is None or v is False:
        continue
    opt = "--" + k.replace("_", "-")
    base += [opt] if v is True else ([opt] + [str(x) for x in v] if isinstance(v, (list, tuple)) else [opt, str(v)])
U_LAST = OFFSET + N_UPD
f0 = OFFSET / (U_LAST - 1)                     # plan_rl's frac at the first update (upd - 1) / (updates - 1)
LR_ARG = (LR0 - LR1 * f0) / (1.0 - f0)
base += ["--world", WORLD, "--envs", str(ENVS), "--updates", str(U_LAST), "--update-offset", str(OFFSET),
         "--save-every", str(EVERY), "--scene-replay", "0", "--init", PARENT, "--lr", f"{LR_ARG:.6g}", "--lr-end", f"{LR1:g}",
         "--name", NAME] + (["--init-critic"] if INIT_CRITIC else [])


def lr_at(upd):
    frac = (upd - 1) / max(1, U_LAST - 1)
    return LR_ARG + (LR1 - LR_ARG) * frac


def shaping_at(upd):
    f0_, f1_ = (float(x) for x in A["shaping_fade"].split(","))
    return 1.0 - min(1.0, max(0.0, (upd - f0_) / max(1.0, f1_ - f0_)))


def log(msg):
    with open(OUT + "/chain.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


if __name__ == "__main__":
    print(f"{NAME}: {PARENT} -> updates {OFFSET + 1}..{U_LAST}, lr {lr_at(OFFSET + 1):.3g} -> {lr_at(U_LAST):.3g} (--lr {LR_ARG:.6g}), "
          f"distance shaping weight {shaping_at(OFFSET + 1):g} (parent at u600: {shaping_at(600):g}), critic "
          f"{'the parent' if INIT_CRITIC else 'fresh (arms2)'}, world {os.path.basename(WORLD)}", flush=True)
    if sys.argv[1] == "check":
        r = subprocess.run([PY, "-m", "f1sim.learn.plan_rl"] + base + ["--check-args"], cwd=SRC, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"check-args FAILED: {r.stdout[-2000:]} {r.stderr[-2000:]}")
        got = json.loads(r.stdout.strip().splitlines()[-1])["args"]
        for k in ("world", "envs", "updates", "update_offset", "save_every", "scene_replay", "init", "init_critic", "lr", "lr_end",
                  "shaping_fade", "name"):
            print(f"  {k} = {got[k]}")
        assert os.path.isfile(got["world"]) and json.load(open(got["world"])).get("tmaze"), got["world"]
        print("check-args OK:\n" + " ".join(["cd", SRC, "&&", "CUDA_VISIBLE_DEVICES=<gpu>", PY, "-u", "-m", "f1sim.learn.plan_rl"] + base))
        sys.exit(0)
    gpu = int(sys.argv[1])
    log(f"train {NAME} (scan31b2 u600 + {N_UPD}, --update-offset {OFFSET}, T-maze v4) on CUDA {gpu}")
    with open(OUT + f"/{NAME}.log", "w") as fh:
        r = subprocess.run([PY, "-u", "-m", "f1sim.learn.plan_rl"] + base, cwd=SRC, stdout=fh, stderr=subprocess.STDOUT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
    log(f"train {NAME} exit {r.returncode}")
    for u in range(OFFSET + EVERY, U_LAST + 1, EVERY):
        ck = f"{RUNS}/{NAME}/plan_u{u}.pt"
        if not os.path.exists(ck):
            log(f"exam {NAME} u{u}: no checkpoint"); continue
        npz = f"{OUT}/exam_tmaze4_{NAME}_u{u}_n384.npz"
        with open(f"{OUT}/exam_{NAME}_u{u}.log", "w") as fh:
            subprocess.run([PY, "exam_record.py", ck, npz, "ideal", "384", "tmaze", "32", "999"], cwd=T, stdout=fh,
                           stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), F1SIM_EXAM_WORLD=WORLD))
        with open(f"{OUT}/measure_{NAME}_u{u}.txt", "w") as fh:
            subprocess.run([PY, T + "/tmaze_measure.py", npz], stdout=fh, stderr=subprocess.STDOUT)
        log(f"exam {NAME} u{u} done")
    log(f"chain {NAME} done")
