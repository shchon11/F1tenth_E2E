"""(11-26) RQ4/RQ6 racing T-maze arms: scan31b2's recipe from scratch in configs/plan_world_korea26scan_tmaze.json, 256 envs,
150 updates, checkpoints every 25; per GPU a sequential chain (train ... then tmaze exams + tmaze_measure at u50/u100/u150).
  GPU0 (4070): tm38a (control) -> tm38b (--contact-cond 0.3,10) -> exams of tm38b
  GPU1 (5060): tm38c (--intent 4,15,0.1)            -> exams of tm38a (after it is done), tm38c
At most two heavy processes at once (one per GPU); outputs in out/rq/."""
import json, os, subprocess, sys, time
R = "/home/shchon11/F1tenth_E2E"; SRC = R + "/f1sim"; PY = R + "/.venv/bin/python"
T = R + "/docs/research/plan-output-2026-11-01/tools"; OUT = R + "/docs/research/plan-output-2026-11-01/out/rq"
WORLD = SRC + "/f1sim/learn/configs/plan_world_korea26scan_tmaze.json"; RUNS = "/home/shchon11/f1sim_runs"
A = json.load(open(RUNS + "/scan31b2/args.json"))["args"]
SKIP = {"name", "dist_world", "dist_rank", "dist_init", "dist_backend", "dist_same_seed", "runs", "check_args", "init",
        "update_offset", "envs", "updates", "init_critic", "self_opp_init", "world", "save_every", "scene_replay"}
base = []
for k, v in A.items():
    if k in SKIP or v is None or v is False:
        continue
    opt = "--" + k.replace("_", "-")
    base += [opt] if v is True else ([opt] + [str(x) for x in v] if isinstance(v, (list, tuple)) else [opt, str(v)])
base += ["--world", WORLD, "--envs", "256", "--updates", "150", "--save-every", "25", "--scene-replay", "0"]
ARMS = {"tm38a": [], "tm38b": ["--contact-cond", "0.3,10"], "tm38c": ["--intent", "4,15,0.1"]}

def log(msg):
    with open(OUT + "/chain.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")

def train(name, gpu):
    log(f"train {name} on CUDA {gpu}")
    with open(OUT + f"/{name}.log", "w") as fh:
        r = subprocess.run([PY, "-u", "-m", "f1sim.learn.plan_rl"] + base + ["--name", name] + ARMS[name], cwd=SRC,
                           stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
    log(f"train {name} exit {r.returncode}")

def exams(name, gpu):
    for u in (50, 100, 150):
        ck = f"{RUNS}/{name}/plan_u{u}.pt"
        if not os.path.exists(ck):
            log(f"exam {name} u{u}: no checkpoint"); continue
        npz = f"{OUT}/exam_tmaze_{name}_u{u}_n384.npz"
        with open(f"{OUT}/exam_{name}_u{u}.log", "w") as fh:
            subprocess.run([PY, "exam_record.py", ck, npz, "ideal", "384", "tmaze", "32", "999"], cwd=T, stdout=fh,
                           stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), F1SIM_EXAM_WORLD=WORLD))
        with open(f"{OUT}/measure_{name}_u{u}.txt", "w") as fh:
            subprocess.run([PY, T + "/tmaze_measure.py", npz], stdout=fh, stderr=subprocess.STDOUT)
        log(f"exam {name} u{u} done")

gpu = int(sys.argv[1])
if gpu == 0:
    train("tm38a", 0); open(OUT + "/tm38a.done", "w").close()
    train("tm38b", 0); exams("tm38b", 0)
else:
    train("tm38c", 1)
    while not os.path.exists(OUT + "/tm38a.done"):
        time.sleep(60)
    exams("tm38a", 1); exams("tm38c", 1)
log(f"chain GPU{gpu} done")
