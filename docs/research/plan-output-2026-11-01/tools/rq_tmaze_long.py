"""(11-26 23:00) RQ4 follow-up: does PPO learn a memory-conditioned approach with LONGER training? tm39a (control, forked
from scan31b2 u600) is continued from its u150 for 450 more updates (lr 1e-4 flat) as tm39a2 on CUDA 1; tmaze exams +
tmaze_measure at its u150 / u300 / u450 (= tm39a u300 / u450 / u600). At u150 tm39a holds the side in memory (0.99 bit)
yet approaches at ~3.5 m/s for every b (memoryless caution)."""
import json, os, subprocess, sys, time
exec(open(os.path.join(os.path.dirname(__file__), "rq_tmaze_arms2.py")).read().split("def log(")[0])
argv = list(base)
i = argv.index("--init"); argv[i + 1] = RUNS + "/tm39a/plan_u150.pt"
argv[argv.index("--updates") + 1] = "450"
argv[argv.index("--save-every") + 1] = "50"
argv += ["--lr", "1e-4", "--lr-end", "1e-4", "--name", "tm39a2"]
def log(m):
    with open(OUT + "/chain.log", "a") as f: f.write(time.strftime("%H:%M:%S ") + m + "\n")
log("train tm39a2 (tm39a u150 + 450) on CUDA 1")
with open(OUT + "/tm39a2.log", "w") as fh:
    r = subprocess.run([PY, "-u", "-m", "f1sim.learn.plan_rl"] + argv, cwd=SRC, stdout=fh, stderr=subprocess.STDOUT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"))
log(f"train tm39a2 exit {r.returncode}")
for u in (150, 300, 450):
    ck = f"{RUNS}/tm39a2/plan_u{u}.pt"
    if not os.path.exists(ck): log(f"exam tm39a2 u{u}: none"); continue
    npz = f"{OUT}/exam_tmaze_tm39a2_u{u}_n384.npz"
    with open(f"{OUT}/exam_tm39a2_u{u}.log", "w") as fh:
        subprocess.run([PY, "exam_record.py", ck, npz, "ideal", "384", "tmaze", "32", "999"], cwd=T, stdout=fh,
                       stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="1", F1SIM_EXAM_WORLD=WORLD))
    with open(f"{OUT}/measure_tm39a2_u{u}.txt", "w") as fh:
        subprocess.run([PY, T + "/tmaze_measure.py", npz], stdout=fh, stderr=subprocess.STDOUT)
    log(f"exam tm39a2 u{u} done")
log("chain long done")
