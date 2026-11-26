"""RQ2 (11-26): 보상 항목끼리 gradient가 상쇄되나, 신호가 모자라나 -- 얼린 checkpoint에서 plan_rl --term-split을 그 run의 인자·세계
그대로 돌림 (학습 안 함). 한 프로세스에서 rollout 크기는 그 run의 update와 같게: 차 = 모든 rank의 --envs 합, 결정 = 그 update의
horizon(progress.jsonl), minibatch = --minibatches x (horizon / --horizon). 결과는 term_split.measure가 OUT.txt / OUT.npz로.
인자: run 폴더 update OUT.txt [warm-up rollout 수, 기본 1] [value 맞춤 step, 기본 400]
환경: F1SIM_SRC = 돌릴 코드의 f1sim 폴더 (기본: 이 파일 옆 repo의 f1sim), CUDA_VISIBLE_DEVICES는 부르는 쪽이.
RQ2_SMALL="차,결정,minibatch": 경로만 빨리 확인 (측정 아님)."""
import os, sys, json, glob, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("F1SIM_SRC", os.path.normpath(os.path.join(HERE, "../../../../f1sim")))
run, upd, out = sys.argv[1], int(sys.argv[2]), os.path.abspath(sys.argv[3])
warm = int(sys.argv[4]) if len(sys.argv) > 4 else 1
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 400
A = json.load(open(os.path.join(run, "args.json")))["args"]
envs = int(A["envs"]) + sum(int(json.load(open(f))["args"]["envs"]) for f in sorted(glob.glob(os.path.join(run, "rank*", "args.json"))))
hor = None
for line in open(os.path.join(run, "progress.jsonl")):
    r = json.loads(line)
    if int(r["update"]) == upd:
        hor = int(r["horizon"])
assert hor is not None, f"update {upd} not in {run}/progress.jsonl"
ck = os.path.join(run, f"plan_u{upd}.pt")
assert os.path.isfile(ck), ck
mb = int(A["minibatches"]) * max(1, hor // int(A["horizon"]))
if os.environ.get("RQ2_SMALL"):                                   # a quick check of the path: "ENVS,HORIZON,MINIBATCHES"
    envs, hor, mb = (int(x) for x in os.environ["RQ2_SMALL"].split(","))
SKIP = {"name", "dist_world", "dist_rank", "dist_init", "dist_backend", "dist_same_seed", "runs", "check_args", "init",
        "update_offset", "envs", "horizon", "horizon_auto", "horizon_auto_min", "horizon_auto_down", "minibatches", "updates",
        "init_critic", "self_opp_init", "term_split", "term_split_warm", "term_split_steps", "live_tap", "live_taps"}   # no live view
argv = []
for k, v in A.items():
    if k in SKIP or v is None or v is False:
        continue
    opt = "--" + k.replace("_", "-")
    if v is True:
        argv.append(opt)
    elif isinstance(v, (list, tuple)):
        argv += [opt] + [str(x) for x in v]
    else:
        argv += [opt, str(v)]
name = f"rq2_{os.path.basename(run.rstrip('/'))}_u{upd}"
runs = os.path.join(os.path.dirname(out), ".rq2_runs")
argv += ["--name", name, "--runs", runs, "--init", ck, "--init-critic", "--update-offset", str(upd), "--envs", str(envs),
         "--horizon", str(hor), "--horizon-auto", "1", "--minibatches", str(mb), "--term-split", out,
         "--term-split-warm", str(warm), "--term-split-steps", str(steps)]
print(f"[rq2] {ck}: 차 {envs}대 x {hor}결정, minibatch {mb}개, warm-up {warm}번 -> {out}", flush=True)
print("[rq2] python -m f1sim.learn.plan_rl " + " ".join(argv), flush=True)
r = subprocess.run([sys.executable, "-m", "f1sim.learn.plan_rl"] + argv, cwd=SRC)
import shutil
shutil.rmtree(os.path.join(runs, name), ignore_errors=True)        # the run folder holds only its args and an empty log
sys.exit(r.returncode)
