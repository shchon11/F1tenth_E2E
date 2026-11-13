"""How alike are the pushes two independent rollouts give the actor?  (11-13, structure audit; input: grad_dump.sh)

Every rollout's policy gradient (plan_rl --dump-rollout: g<N>.npz, kept apart for the steering mean, the speed mean and the
noise sizes) is one sample of "signal + rollout noise". For two independent rollouts the cosine between their gradients is
|signal|^2 / (|signal|^2 + |noise|^2): the share of one rollout's push that is signal. Printed per part and per parameter
group, for every dump given, with the cosine between the dumps' mean gradients (do they push the same way).

    python3 grad_cos.py OUT.txt NAME=DUMP_DIR [NAME=DUMP_DIR ...] [first update, default 4]
"""
import sys, glob, os, itertools
import numpy as np

LOG = open(sys.argv[1], "w")
first = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 4
runs = [a.split("=") for a in sys.argv[2:] if "=" in a]


def say(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


def group(name):
    return {"enc": "LiDAR 인코더", "gru": "기억(GRU)", "trunk": "몸통", "pi": "plan 출력 층", "log_std": "잡음 크기"}.get(name.split(".")[0], "그 밖")


PARTS = (("steer", "조향 평균"), ("speed", "속도 평균"), ("std", "잡음 크기"))
MEAN = {}
for name, d in runs:
    fs = sorted(glob.glob(os.path.join(d, "g*.npz")), key=lambda f: int(os.path.basename(f)[1:-4]))
    fs = [f for f in fs if int(os.path.basename(f)[1:-4]) >= first]
    Z = [np.load(f) for f in fs]
    names, sizes = [str(n) for n in Z[0]["names"]], Z[0]["sizes"]
    gid = np.repeat(np.array([group(n) for n in names]), sizes)
    say(f"\n=== {name}: rollout {len(Z)}개 (쌍 {len(Z) * (len(Z) - 1) // 2}개), actor 가중치 {sizes.sum():,}개 ===")
    say("  부분 / 가중치 묶음 / 서로 다른 rollout 둘의 gradient cosine 평균 ± 표준오차 (= 한 rollout의 gradient 가운데 신호의 몫) / gradient 크기 중앙 / 평균 gradient의 크기")
    for key, kn in PARTS:
        G = np.stack([z[key].astype(np.float64) for z in Z])
        MEAN[name, key] = G.mean(0)
        for gname in ["전체"] + sorted(set(gid)):
            m = slice(None) if gname == "전체" else gid == gname
            g = G[:, m]; nr = np.linalg.norm(g, axis=1)
            if np.median(nr) == 0:
                continue
            cs = np.array([g[i] @ g[j] / (nr[i] * nr[j]) for i, j in itertools.combinations(range(len(g)), 2)])
            # the pairs share rollouts, so the error is taken over rollouts left out one at a time
            jk = []
            for o in range(len(g)):
                idx = [i for i in range(len(g)) if i != o]
                jk.append(np.mean([g[i] @ g[j] / (nr[i] * nr[j]) for i, j in itertools.combinations(idx, 2)]))
            se = np.sqrt((len(g) - 1) / len(g) * ((np.array(jk) - np.mean(jk)) ** 2).sum())
            say(f"  {kn:<6} {gname:<14} {cs.mean():+.3f} ±{se:.3f}   {np.median(nr):.4f}   {np.linalg.norm(g.mean(0)):.4f}")
if len(runs) > 1:
    say("\n=== 서로 다른 잡음 방식의 평균 gradient가 같은 쪽을 가리키나 (cosine) ===")
    for (a, _), (b, _) in itertools.combinations(runs, 2):
        say(f"  {a} <-> {b}: " + " · ".join(f"{kn} {MEAN[a, k] @ MEAN[b, k] / (np.linalg.norm(MEAN[a, k]) * np.linalg.norm(MEAN[b, k])):+.3f}" for k, kn in PARTS))
