"""이어지는 조향 잡음(--noise-corr ... --noise-corr-steer)으로 모은 batch에서 (11-15 01:00, 사용자: "뭐 달라진게 없냐"):
(1) 그 결정의 조향 잡음(이어진 값)이 상자 통과 틈(중심 거리)을 얼마나 바꾸나 -- 흰 잡음 plan26l u150: 0.0-0.4 cm/표준편차
(2) advantage가 "상자 반대쪽" 조향 잡음에 대해 갖는 기울기 -- 흰 잡음 plan26l u150: 0 +- 0.01 s/도
행끼리 이어져 있어 한데 모은 표준오차는 너무 작으므로(11-13), rollout마다 기울기를 따로 구해 그 평균 +- rollout 사이 표준오차로 봄.
인자: DUMP_DIR [비교 DUMP_DIR ...]"""
import sys
import numpy as np

src = open(__file__.replace("batch_held.py", "batch_expcost.py")).read()
_argv = sys.argv; sys.argv = [_argv[0], _argv[1]]               # batch_expcost reads DIR [U0] at the top
exec(src[:src.index("files = sorted(")])                       # load(), gae(), BINS, T_REF from batch_expcost.py
sys.argv = _argv
import glob, os


def per_rollout(D, ykey, lo, hi):
    out = []
    for d in D:
        m = d["has"] & d["full"] & (d["dist"] >= lo) & (d["dist"] < hi) & (d["dmin"] < 0.8) & (d["side"] != 0)
        if m.sum() < 30:
            continue
        x = (d["noise"] * -d["side"])[m]; y = (d["adv0"] * T_REF if ykey == "adv" else d["dmin"] * 100)[m]
        x = x - x.mean(); y = y - y.mean()
        out.append(((x * y).sum() / (x * x).sum(), x.std()))
    if len(out) < 3:
        return None
    b = np.array([o[0] for o in out]); sd = np.mean([o[1] for o in out])
    return b.mean(), b.std(ddof=1) / np.sqrt(len(b)), sd, len(b)


for DIR in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(DIR, "u*.npz")), key=lambda p: int(os.path.basename(p)[1:-4]))[2:]
    D = [load(f) for f in files]
    print(f"== {DIR.rstrip('/').split('/')[-1]}: rollout {len(D)}개 (rollout마다 기울기, 평균 ± rollout 사이 표준오차)")
    print(f"  {'상자까지':>10}  {'통과 틈 변화 [cm/도]':>26}  {'잡음 1 표준편차가 바꾸는 틈':>16}  {'advantage [s/도]':>24}")
    for lo, hi in BINS:
        g = per_rollout(D, "gap", lo, hi); a = per_rollout(D, "adv", lo, hi)
        if g is None or a is None:
            continue
        print(f"  {lo:>4.1f}-{hi:<4.1f} m  {g[0]:+.2f} ± {g[1]:.2f} ({g[0] / g[1]:+.1f})      {abs(g[0]) * g[2]:6.1f} cm        "
              f"{a[0]:+.4f} ± {a[1]:.4f} ({a[0] / a[1]:+.1f})")
