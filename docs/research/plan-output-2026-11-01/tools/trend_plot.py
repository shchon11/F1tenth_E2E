"""학습 추이 그림 (11-09, 사용자: "초반이라 추이 확인하는게 엄청 중요해보여서 시각화등등 분석 많이 해봐"). progress.jsonl(update마다 한 줄)을
그림 한 장으로: 바퀴/분, 속도, 접촉(벽·상자·차), 보상, KL·잘림, 탐색 표준편차(평균과 첫 단계 속도 출력), 붙잡은 결정 비율, 상자 치운 판,
critic 오차, 속도 짐작 오차. 비교 run(점선)은 같은 축에 원래 update 번호를 맞춰서.
인자: run 이름 출력png [비교 run 이름 [비교 run 시작 update 차이]]"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
R = os.path.expanduser("~/f1sim_runs")


def load(name):
    rows = [json.loads(l) for l in open(os.path.join(R, name, "progress.jsonl")) if l.strip()]
    return rows


def col(rows, k, f=None):
    out = []
    for r in rows:
        v = r.get(k)
        if v is None:
            out.append(np.nan)
        else:
            out.append(f(v) if f else v)
    return np.array(out, float)


def smooth(y, n=5):
    if len(y) < n:
        return y
    k = np.ones(n) / n
    yp = np.concatenate([np.full(n // 2, y[0]), y, np.full(n - 1 - n // 2, y[-1])])
    return np.convolve(yp, k, "valid")


run, out = sys.argv[1], sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 else ""
ref_off = int(sys.argv[4]) if len(sys.argv) > 4 else 0
A = load(run)
B = load(ref)[-150:] if ref else []
ua = col(A, "update")
ub = col(B, "update") - ref_off if B else None
panels = [
    ("바퀴 / 분 (학습 차 평균)", lambda rows: col(rows, "laps_per_min")),
    ("속도 [m/s]", lambda rows: col(rows, "speed")),
    ("접촉 / 분: 벽", lambda rows: col(rows, "contacts_wall_per_min")),
    ("접촉 / 분: 상자", lambda rows: col(rows, "contacts_prop_per_min")),
    ("접촉 / 분: 차", lambda rows: col(rows, "contacts_car_per_min")),
    ("보상 / 결정", lambda rows: col(rows, "reward")),
    ("KL (policy 변화 폭)", lambda rows: col(rows, "kl")),
    ("잘림 비율 (PPO clip)", lambda rows: col(rows, "clipfrac")),
    ("탐색 표준편차: 평균 / 첫 단계 속도", None),
    ("붙잡은 결정 비율", lambda rows: col(rows, "held_share")),
    ("상자 치운 판 / update", lambda rows: col(rows, "removed")),
    ("critic 오차 / 속도 짐작 오차 [m/s]", None),
]
fig, axs = plt.subplots(3, 4, figsize=(20, 11))
for ax, (title, fn) in zip(axs.flat, panels):
    if title.startswith("탐색"):
        sa = col(A, "std", lambda v: float(np.mean(v))); s1 = col(A, "std", lambda v: float(v[10]) if len(v) > 10 else np.nan)
        ax.plot(ua, sa, color="tab:blue", lw=1, label="평균"); ax.plot(ua, s1, color="tab:red", lw=1.5, label="첫 단계 속도")
        if B:
            ax.plot(ub, col(B, "std", lambda v: float(np.mean(v))), color="tab:blue", lw=1, ls=":")
            ax.plot(ub, col(B, "std", lambda v: float(v[10]) if len(v) > 10 else np.nan), color="tab:red", lw=1, ls=":")
        ax.legend(fontsize=8)
    elif title.startswith("critic"):
        ax.plot(ua, col(A, "vloss"), color="tab:purple", lw=1, label="critic 오차")
        ax2 = ax.twinx(); ax2.plot(ua, col(A, "vest_mae"), color="tab:green", lw=1, label="속도 짐작 오차"); ax2.set_ylabel("속도 짐작 [m/s]", fontsize=8)
        ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="upper right")
    else:
        ya = fn(A)
        ax.plot(ua, ya, color="tab:blue", lw=0.6, alpha=0.4)
        ax.plot(ua, smooth(ya), color="tab:blue", lw=1.8, label=run)
        if B:
            yb = fn(B)
            ax.plot(ub, smooth(yb), color="0.4", lw=1.2, ls=":", label=f"{ref} (마지막 {len(B)} update)")
        if title.startswith("바퀴"):
            ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlabel("update", fontsize=8)
fig.suptitle(f"{run} 학습 추이 (update {int(np.nanmax(ua))}까지, 굵은 선 = 5 update 평균, 점선 = {ref or '없음'})", fontsize=13)
plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(out, dpi=70)
print(out, flush=True)
