"""gru_probe.py 결과에서 결론에 쓰는 줄만 뽑아 한 장으로 (11-12). 인자: 그림.png 줄.npy [시드별 줄.npy ...]
가로 = '이 세션에 상자가 있다는 것'을 기억에서 읽어 낸 정도(상자 세션의 순간이 상자가 한 번도 없던 세션의 같은 자리·같은 시각보다 점수가 높은 비율,
0.5 = 구분 못 함), 지난 바퀴에 지나간 상자가 다시 보이기 0.5~1.0 s 전의 순간에서 잼. 막대 = ± 표준오차, 작은 점 = 다른 시드."""
import sys
import numpy as np
from rec_lib import plt

rows = {}
for i, fn in enumerate(sys.argv[2:]):
    for r in np.load(fn):
        rows.setdefault(str(r[0]), []).append((float(r[3]), float(r[4])))
G, B, O = "#52514e", "#2a78d6", "#eb6834"
GATE = "처음부터 보는 GRU, update gate bias를 1~30 s에 퍼지게"; DEF = "처음부터 보는 GRU, 기본 초기값"
SHOW = [("지금 policy의 기억 (plan25r update 300)", "policy가 들고 있던 기억(128)", G),
        ("기억 없이 지금 입력만", "지금 입력(292)", G),
        ("직접 가르친 GRU · 기본 초기값", f"(가) {DEF}", B),
        ("직접 가르친 GRU · 오래 붙들게 초기화", f"(가) {GATE}", O),
        ("앞으로 3 s 진행을 맞히게 가르친 GRU · 기본 초기값", f"(나) {DEF} · 기억 값을 직선으로 읽음", B),
        ("앞으로 3 s 진행을 맞히게 가르친 GRU · 오래 붙들게 초기화", f"(나) {GATE} · 기억 값을 직선으로 읽음", O),
        ("value(앞으로의 진행 − 접촉 벌)를 맞히게 가르친 GRU · 기본 초기값", f"(다) {DEF} · 기억 값을 직선으로 읽음", B),
        ("value(앞으로의 진행 − 접촉 벌)를 맞히게 가르친 GRU · 오래 붙들게 초기화", f"(다) {GATE} · 기억 값을 직선으로 읽음", O)]
fig, ax = plt.subplots(figsize=(11.5, 5.2), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
for i, (lab, key, c) in enumerate(SHOW):
    y = len(SHOW) - 1 - i; v = rows[key]
    ax.errorbar(v[0][0], y, xerr=v[0][1], fmt="o", color=c, ms=9, lw=2, capsize=0, zorder=3)
    for a, _ in v[1:]:
        ax.plot(a, y, "o", color=c, ms=5, alpha=0.55, zorder=2)
    best = max(a for a, _ in v)
    ax.text(best + max(v[0][1], 0.012) + 0.012, y, f"{v[0][0]:.2f}" + ("" if len(v) == 1 else " (다른 시드 " + ", ".join(f"{a:.2f}" for a, _ in v[1:]) + ")"),
            va="center", fontsize=11, color="#0b0b0b")
ax.axvline(0.5, color="#52514e", lw=1, ls="--", zorder=1); ax.text(0.493, len(SHOW) - 0.45, "구분 못 함", ha="right", fontsize=9, color="#52514e")
ax.set_yticks(range(len(SHOW))); ax.set_yticklabels([s[0] for s in SHOW][::-1], fontsize=11)
ax.set_xlim(0.4, 1.12); ax.set_ylim(-0.6, len(SHOW) - 0.1); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.grid(axis="x", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.set_xlabel("기억에서 '이 세션에 상자가 있다'를 읽어 낸 정도 (가르치지 않은 세션에서 채점; 0.5 = 구분 못 함, 1.0 = 늘 구분)", fontsize=10, color="#52514e")
fig.suptitle("policy와 같은 크기의 GRU에 같은 입력을 넣고 따로 가르치면, 세션 내내 '상자가 있다'를 들고 가는가\n"
             "기록으로 한 측정 · 지난 바퀴에 지나간 상자가 다시 보이기 0.5~1.0 s 전 순간 606개 · 파랑 = 기본 초기값, 주황 = 오래 붙들게 초기화",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=110); print("그림", sys.argv[1])
