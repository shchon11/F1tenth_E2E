"""학습 추이 그림 (11-09, 사용자: "초반이라 추이 확인하는게 엄청 중요해보여서 시각화등등 분석 많이 해봐"). progress.jsonl(update마다 한 줄)을
그림 한 장으로: 바퀴/분, 속도, 접촉(벽·상자·차), 보상, 박는 plan 비율, plan 뒷부분(다음 plan과 0.5초 차이), 탐색 표준편차(평균과 첫 단계 속도
출력: 강화학습 행동인 첫 단계 조향·속도 두 개, 속도는 run의 --speed-std-floor 하한을 적용한 실제 값), KL, 상자 치운 판, critic 오차.
--chain 시작run (11-09, 사용자: "아니 씨발 지금 그럼 세개를 학습하고 있는거야?"): 지금 run이 이어 받은 checkpoint(args.json의 init)를 시작 run까지
거슬러 한 줄로 이어 붙임 -- x = 이어 온 update 수, 단계가 바뀐 곳에 세로 점선 + 그 단계에서 바꾼 것.
인자: run 이름 출력png --chain 시작run   (예전 방식: run 이름 출력png [비교 run 이름 [비교 run 시작 update 차이]])"""
import os, re, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
from run_chain import R, STAGE_NOTE, chain, run_args


def load(name):
    rows = [json.loads(l) for l in open(os.path.join(R, name, "progress.jsonl")) if l.strip()]
    return rows


def std_floor(name):
    """the first-step speed output's exploration floor this run trained with (args.speed_std_floor; the logged std is before it)"""
    return float(run_args(name).get("speed_std_floor") or 0.0)


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
if len(sys.argv) > 4 and sys.argv[3] == "--chain":
    segs = []
    off = 0
    for name, cut in chain(run, sys.argv[4]):
        rows = [r for r in load(name) if cut is None or r["update"] <= cut]
        if not rows:
            continue
        segs.append((name, off, rows))
        off += cut if cut is not None else rows[-1]["update"]
    ref = None
else:
    segs = [(run, 0, load(run))]
    ref = sys.argv[3] if len(sys.argv) > 3 else ""
    ref_off = int(sys.argv[4]) if len(sys.argv) > 4 else 0
panels = [
    ("바퀴 / 분 (학습 차 평균)", lambda rows: col(rows, "laps_per_min")),
    ("속도 [m/s]", lambda rows: col(rows, "speed")),
    ("접촉 / 분: 벽", lambda rows: col(rows, "contacts_wall_per_min")),
    ("접촉 / 분: 상자", lambda rows: col(rows, "contacts_prop_per_min")),
    ("접촉 / 분: 차", lambda rows: col(rows, "contacts_car_per_min")),
    ("보상 / 결정 (단계가 바뀌면 보상 정의도 바뀔 수 있음)", lambda rows: col(rows, "reward")),
    ("박는 plan 비율 [%]", lambda rows: col(rows, "plan_hit_share") * 100),
    ("plan 뒷부분: 다음 plan과 0.5초 차이 [cm]", lambda rows: col(rows, "tail_next_05") * 100),
    ("탐색 표준편차: 첫 단계 조향 / 첫 단계 속도 (하한 적용)", None),
    ("KL (policy 변화 폭)", lambda rows: col(rows, "kl")),
    ("상자 치운 판 / update", lambda rows: col(rows, "removed")),
    ("critic 오차", lambda rows: col(rows, "vloss")),
]
fig, axs = plt.subplots(3, 4, figsize=(20, 11))
for pi, (ax, (title, fn)) in enumerate(zip(axs.flat, panels)):
    for si, (name, off, rows) in enumerate(segs):
        x = col(rows, "update") + off
        if fn is None:
            s0 = col(rows, "std", lambda v: float(v[0]))
            s1 = np.maximum(col(rows, "std", lambda v: float(v[10]) if len(v) > 10 else np.nan), std_floor(name))
            ax.plot(x, s0, color="tab:blue", lw=1.5, label="첫 단계 조향" if si == 0 else None)
            ax.plot(x, s1, color="tab:red", lw=1.5, label="첫 단계 속도" if si == 0 else None)
        else:
            y = fn(rows)
            ax.plot(x, y, color="tab:blue", lw=0.6, alpha=0.4)
            ax.plot(x, smooth(y), color="tab:blue", lw=1.8)
        if si > 0:
            ax.axvline(off, color="0.3", ls="--", lw=1)
        if pi < 4 and len(segs) > 1:      # stage names along the top row
            ax.text(off + 1, 1.0, f" {name}", transform=ax.get_xaxis_transform(), fontsize=8, color="0.3", va="top", ha="left")
    if fn is None:
        ax.legend(fontsize=8)
    if ref and fn is not None:
        B = load(ref)[-150:]
        ax.plot(col(B, "update") - ref_off, smooth(fn(B)), color="0.4", lw=1.2, ls=":", label=f"{ref} (마지막 {len(B)} update)")
        if pi == 0:
            ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlim(0, segs[-1][1] + col(segs[-1][2], "update")[-1] + 2)     # same x range in every panel
    ax.set_xlabel("이어 온 update" if len(segs) > 1 else "update", fontsize=8)
last = segs[-1]
upto = int(col(last[2], "update")[-1])
if len(segs) > 1:
    stages = "  →  ".join(f"{n} ({o + 1}-{o + int(col(r, 'update')[-1])}: {STAGE_NOTE.get(n, '')})" for n, o, r in segs)
    fig.suptitle(f"지금 학습 {run} 추이 (update {upto}까지, 굵은 선 = 5 update 평균, 세로 점선 = 단계가 바뀐 곳)\n{stages}", fontsize=12)
else:
    fig.suptitle(f"{run} 학습 추이 (update {upto}까지, 굵은 선 = 5 update 평균, 점선 = {ref or '없음'})", fontsize=13)
plt.tight_layout(rect=(0, 0, 1, 0.94)); plt.savefig(out, dpi=70)
print(out, flush=True)
