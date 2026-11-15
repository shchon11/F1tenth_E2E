"""(11-15) MAP 변형별: plan 옆가속 구간마다 0.4 s 뒤 plan 대비 바깥 밀림(중앙, 10-90 % 띠) + 벽 세게/세션 막대.
인자: 출력png 이름=기록npz ..."""
import sys
import numpy as np
from rec_lib import plt


def rows(fn):
    z = np.load(fn)
    st, pl, rs = z["st"], z["pl"], z["rs"]
    c = list(z["cols"]); hw = z["cnt"][:, :, c.index("hard_wall")]
    T = pl.shape[0]; K = 4; out = []
    for t in range(T - K - 1):
        ok = ~rs[t:t + K + 1].any(0)
        pp = np.concatenate([st[t, :, None, :2], pl[t, :, :K]], 1)
        seg = pp[:, 1:] - pp[:, :-1]
        vpl = np.linalg.norm(seg, axis=-1) / 0.1
        tang = seg / np.maximum(np.linalg.norm(seg, axis=-1, keepdims=True), 1e-6)
        d = st[t + K, :, :2] - pl[t, :, K - 1]
        crs = tang[:, -1, 0] * d[:, 1] - tang[:, -1, 1] * d[:, 0]
        h = np.arctan2(seg[..., 1], seg[..., 0]); turn = np.angle(np.exp(1j * (h[:, -1] - h[:, 0])))
        alat = vpl.mean(1) ** 2 * np.abs(turn) / np.maximum(np.linalg.norm(seg, axis=-1).sum(1), 0.05)
        m = ok & (vpl[:, 0] > 4.0)
        out.append(np.stack([alat[m], -np.sign(turn[m]) * crs[m]], 1))
    return np.concatenate(out), float(hw[-1].mean())


out = sys.argv[1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
cols = ("#868e96", "#1c7ed6", "#e8590c", "#2b8a3e", "#ae3ec9", "#f08c00")
edges = np.array([1, 2, 3, 4, 5, 6, 7])
names, walls = [], []
for i, a in enumerate(sys.argv[2:]):
    nm, fn = a.split("=", 1)
    r, w = rows(fn)
    mid, med, lo, hi = [], [], [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        v = r[(r[:, 0] >= e0) & (r[:, 0] < e1), 1] * 100
        if len(v) < 200:
            continue
        mid.append(0.5 * (e0 + e1)); med.append(np.median(v)); lo.append(np.percentile(v, 10)); hi.append(np.percentile(v, 90))
    a1.plot(mid, med, "-o", color=cols[i], lw=2, ms=5, label=nm)
    a1.fill_between(mid, lo, hi, color=cols[i], alpha=0.10)
    names.append(nm); walls.append(w)
a1.axhline(0, color="#495057", lw=0.8)
a1.set_xlabel("plan이 요구한 옆가속 [m/s²]  (정상 선회 최대 8.2)"); a1.set_ylabel("0.4 s 뒤 plan 대비 바깥(+)/안쪽(−) [cm]")
a1.set_title("코너에서 plan보다 바깥으로 밀린 거리 (선: 중앙, 띠: 10–90 %)"); a1.legend(fontsize=9, loc="upper left"); a1.grid(alpha=0.3)
a2.barh(range(len(names))[::-1], walls, color=[cols[i] for i in range(len(names))])
for j, w in enumerate(walls):
    a2.text(w + 0.2, len(names) - 1 - j, f"{w:.2f}", va="center", fontsize=10)
a2.set_yticks(range(len(names))[::-1]); a2.set_yticklabels(names, fontsize=9)
a2.set_xlabel("벽 세게 닿음 / 세션 (90 s)"); a2.set_title("벽 충돌"); a2.set_xlim(0, max(walls) * 1.25 + 0.5)
fig.suptitle("avg26s u900 · frozen exam (128세션, 같은 세션) · MAP 변형별", fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=110); print("그림", out)
