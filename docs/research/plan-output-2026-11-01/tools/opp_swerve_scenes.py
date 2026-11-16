"""(11-16) opp_swerve.py의 장면 그림: 앞에 나만큼 빠른 상대 차(빠르기 >= 0.95)가 있는데 plan이 0.5 m 넘게 비켜 간 순간들.
칸마다 앞뒤 3 s: 내 차 길(파랑, 점 = 0.5 s), 상대 차 길(주황), 그 순간의 plan(초록 점), 세게 닿은 곳(X). 인자: 기록npz 출력png [장면 수]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
for _f in ("NanumGothic", "Noto Sans CJK KR"):
    if any(_f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
fn, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
R = Rec(fn); z = R.z
st, op, on, pl, rs = z["st"].astype(float), z["op"].astype(float), z["on"], z["pl"].astype(float), z["rs"]
T, B = rs.shape; L = float(z["lapL"][0]); BIN = 0.2; nb = int(np.ceil(L / BIN))
rem = z["rem"].astype(float); pos = np.mod(L - rem[:T], L); k = np.floor(np.nan_to_num(pos) / BIN).astype(int) % nb
solo = ~on.any(0) & ~(z["sid"][0] >= 0).any(-1); sel = solo[None] & ~rs & np.isfinite(pos)
sx, sy, sn, sv = (np.zeros(nb) for _ in range(4))
np.add.at(sx, k[sel], st[:T, :, 0][sel]); np.add.at(sy, k[sel], st[:T, :, 1][sel]); np.add.at(sn, k[sel], 1); np.add.at(sv, k[sel], st[:T, :, 3][sel])
line = np.stack([sx[sn > 0] / sn[sn > 0], sy[sn > 0] / sn[sn > 0]], 1); v_solo = np.where(sn > 0, sv / np.maximum(sn, 1), np.nan)
tang = np.roll(line, -1, 0) - np.roll(line, 1, 0); tang /= np.linalg.norm(tang, axis=1, keepdims=True)
def lat(q):
    dd = ((q[:, None] - line[None]) ** 2).sum(-1); j = dd.argmin(1); r = q - line[j]
    return tang[j, 0] * r[:, 1] - tang[j, 1] * r[:, 0]
yaw = st[:T, :, 2]; rx, ry = op[:T, :, 0] - st[:T, :, 0], op[:T, :, 1] - st[:T, :, 1]
fx = np.cos(yaw) * rx + np.sin(yaw) * ry; fy = -np.sin(yaw) * rx + np.cos(yaw) * ry
dyaw = np.abs(np.angle(np.exp(1j * (op[:T, :, 2] - yaw))))
opos = np.mod(L - z["orem"][:T].astype(float), L); ratio = np.full(opos.shape, np.nan); ok = np.isfinite(opos)
ratio[ok] = op[:T, :, 3][ok] / v_solo[np.floor(opos[ok] / BIN).astype(int) % nb]
ahead = on[:T] & ~rs & (fx > 0.6) & (fx < 3.0) & (np.abs(fy) < 0.5) & (dyaw < np.pi / 4) & (st[:T, :, 3] > 1.0) & (ratio >= 0.95)
ahead[:30] = False; ahead[T - 30:] = False
ev, last = [], {}
for t, b in zip(*np.nonzero(ahead)):
    if t - last.get(b, -999) < 50:
        continue
    off = np.abs(lat(pl[t, b])).max()
    if off > 0.5:
        ev.append((t, b, off)); last[b] = t
ev = ev[:N]
hard = R.d[..., 3:6].sum(-1)
g, ext = R.grid(0)
cols = 4; rows = int(np.ceil(len(ev) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
for a in axs.flat: a.axis("off")
for a, (t, b, off) in zip(axs.flat, ev):
    a.axis("on"); a.imshow((g <= 0).astype(float), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.6)
    w = slice(max(0, t - 10), min(T, t + 30))
    a.plot(line[:, 0], line[:, 1], color="0.75", lw=0.8, ls=":")
    a.plot(st[w, b, 0], st[w, b, 1], color="#1971c2", lw=1.6); a.plot(st[w, b, 0][::5], st[w, b, 1][::5], ".", color="#1971c2", ms=5)
    mo = on[w, b]
    a.plot(np.where(mo, op[w, b, 0], np.nan), np.where(mo, op[w, b, 1], np.nan), color="#e8590c", lw=1.6)
    a.plot(np.where(mo, op[w, b, 0], np.nan)[::5], np.where(mo, op[w, b, 1], np.nan)[::5], ".", color="#e8590c", ms=5)
    a.plot(pl[t, b, :, 0], pl[t, b, :, 1], "o-", color="#2f9e44", ms=3, lw=1)
    a.plot(st[t, b, 0], st[t, b, 1], "s", color="#1971c2", ms=7); a.plot(op[t, b, 0], op[t, b, 1], "s", color="#e8590c", ms=7)
    hh = [tt for tt in range(w.start, w.stop - 1) if hard[tt, b] > 0]
    for tt in hh:
        a.plot(st[tt + 1, b, 0], st[tt + 1, b, 1], "x", color="#e03131", ms=10, mew=2.5)
    c = st[t, b, :2]; a.set_xlim(c[0] - 4, c[0] + 4); a.set_ylim(c[1] - 4, c[1] + 4); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"세션 {b} t={t * R.dt:.1f}s | 나 {st[t, b, 3]:.1f} · 상대 {op[t, b, 3]:.1f} m/s\nplan 벗어남 {off:.2f} m"
                + (f" · 3 s 안 세게 닿음 {len(hh)}" if hh else ""), fontsize=9)
fig.suptitle(f"{os.path.basename(fn)}: 앞에 나만큼 빠른 차(0.6-3 m), plan이 0.5 m 넘게 비켜 간 순간 {len(ev)}개 "
             "(파랑 = 내 차 0.5 s 점, 주황 = 상대 차, 초록 = 그 순간 plan, 점선 = 혼자 달린 내 길, X = 세게 닿음)", fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=95); print(out, len(ev))
