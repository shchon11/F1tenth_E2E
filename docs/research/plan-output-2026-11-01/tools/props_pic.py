"""콘솔 워커(SimWorker)로 학습 장애물(절차적) 한 배치를 달리고 그림 한 장 (11-02): 맵 + 상자(처음 위치 회색, 끝 위치 테두리)
+ 주행 경로(속도 색) + 0.5 s마다 정책이 낸 궤적(검은 선) + 접촉 시작(벽 빨강, 장애물 주황, 세게 = 큰 표시),
오른쪽에 접촉이 가장 많은 곳 확대. 접촉은 콘솔 계기와 같은 수(`sim.contact_frame`).
인자: 체크포인트 맵id 초 시드 png"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim import props as _props
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, secs, seed, png = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
w._overlay = dict(getattr(w, "_overlay", {}) or {}, plan=True)
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=True, episode_s=max(60.0, secs + 5), randomize=False), 0)
env = s["env"]; proc = env.procedural
from f1sim import maps, tracks
tr = maps.load(tracks.resolve(mp))
from matplotlib import font_manager
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
_fp = {}


def footprints():
    """[(polygon (K, 2) world, style)] of car 0's race's live props."""
    out = []
    sid = proc.p_sid[0].cpu().numpy(); pose = proc.p_poses[0].cpu().numpy()
    for j, k in enumerate(sid):
        if k < 0:
            continue
        sh = proc.shapes[int(k)]
        if int(k) not in _fp:
            _fp[int(k)] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, float)
        p = _fp[int(k)]; c, sn = np.cos(pose[j, 2]), np.sin(pose[j, 2])
        out.append((np.stack([pose[j, 0] + p[:, 0] * c - p[:, 1] * sn, pose[j, 1] + p[:, 0] * sn + p[:, 1] * c], 1), sh.style))
    return out


props0 = footprints()
xs, ys, vs, plans, hits = [], [], [], [], []
i_on = [CC.index(k) for k in ("onsets_wall", "onsets_prop", "onsets_car")]
i_hd = [CC.index(k) for k in ("hard_wall", "hard_prop", "hard_car")]
prev_on = np.zeros(3); prev_hd = np.zeros(3); removed_t = None
for k in range(int(secs / 0.025)):
    w._step_once(s)
    st = env.sim.state[0]
    xs.append(float(st[0])); ys.append(float(st[1])); vs.append(float(st[3]))
    cf = env.sim.contact_frame()[0].float().cpu().numpy()
    on = cf[i_on]; hd = cf[i_hd]
    for j in range(3):
        if on[j] > prev_on[j]:
            hits.append((k * 0.025, j, bool(hd[j] > prev_hd[j]), xs[-1], ys[-1], float(cf[CC.index("severity")]), vs[-1]))
    prev_on, prev_hd = on, hd
    if removed_t is None and props0 and int((proc.p_sid[0] >= 0).sum()) == 0:
        removed_t = k * 0.025
    if k % 20 == 0:
        fr = w._snapshot(s)
        if fr.get("plan") is not None:
            plans.append(np.asarray(fr["plan"], dtype=float)[:, :2])
props1 = footprints()
xs, ys, vs = map(np.asarray, (xs, ys, vs))
occ = tr.occupancy; res = tr.resolution; H, W = occ.shape
ext = [tr.origin[0], tr.origin[0] + W * res, tr.origin[1], tr.origin[1] + H * res]
fig = plt.figure(figsize=(20, 11))
KCOL = ("red", "darkorange", "magenta"); KNAME = ("벽", "장애물", "차")


def draw(ax, lw_plan=0.7):
    ax.imshow(~occ, cmap="gray", origin="lower", extent=ext, alpha=0.6)
    for poly, _ in props0:
        ax.fill(poly[:, 0], poly[:, 1], color="0.35", alpha=0.6, lw=0)
    for poly, _ in props1:
        ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], "b-", lw=0.8)
    sc = ax.scatter(xs, ys, c=vs, s=2, cmap="turbo", vmin=0, vmax=10)
    for p in plans:
        ax.plot(p[:, 0], p[:, 1], "k-", lw=lw_plan)
    for t, j, hard, x, y, sev, v in hits:
        ax.plot(x, y, "x", color=KCOL[j], ms=(16 if hard else 9), mew=(3 if hard else 1.5))
    return sc


ax = fig.add_axes([0.02, 0.05, 0.56, 0.88])
sc = draw(ax)
pad = 1.0
ax.set_xlim(xs.min() - pad, xs.max() + pad); ax.set_ylim(ys.min() - pad, ys.max() + pad); ax.set_aspect("equal")
plt.colorbar(sc, ax=ax, fraction=0.03, label="속도 m/s")
cnt = [sum(1 for h in hits if h[1] == j) for j in range(3)]; hcnt = [sum(1 for h in hits if h[1] == j and h[2]) for j in range(3)]
ax.set_title(f"{os.path.basename(os.path.dirname(run))}/{os.path.basename(run)} @ {mp} 학습 장애물 시드 {seed}: {secs:.0f} s | "
             f"접촉 벽 {cnt[0]}(세게 {hcnt[0]}) 장애물 {cnt[1]}(세게 {hcnt[1]})"
             + (f" | 장애물 치움 {removed_t:.1f} s" if removed_t is not None else ""))
bx = fig.add_axes([0.60, 0.05, 0.38, 0.88])
if hits:
    hx = np.array([[h[3], h[4]] for h in hits])
    # the 2.5 m window holding the most contacts
    best = max(range(len(hx)), key=lambda i: int((np.abs(hx - hx[i]).max(1) < 1.25).sum()))
    cx, cy = hx[best]
    draw(bx, lw_plan=1.0)
    bx.set_xlim(cx - 1.6, cx + 1.6); bx.set_ylim(cy - 1.6, cy + 1.6); bx.set_aspect("equal")
    near = [h for h in hits if abs(h[3] - cx) < 1.6 and abs(h[4] - cy) < 1.6]
    bx.set_title("확대: " + ", ".join(f"{h[0]:.1f}s {KNAME[h[1]]}{'(세게)' if h[2] else ''} {h[6]:.1f}m/s" for h in near[:6]), fontsize=9)
else:
    bx.axis("off")
fig.savefig(png, dpi=60)
print("그림", png, "| 접촉", [(round(h[0], 1), KNAME[h[1]], h[2], round(h[6], 1)) for h in hits][:30],
      "| 장애물", len(props0), "개 | 치움", removed_t)
