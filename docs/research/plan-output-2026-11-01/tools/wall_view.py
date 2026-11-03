"""벽 긁기 버전 (11-03): stall_view.py와 같고 첫 벽 접촉 앞뒤를 봄(2026 gw 배치 10·13 오른쪽 직선). """
"""상자 앞에서 멈추며 미는 장면을 시간 순으로 (11-03): props_hit_audit에서 세게 상자 충돌은 모두 상자가 보인 지 0.5 s 넘어서
(64~71 %는 1.5 s 넘어서, 0.6~0.8 m/s) -- 늦게 보여서가 아니라 보고도 못 돌아감. console_layouts.py와 똑같이 배치 0..L-1을 40 s씩
달린 뒤(난수 순서 그대로) 배치 L에서 매 걸음: 위치·속도, MAP 실행 명령(조향, 속도), 혼합 게이트(상자 정책이 모는지),
가장 가까운 상자의 차 기준 위치와 보임 여부, 접촉. 첫 상자 접촉 앞뒤를 0.1 s마다 표로, 그 구간 계획 경로를 그림으로.
인자: 체크포인트 맵id 배치번호 시드 그림머리"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim import props as _props
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim.learn import state_rl as SR


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, L, seed, head = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
w._overlay = dict(getattr(w, "_overlay", {}) or {}, plan=True)
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=True, episode_s=3600.0, randomize=False), 0)
env = s["env"]; proc = env.procedural; runner = s["act_fn"]
from f1sim import maps, tracks
tr = maps.load(tracks.resolve(mp))
from matplotlib import font_manager
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
i_on, i_hd = CC.index("onsets_wall"), CC.index("hard_wall")
_fp = {}


def footprints():
    out = []
    sid = proc.p_sid[0].cpu().numpy(); pose = proc.p_poses[0].cpu().numpy()
    for j, k in enumerate(sid):
        if k < 0:
            continue
        if int(k) not in _fp:
            sh = proc.shapes[int(k)]
            _fp[int(k)] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, float)
        p = _fp[int(k)]; c, sn = np.cos(pose[j, 2]), np.sin(pose[j, 2])
        out.append(np.stack([pose[j, 0] + p[:, 0] * c - p[:, 1] * sn, pose[j, 1] + p[:, 0] * sn + p[:, 1] * c], 1))
    return out


for l in range(L + 1):
    if l > 0:
        s["obs"], _ = env.reset()
    if l < L:
        for k in range(int(40.0 / 0.025)):
            w._step_once(s)
            if k % 20 == 0:
                w._snapshot(s)
        continue
    fp0 = footprints()
    lid = env.learner_ids
    rows, plans, VH, DH = [], [], [], []
    cf = env.sim.contact_frame()[0].float().cpu().numpy(); p_on, p_hd = cf[i_on], cf[i_hd]
    for k in range(int(40.0 / 0.025)):
        w._step_once(s)
        st = env.sim.state[0]
        x, y, yaw, v = (float(st[i]) for i in range(4))
        vis = SR.visible_props(env, lid)
        live = proc.p_sid[0] >= 0
        d = proc.p_poses[0, :, :2] - st[None, :2]
        dist = torch.where(live, d.norm(dim=-1), torch.full_like(d[:, 0], 99.0))
        j = int(dist.argmin())
        c, sn = np.cos(yaw), np.sin(yaw)
        dx, dy = float(d[j, 0]) * c + float(d[j, 1]) * sn, -float(d[j, 0]) * sn + float(d[j, 1]) * c
        VH.append(vis[0].cpu().numpy().copy() if vis is not None else np.zeros(proc.C, bool)); DH.append(dist.cpu().numpy().copy())
        g = runner.gate
        cf = env.sim.contact_frame()[0].float().cpu().numpy()
        on, hd = cf[i_on] > p_on, cf[i_hd] > p_hd
        p_on, p_hd = cf[i_on], cf[i_hd]
        rows.append((k * 0.025, x, y, v, float(env.sim.cmd[0, 0]), float(env.sim.cmd[0, 1]), int(bool(g[0])) if g is not None else -1,
                     j, dx, dy, float(dist[j]), int(bool(vis[0, j])) if vis is not None else -1, int(on), int(hd)))
        if k % 4 == 0:
            fr = w._snapshot(s)
            if fr.get("plan") is not None:
                plans.append((k * 0.025, np.asarray(fr["plan"], dtype=float)[:, :2]))
    A = np.array(rows)
    hit = np.nonzero(A[:, 12] > 0)[0]
    print(f"배치 {L}: 상자 {len(fp0)}개, 벽 접촉 {int(A[:, 12].sum())} (세게 {int(A[:, 13].sum())})")
    if hit.size == 0:
        sys.exit(0)
    k0 = int(hit[0]); a, b = max(0, k0 - 80), min(len(A), k0 + 160)
    VH, DH = np.array(VH), np.array(DH)
    jh = int(A[k0, 7])
    seen = np.nonzero(VH[:k0 + 1, jh])[0]
    if seen.size:
        k1 = int(seen[0])
        print(f"  부딪힌 상자 #{jh}: 처음 보임 {A[k1, 0]:.2f} s (거리 {DH[k1, jh]:.2f} m, 차 속도 {A[k1, 3]:.2f} m/s), 부딪힘 {A[k0, 0]:.2f} s -- "
              f"그 사이 보인 걸음 {VH[k1:k0 + 1, jh].mean() * 100:.0f} %; 거리 6/4/3/2 m일 때 보임: "
              + " ".join(f"{d_}m:{int(VH[np.argmin(np.abs(DH[:k0 + 1, jh] - d_)), jh])}" for d_ in (6, 4, 3, 2)))
    else:
        print(f"  부딪힌 상자 #{jh}: 부딪힐 때까지 한 번도 안 보임")
    print("  t      x      y    속도  조향명령 속도명령 게이트 상자#  상자dx  상자dy  거리  보임 접촉 세게")
    for r in A[a:b:4]:
        print(f"  {r[0]:5.2f} {r[1]:6.2f} {r[2]:6.2f} {r[3]:5.2f} {r[4]:+7.3f} {r[5]:7.2f}    {int(r[6]):2d}   {int(r[7]):3d} {r[8]:+6.2f} {r[9]:+6.2f} {r[10]:5.2f}   {int(r[11])}   {int(r[12])}   {int(r[13])}")
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(~tr.occupancy, cmap="gray", origin="lower", alpha=0.6,
              extent=[tr.origin[0], tr.origin[0] + tr.occupancy.shape[1] * tr.resolution, tr.origin[1], tr.origin[1] + tr.occupancy.shape[0] * tr.resolution])
    for poly in fp0:
        ax.fill(poly[:, 0], poly[:, 1], color="0.35", alpha=0.6, lw=0)
    for poly in footprints():
        ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], "b-", lw=0.8)
    seg = A[a:b]
    sc = ax.scatter(seg[:, 1], seg[:, 2], c=seg[:, 0], s=6, cmap="viridis")
    t0, t1 = seg[0, 0], seg[-1, 0]
    cm = plt.get_cmap("viridis")
    for t, p in plans:
        if t0 <= t <= t1:
            ax.plot(p[:, 0], p[:, 1], "-", color=cm((t - t0) / max(t1 - t0, 1e-6)), lw=1.0)
    for r in seg[seg[:, 12] > 0]:
        ax.plot(r[1], r[2], "x", color="r" if r[13] else "orange", ms=14, mew=2.5)
    ax.set_xlim(seg[:, 1].min() - 1.5, seg[:, 1].max() + 1.5); ax.set_ylim(seg[:, 2].min() - 1.5, seg[:, 2].max() + 1.5)
    ax.set_aspect("equal"); plt.colorbar(sc, ax=ax, fraction=0.04, label="시간 s")
    ax.set_title(f"{os.path.basename(os.path.dirname(run))} 배치 {L}: 첫 벽 접촉 {A[k0, 0]:.2f} s 앞 2 s ~ 뒤 4 s, 선 = 그 순간의 계획 경로(색 = 시간)")
    fig.tight_layout(); fig.savefig(head + f"_L{L}.png", dpi=70); plt.close(fig)
    print("그림", head + f"_L{L}.png")
