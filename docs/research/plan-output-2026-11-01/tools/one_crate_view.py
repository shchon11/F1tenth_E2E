"""상자 하나에 박는 장면을 상자 없는 같은 주행과 나란히 (11-03): one_crate.py에서 새 선생(srp25h3)이 혼자 주행선 위 상자 하나를
12곳 중 5곳에서 받고 밀며 멈춤(접촉 32, 세게 16). 정책이 상자에 반응은 하는지(안 함 / 늦게 / 방향이 틀림 / 반응은 하는데 차가 못 따라감):
같은 리셋에서 상자 없이 한 바퀴(기준), 상자를 놓고 한 바퀴. 트랙 위치(s)를 맞춰 실행 속도 명령·조향 명령·실제 속도·가로 위치를 나란히,
상자까지 거리·보임과 함께 0.1 s마다. 그림: 두 주행의 길과 그때그때의 계획 경로.
인자: 맵id 상자모양 체크포인트 자리번호들(쉼표) 그림머리 [선 체크포인트: one_crate.py의 첫 체크포인트, 자리가 같도록]"""
import os, sys, math
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer import prop_layout as PL
from f1sim.viewer.console import protocol as P
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim.learn import state_rl as SR
from f1sim import maps, tracks


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


mp, want, path, which, head = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(",")], sys.argv[5]
line_path = sys.argv[6] if len(sys.argv) > 6 else path
i_on, i_hd = CC.index("onsets_prop"), CC.index("hard_prop")
N_POS = 12
from matplotlib import font_manager
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
tr = maps.load(tracks.resolve(mp))
def session(p):
    global w, s, env, proc, runner, cat, shape, hold, Lt
    w = sim_worker.SimWorker(_N(), _N())
    w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
    w._overlay = dict(getattr(w, "_overlay", {}) or {}, plan=True)
    s = w.build_session(P.SessionConfig(run=p, map_name=mp, seed=0, races=1, cars_per_race=1, device="cuda", compile=False,
                                        collision_mode="soft", procedural=True, episode_s=3600.0, randomize=False), 0)
    env = s["env"]; proc = env.procedural; runner = s["act_fn"]
    cat = PL.catalogue(proc)
    shape = next(e for e in cat if e["label"].endswith(want.replace("x", "×") + " cm"))
    hold = PL.LayoutHold(proc, int(proc.C), env.sim.tid[:1]).install()
    Lt = float(env.sim.track.length[env.sim.tid[0]])


def run(pieces, secs, crate=None):
    hold.set_pieces(pieces)
    s["obs"], _ = env.reset()
    cf = env.sim.contact_frame()[0].float().cpu().numpy(); p_on, p_hd = cf[i_on], cf[i_hd]
    rows, plans = [], []
    lid = env.learner_ids
    for k in range(int(secs / 0.025)):
        w._step_once(s)
        st = env.sim.state[0]
        r = env.last_result
        vis = SR.visible_props(env, lid) if crate is not None else None
        cf = env.sim.contact_frame()[0].float().cpu().numpy()
        on, hd = cf[i_on] > p_on, cf[i_hd] > p_hd
        p_on, p_hd = cf[i_on], cf[i_hd]
        g = runner.gate
        d = math.hypot(float(st[0]) - crate[0], float(st[1]) - crate[1]) if crate is not None else float("nan")
        rows.append((k * 0.025, float(r.s[0]) % Lt, float(r.lateral[0]), float(st[3]), float(env.sim.cmd[0, 0]), float(env.sim.cmd[0, 1]),
                     int(bool(g[0])) if g is not None else -1, d, int(bool(vis[0, 0])) if vis is not None else -1, int(on), int(hd),
                     float(st[0]), float(st[1])))
        if k % 4 == 0:
            fr = w._snapshot(s)
            if fr.get("plan") is not None:
                plans.append((k * 0.025, np.asarray(fr["plan"], dtype=float)[:, :2]))
    return np.array(rows), plans


session(line_path)
base, _ = run([], 10.0)
X = base[120:120 + int(6.5 / 0.025)]
if line_path != path:
    del w, s, env
    torch.cuda.empty_cache()
    session(path)
    base, _ = run([], 10.0)
idx = np.linspace(0, len(X) - 1, N_POS, endpoint=False).astype(int)
# the solo line point and heading, as one_crate.py placed them
env_xy = X[:, 11:13]
for j in which:
    i = idx[j]
    x, y = float(env_xy[i, 0]), float(env_xy[i, 1])
    yaw = math.atan2(env_xy[min(i + 1, len(X) - 1), 1] - env_xy[max(i - 1, 0), 1], env_xy[min(i + 1, len(X) - 1), 0] - env_xy[max(i - 1, 0), 0])
    A, plans = run([{"sid": shape["id"], "x": x, "y": y, "yaw": yaw}], 12.0, crate=(x, y))
    hit = np.nonzero(A[:, 9] > 0)[0]
    s_c = float(X[i, 1])
    print(f"== {os.path.basename(os.path.dirname(path))} 자리 {j}: 상자 s = {s_c:.1f} m, 접촉 {int(A[:, 9].sum())} (세게 {int(A[:, 10].sum())})")
    if hit.size == 0:
        continue
    k0 = int(hit[0])
    seen = np.nonzero(A[:k0 + 1, 8] > 0)[0]
    if seen.size:
        print(f"  처음 보임 {A[seen[0], 0]:.2f} s 거리 {A[seen[0], 7]:.2f} m (첫 접촉 {A[k0, 0]:.2f} s), 그 뒤 보인 걸음 {A[seen[0]:k0 + 1, 8].mean() * 100:.0f} %")
    print("   t     s    | 상자 있을 때: 속도 속도명령 조향명령 가로 게이트 | 상자 없을 때 같은 s: 속도 속도명령 조향명령 가로 | 상자거리 보임 접촉")
    a = max(0, k0 - 60)
    for r in A[a:k0 + 60:4]:
        jb = int(np.argmin(np.abs(((base[:, 1] - r[1] + Lt / 2) % Lt) - Lt / 2) + 1e3 * (base[:, 0] < 2.0)))
        b = base[jb]
        print(f"  {r[0]:5.2f} {r[1]:5.1f} | {r[3]:5.2f} {r[5]:6.2f} {r[4]:+7.3f} {r[2]:+5.2f}  {int(r[6]):2d} | {b[3]:5.2f} {b[5]:6.2f} {b[4]:+7.3f} {b[2]:+5.2f} | {r[7]:5.2f}  {int(r[8])}  {int(r[9])}")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(~tr.occupancy, cmap="gray", origin="lower", alpha=0.6,
              extent=[tr.origin[0], tr.origin[0] + tr.occupancy.shape[1] * tr.resolution, tr.origin[1], tr.origin[1] + tr.occupancy.shape[0] * tr.resolution])
    fp = np.asarray(shape["footprint"]); c_, s_ = math.cos(yaw), math.sin(yaw)
    ax.fill(x + fp[:, 0] * c_ - fp[:, 1] * s_, y + fp[:, 0] * s_ + fp[:, 1] * c_, color="0.3")
    seg = A[a:k0 + 80]
    ax.plot(base[:, 11], base[:, 12], "-", color="tab:blue", lw=1, label="상자 없을 때 길")
    sc = ax.scatter(seg[:, 11], seg[:, 12], c=seg[:, 0], s=8, cmap="viridis", label="상자 있을 때 길(색 = 시간)")
    t0, t1 = seg[0, 0], seg[-1, 0]
    cm = plt.get_cmap("autumn")
    for t, p in plans:
        if t0 <= t <= t1:
            ax.plot(p[:, 0], p[:, 1], "-", color=cm((t - t0) / max(t1 - t0, 1e-6)), lw=0.9)
    for r in seg[seg[:, 9] > 0]:
        ax.plot(r[11], r[12], "x", color="r" if r[10] else "orange", ms=12, mew=2.5)
    ax.set_xlim(seg[:, 11].min() - 1.5, seg[:, 11].max() + 1.5); ax.set_ylim(seg[:, 12].min() - 1.5, seg[:, 12].max() + 1.5)
    ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8); plt.colorbar(sc, ax=ax, fraction=0.04, label="시간 s")
    ax.set_title(f"{os.path.basename(os.path.dirname(path))} 상자 하나 자리 {j}: 계획 경로(빨강 -> 노랑 = 시간)")
    fig.tight_layout(); fig.savefig(f"{head}_{j}.png", dpi=70); plt.close(fig)
    print("  그림", f"{head}_{j}.png", flush=True)
