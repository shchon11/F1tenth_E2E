"""1:1 경주의 세게 부딪힌 순간 확대 그림 (11-02): console_races.py와 같은 콘솔 세션·같은 리셋 순서로 경주를 다시 돌리고, 차와 세게
부딪힐 때마다 그 앞 1.5 s를 ±3.5 m로 확대해 그린다 -- 두 차의 차체(0.58 x 0.30 m, 시간 색), 그 시각 우리가 그린 궤적(MAP이 따라간
경로점), 정책이 본 통로(상자 + 우리가 닿을 때의 차 = 막힌 칸, 빨간 점), 누가 운전했는지(상대 단계 정책 / 혼자 선생), 두 차 속도·간격.
인자: 체크포인트 맵id 판당초 판수 상대종류 상대속도배율 시드 그림머리 [학습장애물 0/1]"""
import os, sys, math
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim.learn.state_rl import AHEAD, CORRIDOR_U, CORRIDOR_BINS


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, secs, n_race, kind, sc, seed, head = (sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5],
                                               float(sys.argv[6]), int(sys.argv[7]), sys.argv[8])
proc = bool(int(sys.argv[9])) if len(sys.argv) > 9 else False
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=2, device="cuda", compile=False,
                                    collision_mode="soft", procedural=proc, episode_s=3600.0, randomize=False,
                                    opponent_slots=[{"kind": kind, "speed_scale": [sc, sc]}]), 0)
env = s["env"]
runner = s["act_fn"]
tp = getattr(runner, "tp", None)
from f1sim import maps, tracks
tr = maps.load(tracks.resolve(mp))
from matplotlib import font_manager
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
occ = tr.occupancy; res = tr.resolution; H, W = occ.shape
ext = [tr.origin[0], tr.origin[0] + W * res, tr.origin[1], tr.origin[1] + H * res]
i_on, i_hd = CC.index("onsets_car"), CC.index("hard_car")
ctr = np.array([0.5] + [0.5 * (AHEAD[k - 1] + AHEAD[k]) for k in range(1, len(AHEAD))])        # segment centres [m]
U = np.asarray(CORRIDOR_U)
ubin = np.array([0.5 * (U[2 * j] + U[2 * j + 1]) for j in range(CORRIDOR_BINS)])                 # bin centres [-1, 1]
NC = len(AHEAD) * CORRIDOR_BINS
LB, WB = 0.58, 0.30
track = env.sim.track


def body(x, y, yaw):
    c, s_ = math.cos(yaw), math.sin(yaw)
    pts = [(LB / 2, WB / 2), (LB / 2, -WB / 2), (-LB / 2, -WB / 2), (-LB / 2, WB / 2)]
    return [(x + c * a - s_ * b, y + s_ * a + c * b) for a, b in pts]


def corridor_xy(s0, tid, blk):
    """the blocked bins of the policy's corridor (13 x 7) at arc s0 -> world points"""
    ks, js = np.nonzero(blk.reshape(len(AHEAD), CORRIDOR_BINS) > 0.5)
    if not len(ks):
        return np.zeros((0, 2))
    sq = torch.tensor(s0 + ctr[ks], dtype=torch.float32, device=env.device)
    xy, yaw = track.pose_at_s(sq, torch.full_like(sq, tid, dtype=torch.long))
    wv = track.sample_edt(xy[None], torch.full((1, len(ks)), tid, dtype=torch.long, device=env.device))[0]
    xy, yaw, wv = xy.cpu().numpy(), yaw.cpu().numpy(), wv.cpu().numpy()
    lat = ubin[js] * wv
    return np.stack([xy[:, 0] - np.sin(yaw) * lat, xy[:, 1] + np.cos(yaw) * lat], 1)


tot_hard = 0
for L in range(n_race):
    if L > 0:
        s["obs"], _ = env.reset()                                    # the console's 리셋
    cf0 = env.sim.contact_frame()[0].float().cpu().numpy()
    prev_c, prev_h = cf0[i_on], cf0[i_hd]
    rec = []; hard_k = []
    for k in range(int(secs / 0.025)):
        w._step_once(s)
        st = env.sim.state
        lr = env.last_result
        tk = getattr(env, "tracker", None)
        held = getattr(tk, "_held", None) if tk is not None else None
        plan = None
        if held is not None:
            pth, _pv, _pk, pose = held
            # the plan's own length (0.6 s at its speed, +0.5 m), not the straight extension MAP's search runs on
            n_ = max(3, int((0.6 * float(st[0, 3].abs()) + 0.5) / 0.1))
            p_ = pth[0].float().cpu().numpy()[:n_:2]; q = pose[0].float().cpu().numpy()
            c, s_ = math.cos(q[2]), math.sin(q[2])
            plan = np.stack([q[0] + c * p_[:, 0] - s_ * p_[:, 1], q[1] + s_ * p_[:, 0] + c * p_[:, 1]], 1)
        gate = runner.gate
        rec.append(dict(t=k * 0.025, a=st[0, [0, 1, 2, 3]].float().cpu().numpy().copy(),
                        b=st[1, [0, 1, 2, 3]].float().cpu().numpy().copy(), plan=plan,
                        race=(bool(gate[0]) if gate is not None else True),
                        s=float(lr.s[0]), tid=int(env.sim.tid[0]), s_b=float(lr.s[1]), tid_b=int(env.sim.tid[1]),
                        cor=(tp.o[0, -NC:].float().cpu().numpy().copy() if tp is not None else None)))
        cf = env.sim.contact_frame()[0].float().cpu().numpy()
        if cf[i_hd] > prev_h:
            hard_k.append(k)
        prev_c, prev_h = cf[i_on], cf[i_hd]
    tot_hard += len(hard_k)
    if not hard_k:
        print(f"판 {L}: 세게 0", flush=True)
        continue
    # one panel per hard contact (at most 4)
    ks = hard_k[:4]
    fig, axs = plt.subplots(1, len(ks), figsize=(7.5 * len(ks), 7.8), squeeze=False)
    for ax, kc in zip(axs[0], ks):
        a = rec[kc]["a"]
        ax.imshow(~occ, cmap="gray", origin="lower", extent=ext, alpha=0.55)
        cmap = plt.get_cmap("viridis")
        back = [60, 40, 20, 10, 0]                                     # steps before: 1.5, 1.0, 0.5, 0.25, 0 s
        for j, db in enumerate(back):
            i = max(0, kc - db); r = rec[i]; col = cmap(j / (len(back) - 1))
            ax.add_patch(Polygon(body(*r["a"][:3]), closed=True, fc=col, ec="k", alpha=0.75, lw=0.8))
            ax.add_patch(Polygon(body(*r["b"][:3]), closed=True, fc="orange", ec=col, alpha=0.75, lw=2.0))
            if db in (40, 20, 0) and r["plan"] is not None:
                ax.plot(r["plan"][:, 0], r["plan"][:, 1], "-", color=col, lw=1.8)
            if db == 20:
                # round trip of the frame the bins are drawn in: the opponent's centre rebuilt from its arc and its
                # lateral offset -- it lands on its body when the drawing is right
                sq = torch.tensor([r["s_b"]], dtype=torch.float32, device=env.device)
                xy_c, yaw_c = track.pose_at_s(sq, torch.tensor([r["tid_b"]], device=env.device))
                xc, yc, yc_ = float(xy_c[0, 0]), float(xy_c[0, 1]), float(yaw_c[0])
                lo = -(r["b"][0] - xc) * math.sin(yc_) + (r["b"][1] - yc) * math.cos(yc_)
                ax.plot(xc - math.sin(yc_) * lo, yc + math.cos(yc_) * lo, "+", color="red", ms=14, mew=2)
            if db == 20 and r["cor"] is not None:
                cxy = corridor_xy(r["s"], r["tid"], r["cor"])
                if len(cxy):
                    ax.scatter(cxy[:, 0], cxy[:, 1], s=26, c="red", marker="s", alpha=0.5, label="0.5 s 전 통로 막힌 칸")
        xs = [rec[max(0, kc - 60)]["a"][0], a[0]]; ys = [rec[max(0, kc - 60)]["a"][1], a[1]]
        cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
        R_ = max(3.5, 0.6 * max(max(xs) - min(xs), max(ys) - min(ys)) + 1.0)
        ax.set_xlim(cx - R_, cx + R_); ax.set_ylim(cy - R_, cy + R_); ax.set_aspect("equal")
        lines = []
        for db in back:
            r = rec[max(0, kc - db)]
            lines.append(f"-{db * 0.025:.2f}s 내 {r['a'][3]:.1f} 상대 {r['b'][3]:.1f} m/s "
                         f"{'상대단계' if r['race'] else '혼자선생'}")
        ax.set_title(f"판 {L}, t={kc * 0.025:.2f} s 세게 부딪힘\n" + "\n".join(lines), fontsize=8, loc="left")
        h_, l_ = ax.get_legend_handles_labels()
        if h_:
            ax.legend(h_[:1], l_[:1], fontsize=7, loc="lower right")
    fig.suptitle(f"{os.path.basename(os.path.dirname(run))} vs {kind} x{sc} (시드 {seed}) | 차체: 1.5/1.0/0.5/0.25/0 s 전(보라->노랑), "
                 f"상대 = 주황 | 선 = 그 시각 그린 궤적", fontsize=9)
    png = f"{head}_R{L}.png"; fig.tight_layout(); fig.savefig(png, dpi=60); plt.close(fig)
    print(f"판 {L}: 세게 {len(hard_k)} (t = {', '.join(f'{x * 0.025:.1f}' for x in hard_k)} s) | 그림 {png}", flush=True)
print(f"합계 세게 {tot_hard}", flush=True)
