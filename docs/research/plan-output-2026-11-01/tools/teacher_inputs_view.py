"""선생이 무엇을 보고 궤적을 내나 (11-03, 사용자: "occupancy는 어떤 class가 있어? 좀 보여줘바 / planner가 뭘 보고 뽑히는지,
입력은 GT로 줘도 occupancy 기반으로 추론되는 거 아니야?"). 콘솔 워커, 학습 장애물 세션, 상자 정책이 운전하는 결정에서:
(1) 그림 -- 맵 위에 선생이 받는 상자 정보 셋: 통로 지도 13 x 7 칸(막힘 = 주황, 빔 = 회색; 클래스 없이 막힘/빔 둘뿐), 가까운 상자 3개의
    GT 중심(점; 크기 없음), 108줄 라이다 끝점(파랑), 앞길 점 13개(검정) + 오른쪽에 13 x 7 행렬 그대로.
(2) 지우기 시험 -- 같은 입력에서 한 묶음만 '상자 없음' 값으로 바꿨을 때 실행 속도 명령(첫 경로점, 제동 한계 적용)과 첫 곡률이 얼마나
    바뀌나: 통로 지도 -> 0, GT 상자 칸 -> 0(혼자 정책에 넣는 값과 같음), 라이다 -> 학습 평균(정규화 0). 바뀜이 클수록 그 입력을 봄.
인자: 체크포인트 맵id 초 시드 그림머리"""
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
from f1sim.learn import state_rl as SR
from f1sim import mpc
from f1sim.gym_env import PRIV_V2_OPP_KEYS, PRIV_V2_OPP_SLOTS, PRIV_V2_PROP_DIM


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, secs, seed, head = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
REC = {}
_orig = SR.SoloMix.__call__


def _rec(self, a_pol, last):
    out = _orig(self, a_pol, last)
    REC.update(gate=bool(self.gate[0]), new=True)
    return out


SR.SoloMix.__call__ = _rec
TP = []
_ti = SR.TeacherPolicy.__init__


def _ti_rec(self, *a, **k):
    _ti(self, *a, **k); TP.append(self)


SR.TeacherPolicy.__init__ = _ti_rec
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=True, episode_s=3600.0, randomize=False), 0)
env = s["env"]; tr = env.sim.track; po = env.procedural; lid = env.learner_ids
inner = env.tracker._inner if hasattr(env.tracker, "_inner") else env.tracker
spec = inner.spec
from f1sim import maps, tracks
tm = maps.load(tracks.resolve(mp))
from matplotlib import font_manager
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
occ = tm.occupancy; res = tm.resolution; H, W = occ.shape
ext = [tm.origin[0], tm.origin[0] + W * res, tm.origin[1], tm.origin[1] + H * res]
tp = None
pvd = int(env.privileged_v2().shape[1])
ko = PRIV_V2_OPP_SLOTS * len(PRIV_V2_OPP_KEYS)
PROP = slice(8 + ko + 4, 8 + ko + 4 + PRIV_V2_PROP_DIM)               # ego 8, opponents, nearest-car geometry 4, props


def v1k1(a, v):
    k, _, vv = mpc.decode_wp4(a.float(), v, 10.0, env.speed_cap[lid][:1].float(), spec)
    return float(vv[0, 0]), float(k[0, 0])


abl = {"통로 지도 지움": [], "GT 상자 칸 지움": [], "라이다 평균으로": []}
snaps = []
t = 0.0
for k in range(int(secs / 0.025)):
    REC["new"] = False
    w._step_once(s)
    if not REC.get("new") or k < 40:
        continue
    tp = TP[0]
    o = tp.o[:1].clone()
    nd = o.shape[1]; G0 = nd - 91; S0 = G0 - 108
    v = env.sim.state[lid[:1], 3].float()
    base = v1k1(tp.policy(o), v)
    if REC["gate"]:
        for name in abl:
            o2 = o.clone()
            if name == "통로 지도 지움":
                o2[:, G0:] = 0.0
            elif name == "GT 상자 칸 지움":
                o2[:, PROP] = 0.0
            else:
                o2[:, S0:G0] = tp.nrm.mean[S0:G0]
            a2 = v1k1(tp.policy(o2), v)
            abl[name].append((a2[0] - base[0], a2[1] - base[1], float(o[0, G0:].sum())))
    # snapshots: a few decisions with several cells blocked
    if REC["gate"] and len(snaps) < 4 and float(o[0, G0:].sum()) >= 4 and (not snaps or k * 0.025 - snaps[-1]["t"] > 5.0):
        st = env.sim.state[0].cpu().numpy()
        sres = env.last_result.s[lid][:1]
        offs = torch.tensor(SR.CORRIDOR_S, device="cuda")
        xy, yaw = tr.pose_at_s((sres + offs), env.sim.tid[:1].expand(offs.numel()))
        wd = tr.sample_edt(xy[None], env.sim.tid[:1, None].expand(1, offs.numel()))[0]
        live = po.p_sid[lid[0]] >= 0
        polys = []
        for c in torch.nonzero(live).flatten().tolist():
            pp = po.p_poses[lid[0], c].cpu().numpy()
            nrm = po.p_n[lid[0], c].cpu().numpy(); dd = po.p_d[lid[0], c].cpu().numpy()
            fin = np.isfinite(dd)
            # box corners from the half-planes (axis-aligned in the prop frame: |x| <= a, |y| <= b)
            ax_ = max([dd[i] for i in range(len(dd)) if fin[i] and abs(nrm[i, 0]) > 0.9] + [0.05])
            by_ = max([dd[i] for i in range(len(dd)) if fin[i] and abs(nrm[i, 1]) > 0.9] + [0.05])
            cc, ss = math.cos(pp[2]), math.sin(pp[2])
            pts = [(pp[0] + cc * x - ss * y, pp[1] + ss * x + cc * y) for x, y in ((ax_, by_), (-ax_, by_), (-ax_, -by_), (ax_, -by_))]
            polys.append(pts)
        scan = env.last_result.scan_true[lid[0]].float().cpu().numpy() if env.last_result.scan_true is not None else None
        snaps.append(dict(t=k * 0.025, st=st, grid=o[0, G0:].reshape(13, 7).cpu().numpy(), xy=xy.cpu().numpy(), yaw=yaw.cpu().numpy(),
                          w=wd.cpu().numpy(), polys=polys, gt=o[0, PROP].reshape(3, 4).cpu().numpy(), v=float(v[0]), base=base,
                          scan=scan, ahead=None))

print(f"== {os.path.basename(os.path.dirname(run))} @ {mp} 학습 장애물 {secs:.0f} s: 상자 정책 결정 {len(abl['통로 지도 지움'])}")
for name, L_ in abl.items():
    if not L_:
        continue
    A = np.array(L_)
    busy = A[:, 2] >= 1
    print(f"  {name}: 실행 속도 명령 바뀜 평균 |{np.abs(A[:, 0]).mean():.2f}| m/s (평균 {A[:, 0].mean():+.2f}), "
          f"지도에 막힌 칸 있을 때 |{np.abs(A[busy, 0]).mean() if busy.any() else float('nan'):.2f}| (평균 {A[busy, 0].mean() if busy.any() else float('nan'):+.2f}) | "
          f"첫 곡률 바뀜 |{np.abs(A[:, 1]).mean():.3f}| 1/m")

if snaps:
    U = np.array(SR.CORRIDOR_U); AH = np.array(SR.AHEAD); CS = np.array(SR.CORRIDOR_S)
    seg = np.searchsorted(AH, CS)
    fig = plt.figure(figsize=(16, 7.5 * len(snaps)))
    for r_, sn in enumerate(snaps):
        ax = fig.add_subplot(len(snaps), 2, 2 * r_ + 1)
        ax.imshow(occ, cmap="gray_r", origin="lower", extent=ext, alpha=0.45, vmin=0, vmax=1)
        # cells: each dense row (0.4 m) x 7 bins, coloured by its 13-row segment's value
        for i in range(len(CS) - 1):
            k_ = seg[i]
            if k_ >= 13:
                continue
            for b in range(7):
                u0, u1 = U[2 * b] - 0.07, U[2 * b + 1] + 0.07
                corners = []
                for (ii, uu) in ((i, u0), (i, u1), (i + 1, u1), (i + 1, u0)):
                    x, y = sn["xy"][ii]; yw = sn["yaw"][ii]; wd = sn["w"][ii]
                    corners.append((x - uu * wd * math.sin(yw), y + uu * wd * math.cos(yw)))
                blocked = sn["grid"][k_, b] > 0.5
                ax.add_patch(Polygon(corners, closed=True, facecolor=(1.0, 0.55, 0.15, 0.55) if blocked else (0.6, 0.65, 0.75, 0.12),
                                     edgecolor=(0.3, 0.3, 0.3, 0.25), lw=0.3))
        for pts in sn["polys"]:
            ax.add_patch(Polygon(pts, closed=True, facecolor="saddlebrown", edgecolor="k", lw=0.8))
        st = sn["st"]
        # the 3 nearest props as the GT slots carry them: centre only (dx, dy in the body frame /5)
        for q in sn["gt"]:
            if q[0] > 0.5:
                dx, dy = q[1] * 5, q[2] * 5
                gx = st[0] + dx * math.cos(st[2]) - dy * math.sin(st[2]); gy = st[1] + dx * math.sin(st[2]) + dy * math.cos(st[2])
                ax.plot(gx, gy, "x", color="lime", ms=10, mew=2.5)
        if sn["scan"] is not None:
            n = len(sn["scan"]); ang = np.linspace(-0.75 * math.pi, 0.75 * math.pi, n)
            r = np.minimum(sn["scan"], 10.0)
            ax.plot(st[0] + r * np.cos(st[2] + ang), st[1] + r * np.sin(st[2] + ang), ".", color="tab:blue", ms=2)
        ax.add_patch(Polygon([(st[0] + math.cos(st[2]) * x - math.sin(st[2]) * y, st[1] + math.sin(st[2]) * x + math.cos(st[2]) * y)
                              for x, y in ((0.25, 0.15), (-0.25, 0.15), (-0.25, -0.15), (0.25, -0.15))], closed=True, facecolor="tab:red"))
        cx, cy = sn["xy"][len(CS) // 3]
        ax.set_xlim(st[0] - 6, st[0] + 6); ax.set_ylim(st[1] - 6, st[1] + 6); ax.set_aspect("equal")
        ax.set_title(f"{sn['t']:.1f} s, 속도 {sn['v']:.1f} m/s, 실행 속도 명령 {sn['base'][0]:.1f} m/s | 주황 = 지도에서 막힘, 회색 = 빔, "
                     f"갈색 = 실제 상자, 초록 x = GT 상자 칸(중심만), 파랑 = 라이다 108줄", fontsize=9)
        bx = fig.add_subplot(len(snaps), 2, 2 * r_ + 2)
        bx.imshow(sn["grid"][::-1], cmap="Oranges", vmin=0, vmax=1.5, aspect="auto")
        bx.set_yticks(range(13)); bx.set_yticklabels([f"{a:g} m" for a in AH[::-1]])
        bx.set_xticks(range(7)); bx.set_xticklabels(["왼쪽 끝", "", "", "가운데", "", "", "오른쪽 끝"])
        for i in range(13):
            for j in range(7):
                bx.text(j, i, f"{int(sn['grid'][::-1][i, j])}", ha="center", va="center", fontsize=9)
        bx.set_title("선생이 받는 통로 지도 그대로 (줄 = 앞으로 그 거리까지의 구간, 칸 = 트랙 폭을 7등분, 1 = 막힘)", fontsize=9)
    fig.tight_layout(); fig.savefig(head + ".png", dpi=60); plt.close(fig)
    print("그림", head + ".png")
