"""상자 있을 때 선생이 망가지는 순간 진단 (11-03 02:00, 사용자: "깨끗한 맵에서 혼자 달릴 때 선생은 너무 잘해. 장애물만 있어도
출력이 망가지는 것 같아. 갑자기 속도가 너무 줄거나 그냥 벽으로 돌진. 상대차도 없어"). 콘솔 워커(SimWorker) 한 세션, 콘솔 리셋과 같은
배치마다, 선생(SoloMix)의 결정마다 기록: 누가 운전했나(상자 정책 / 혼자 정책), 두 정책의 명령(경로점 4개 속도·곡률 -- 운전 안 한
쪽도 같은 상태에서 계산), 실제 속도, 벽까지 거리, 가장 가까운 상자 거리. 사건: (가) 급감속 -- 실행된 첫 경로점 속도가 0.5 s 안에
2 m/s 넘게 떨어짐, (나) 벽 접근 -- 벽까지 0.10 m 아래 또는 벽 접촉. 사건마다 누가 운전했나, 혼자 정책이면 뭘 했을지, 상자 거리.
사건 장면 그림(앞뒤 경로 + 상자 + 그 순간 두 정책의 궤적, 아래에 속도·명령·운전자 곡선).
인자: 체크포인트 맵id 배치당초 배치수 시드 그림머리"""
import os, sys, math, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.learn import state_rl as SR
from f1sim import mpc
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, secs, n_lay, seed, head = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
REC = {}
_orig = SR.SoloMix.__call__


def _rec_call(self, a_pol, last):
    out = _orig(self, a_pol, last)
    o = SR.blank_opponents(self.sobs(last.clamp(-1, 1)))
    o[:, self.prop_cols] = 0.0
    REC["a_pol"] = a_pol[0].detach().clone(); REC["a_b"] = self.net.pi(self.nrm(o)).clamp(-1, 1)[0].detach().clone()
    REC["gate"] = bool(self.gate[0]); REC["new"] = True
    return out


SR.SoloMix.__call__ = _rec_call
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=True, episode_s=3600.0, randomize=False), 0)
env = s["env"]; po = env.procedural
inner = env.tracker._inner if hasattr(env.tracker, "_inner") else env.tracker
spec = inner.spec
lay = mpc._LAYOUTS["wp4"]
from f1sim import maps, tracks
tr = maps.load(tracks.resolve(mp))
from matplotlib import font_manager
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
occ = tr.occupancy; res = tr.resolution; H, W = occ.shape
ext = [tr.origin[0], tr.origin[0] + W * res, tr.origin[1], tr.origin[1] + H * res]
i_wall = CC.index("onsets_wall"); i_prop = CC.index("onsets_prop")
VMAX = 10.0


def speeds(a):
    return ((a[lay.speed].float() + 1.0) * 0.5 * VMAX).cpu().numpy()


def kap(a):
    return a[lay.path].float().cpu().numpy()


def prop_xy():
    if po is None or po.C == 0:
        return np.zeros((0, 2))
    live = po.p_sid[0] >= 0
    return po.p_poses[0][live][:, :2].cpu().numpy()


rows = []                       # one per decision
PROPS = {}
events = []
t0 = time.time()
for L in range(n_lay):
    if L > 0:
        s["obs"], _ = env.reset()
    props = prop_xy(); PROPS[L] = props
    cf_prev = env.sim.contact_frame()[0].float().cpu().numpy()
    start = len(rows)
    for k in range(int(secs / 0.025)):
        REC["new"] = False
        w._step_once(s)
        st = env.sim.state[0]
        cf = env.sim.contact_frame()[0].float().cpu().numpy()
        wall_on = cf[i_wall] > cf_prev[i_wall]; prop_on = cf[i_prop] > cf_prev[i_prop]; cf_prev = cf
        if not REC.get("new") and not wall_on and not prop_on:
            continue
        x, y, v = float(st[0]), float(st[1]), float(st[3])
        pp = prop_xy()
        dp = float(np.min(np.hypot(pp[:, 0] - x, pp[:, 1] - y))) if len(pp) else 99.0
        r = dict(L=L, t=k * 0.025, x=x, y=y, yaw=float(st[2]), v=v, wall=float(env.last_result.wall_dist[0]), dp=dp,
                 wall_on=bool(wall_on), prop_on=bool(prop_on))
        if REC.get("new"):
            r.update(gate=REC["gate"], vp=speeds(REC["a_pol"]), vb=speeds(REC["a_b"]), kp=kap(REC["a_pol"]), kb=kap(REC["a_b"]))
        elif rows and rows[-1]["L"] == L and "gate" in rows[-1]:
            r.update({q: rows[-1][q] for q in ("gate", "vp", "vb", "kp", "kb")})
        if "gate" in r:
            rows.append(r)
    print(f"배치 {L}: 상자 {len(props)}개, {len(rows) - start}결정, {time.time() - t0:.0f} s", flush=True)

# ---------------------------------------------------------------- summary
dec = rows
g = np.array([r["gate"] for r in dec]); v = np.array([r["v"] for r in dec])
vx = np.array([(r["vp"] if r["gate"] else r["vb"])[0] for r in dec])           # executed first-waypoint speed
vp0 = np.array([r["vp"][0] for r in dec]); vb0 = np.array([r["vb"][0] for r in dec])
dp = np.array([r["dp"] for r in dec]); wl = np.array([r["wall"] for r in dec])
print(f"\n== {run} @ {mp}: {n_lay}배치 x {secs:.0f} s, 결정 {len(dec)}")
print(f"상자 정책이 운전한 시간 {g.mean() * 100:.1f} % | 실제 속도: 상자 정책 운전 중 {v[g].mean():.2f}, 혼자 정책 운전 중 {v[~g].mean():.2f} m/s")
print(f"같은 상태에서 첫 경로점 속도 명령(상자 정책 운전 중): 상자 정책 {vp0[g].mean():.2f} vs 혼자 정책이었다면 {vb0[g].mean():.2f} m/s"
      f" | 상자 정책이 2 m/s 넘게 낮게 부른 결정 {np.mean((vb0[g] - vp0[g]) > 2.0) * 100:.1f} %")
# where the props policy drives: by distance to the nearest prop
for lo, hi in ((0, 1.5), (1.5, 3), (3, 6), (6, 99)):
    m = g & (dp >= lo) & (dp < hi)
    if m.sum() > 5:
        print(f"  상자까지 {lo}~{hi} m: 상자 정책 운전 {m.sum()}결정, 명령 {vp0[m].mean():.2f} (혼자였다면 {vb0[m].mean():.2f}) m/s, "
              f"실제 {v[m].mean():.2f}, 벽까지 중앙 {np.median(wl[m]):.2f} m")
# events
ev = []
for i, r in enumerate(dec):
    j = i - 1
    while j >= 0 and dec[j]["L"] == r["L"] and r["t"] - dec[j]["t"] < 0.5:
        j -= 1
    j += 1
    if j < i and dec[j]["L"] == r["L"]:
        prev_cmd = max((dec[q]["vp"] if dec[q]["gate"] else dec[q]["vb"])[0] for q in range(j, i))
        cmd = (r["vp"] if r["gate"] else r["vb"])[0]
        if prev_cmd - cmd > 2.0:
            ev.append(("급감속", i, prev_cmd - cmd))
    if r["wall_on"] or r["wall"] < 0.10:
        ev.append(("벽", i, r["wall"]))
# one event per 1 s
kept = []
for kind, i, mag in ev:
    if kept and kept[-1][0] == kind and dec[kept[-1][1]]["L"] == dec[i]["L"] and dec[i]["t"] - dec[kept[-1][1]]["t"] < 1.0:
        continue
    kept.append((kind, i, mag))
for kind in ("급감속", "벽"):
    E = [e for e in kept if e[0] == kind]
    if not E:
        print(f"{kind}: 0건"); continue
    gg = [dec[i]["gate"] for _, i, _ in E]
    # who drove in the 1 s before
    pre = []
    for _, i, _ in E:
        q = [dec[k]["gate"] for k in range(max(0, i - 20), i + 1) if dec[k]["L"] == dec[i]["L"] and dec[i]["t"] - dec[k]["t"] <= 1.0]
        pre.append(np.mean(q))
    sw = []
    for _, i, _ in E:
        q = [dec[k]["gate"] for k in range(max(0, i - 20), i + 1) if dec[k]["L"] == dec[i]["L"] and dec[i]["t"] - dec[k]["t"] <= 1.0]
        sw.append(any(a != b for a, b in zip(q, q[1:])))
    alt = [dec[i]["vb"][0] - dec[i]["vp"][0] for _, i, _ in E if dec[i]["gate"]]
    print(f"{kind}: {len(E)}건 | 그 순간 상자 정책 운전 {sum(gg)}건 | 1 s 안에 운전자 바뀜 {sum(sw)}건 | 상자까지 중앙 "
          f"{np.median([dec[i]['dp'] for _, i, _ in E]):.2f} m | 상자 정책 운전 중이면 혼자 정책은 {np.mean(alt) if alt else float('nan'):+.2f} m/s 더 높게 불렀을 것")
np.save(head + "_rows.npy", np.array([(r["L"], r["t"], r["x"], r["y"], r["v"], r["gate"], r["vp"][0], r["vb"][0], r["dp"], r["wall"])
                                      for r in dec]))

# ---------------------------------------------------------------- pictures: up to 6 of each kind
pick = [e for e in kept if e[0] == "급감속"][:6] + [e for e in kept if e[0] == "벽"][:6]
if pick:
    n = len(pick); fig = plt.figure(figsize=(18, 5.2 * n))
    for row, (kind, i, mag) in enumerate(pick):
        r = dec[i]
        ax = fig.add_subplot(n, 2, 2 * row + 1)
        ax.imshow(occ, cmap="gray_r", origin="lower", extent=ext, alpha=0.5, vmin=0, vmax=1)
        seg = [q for q in dec if q["L"] == r["L"] and r["t"] - 2.0 <= q["t"] <= r["t"] + 1.0]
        xs = np.array([q["x"] for q in seg]); ys = np.array([q["y"] for q in seg]); cs = ["tab:red" if q["gate"] else "tab:blue" for q in seg]
        ax.scatter(xs, ys, c=cs, s=6)
        ax.plot(r["x"], r["y"], "k*", ms=14)
        pr = PROPS.get(r["L"], np.zeros((0, 2)))
        if len(pr):
            ax.plot(pr[:, 0], pr[:, 1], "s", color="saddlebrown", ms=9, alpha=0.8)
        ax.set_xlim(r["x"] - 3, r["x"] + 3); ax.set_ylim(r["y"] - 3, r["y"] + 3); ax.set_aspect("equal")
        ax.set_title(f"{kind} 배치 {r['L']} {r['t']:.1f} s | 빨강 = 상자 정책 운전, 파랑 = 혼자 정책 | 상자까지 {r['dp']:.1f} m, 벽까지 {r['wall']:.2f} m", fontsize=9)
        bx = fig.add_subplot(n, 2, 2 * row + 2)
        seg2 = [q for q in dec if q["L"] == r["L"] and r["t"] - 3.0 <= q["t"] <= r["t"] + 2.0]
        tt = np.array([q["t"] for q in seg2])
        bx.plot(tt, [q["v"] for q in seg2], "k-", label="실제 속도")
        bx.plot(tt, [q["vp"][0] for q in seg2], "r-", lw=1, label="상자 정책 명령(첫 경로점)")
        bx.plot(tt, [q["vb"][0] for q in seg2], "b-", lw=1, label="혼자 정책 명령(첫 경로점)")
        bx.fill_between(tt, 0, 10, where=[q["gate"] for q in seg2], color="red", alpha=0.08, step="mid")
        bx.plot(tt, [min(q["dp"], 10) for q in seg2], "-", color="saddlebrown", lw=0.8, label="상자까지 m")
        bx.plot(tt, [min(q["wall"], 2) * 5 for q in seg2], ":", color="gray", label="벽까지 m x5")
        bx.axvline(r["t"], color="k", lw=0.5); bx.set_ylim(0, 10); bx.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(head + "_events.png", dpi=55); plt.close(fig)
    print("그림", head + "_events.png")
