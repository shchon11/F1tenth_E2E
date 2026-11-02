"""콘솔 워커(SimWorker) 한 세션에서 1:1 경주를 여러 판 (11-02): 콘솔의 리셋(CMD_RESET = env.reset())처럼 판마다 출발을
새로 뽑아 초씩. 판마다 추월 / 추월당함(간격이 -1 m 아래에서 +1 m 위로 = 추월; 나란히 엉켜 0을 오가는 것은 세지 않음),
차 접촉(콘솔 계기와 같은 `sim.contact_frame`, 세게 = 콘솔 기준), 벽·상자 접촉, 멈춘 시간, 끝 간격, 내 차 평균 속도.
추월이 없거나 차와 세게 부딪힌 판은 그림(두 차 경로 + 접촉 + 간격 곡선).
인자: 체크포인트 맵id 판당초 판수 상대종류 상대속도배율 시드 그림머리 [학습장애물 0/1]"""
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC


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
from f1sim import maps, tracks
tr = maps.load(tracks.resolve(mp))
from matplotlib import font_manager
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
occ = tr.occupancy; res = tr.resolution; H, W = occ.shape
ext = [tr.origin[0], tr.origin[0] + W * res, tr.origin[1], tr.origin[1] + H * res]
i_on = [CC.index(k) for k in ("onsets_wall", "onsets_prop", "onsets_car")]
i_hd = [CC.index(k) for k in ("hard_wall", "hard_prop", "hard_car")]
tot = dict(passes=0, passed=0, car=0, car_hard=0, wall=0, prop=0, stuck=0.0, no_pass=0, v=[])
t_all = time.time()
for L in range(n_race):
    if L > 0:
        s["obs"], _ = env.reset()                                    # the console's 리셋
    cf0 = env.sim.contact_frame()[:2].float().cpu().numpy()
    gap = None; side = 0; passes = passed = 0
    xy0, xy1, vs, gaps, hits = [], [], [], [], []
    prev_c, prev_h = cf0[0, i_on[2]], cf0[0, i_hd[2]]
    for k in range(int(secs / 0.025)):
        w._step_once(s)
        r = env.last_result
        if gap is None:
            gap = float(r.s[0] - r.s[1])
            Lt = float(env.sim.track.length[env.sim.tid[0]])                # (T,) per track
            gap = (gap + 0.5 * Lt) % Lt - 0.5 * Lt                   # the shorter way round
            side = -1 if gap < -1.0 else (1 if gap > 1.0 else 0)
        else:
            gap += float(r.progress[0] - r.progress[1])
        if side <= 0 and gap > 1.0:
            passes += int(side < 0); side = 1
        elif side >= 0 and gap < -1.0:
            passed += int(side > 0); side = -1
        st = env.sim.state
        xy0.append(st[0, :2].cpu().numpy().copy()); xy1.append(st[1, :2].cpu().numpy().copy())
        vs.append(float(st[0, 3])); gaps.append(gap)
        cf = env.sim.contact_frame()[0].float().cpu().numpy()
        if cf[i_on[2]] > prev_c:
            hits.append((k * 0.025, xy0[-1][0], xy0[-1][1], bool(cf[i_hd[2]] > prev_h)))
        prev_c, prev_h = cf[i_on[2]], cf[i_hd[2]]
    cf = env.sim.contact_frame()[0].float().cpu().numpy()
    on = np.round(cf[i_on] - cf0[0, i_on]).astype(int); hd = np.round(cf[i_hd] - cf0[0, i_hd]).astype(int)
    vs_a = np.asarray(vs); stuck = 0.025 * int(np.sum(np.abs(vs_a[120:]) < 0.3))
    tot["passes"] += passes; tot["passed"] += passed; tot["car"] += int(on[2]); tot["car_hard"] += int(hd[2])
    tot["wall"] += int(on[0]); tot["prop"] += int(on[1]); tot["stuck"] += stuck; tot["v"].append(float(vs_a.mean()))
    start = gaps[0]
    bad = (start < -1.0 and passes == 0) or hd[2] > 0
    tot["no_pass"] += int(start < -1.0 and passes == 0)
    flag = ""
    if bad:
        a0, a1 = np.asarray(xy0), np.asarray(xy1)
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_axes([0.03, 0.30, 0.94, 0.65])
        ax.imshow(~occ, cmap="gray", origin="lower", extent=ext, alpha=0.6)
        t = np.arange(len(a0)) * 0.025
        sc0 = ax.scatter(a0[:, 0], a0[:, 1], c=t, s=3, cmap="viridis")
        ax.scatter(a1[:, 0], a1[:, 1], c="orange", s=1, alpha=0.5)
        for ht, x, y, h in hits:
            ax.plot(x, y, "x", color="red", ms=(14 if h else 8), mew=(3 if h else 1.5))
        ax.set_xlim(min(a0[:, 0].min(), a1[:, 0].min()) - 1, max(a0[:, 0].max(), a1[:, 0].max()) + 1)
        ax.set_ylim(min(a0[:, 1].min(), a1[:, 1].min()) - 1, max(a0[:, 1].max(), a1[:, 1].max()) + 1)
        ax.set_aspect("equal"); plt.colorbar(sc0, ax=ax, fraction=0.03, label="내 차 시간 s")
        ax.set_title(f"{os.path.basename(os.path.dirname(run))}/{os.path.basename(run)} @ {mp} vs {kind} x{sc}, 판 {L} (시드 {seed}): "
                     f"추월 {passes} / 추월당함 {passed} | 차 접촉 {on[2]} (세게 {hd[2]}) | 주황 = 상대")
        bx = fig.add_axes([0.06, 0.05, 0.9, 0.2])
        bx.plot(t, vs_a, label="내 차 속도 m/s"); bx.plot(t, np.asarray(gaps), label="간격 m (+ 앞섬)")
        bx.axhline(0, color="k", lw=0.5); bx.legend(); bx.grid(alpha=0.3); bx.set_xlabel("시간 s")
        png = f"{head}_R{L}.png"; fig.savefig(png, dpi=55); plt.close(fig)
        flag = f" | 그림 {png}"
    print(f"판 {L}: 출발 간격 {start:+.1f} m | 추월 {passes} / 추월당함 {passed} | 차 접촉 {on[2]} (세게 {hd[2]}) | "
          f"벽 {on[0]} 장애물 {on[1]} | 멈춤 {stuck:.1f} s | 끝 간격 {gap:+.1f} m | 평균 속도 {vs_a.mean():.2f}{flag}", flush=True)
print(f"합계 {n_race}판 x {secs:.0f} s vs {kind} x{sc} ({time.time() - t_all:.0f} s 걸림): 추월 {tot['passes']} / 추월당함 {tot['passed']} | "
      f"뒤에서 출발해 추월 못 한 판 {tot['no_pass']} | 차 접촉 {tot['car']} (세게 {tot['car_hard']}) | 벽 {tot['wall']} 장애물 {tot['prop']} | "
      f"멈춤 {tot['stuck']:.1f} s | 평균 속도 {np.mean(tot['v']):.2f}", flush=True)
