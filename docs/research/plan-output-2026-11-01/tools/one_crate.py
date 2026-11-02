"""상자 하나를 혼자 주행선 위에 (11-03): stall_view에서 상자 정책(--vis-actor)이 바로 앞 상자를 7.4 m부터 계속 보면서도
2.3 -> 6.3 m/s로 가속해 박음(배치 5), 머리핀 상자를 2.6 m에서 보고도 3.7 m/s 유지하다 밀려 박음(배치 20) -- 정보는 있는데 안 씀?
가장 깨끗한 판별: 깨끗한 맵에 상자 하나만, 혼자 정책이 실제로 달린 선 위 12곳(랩을 시간으로 12등분)에 하나씩 놓고 콘솔 워커로 12 s씩.
곳마다: 상자 접촉(세게), 상자 옆을 지날 때 속도, 그 랩 시간. 상자 하나를 못 피하면 상자 정책은 정보를 쓰지 않는 것.
콘솔과 같은 SimWorker 세션, 상자는 `prop_layout.LayoutHold`(콘솔 장애물 편집과 같은 길)로 놓음, 리셋마다 같은 상자.
인자: 맵id 상자모양(가로x세로 cm, 예 30x34) 체크포인트...(이름=경로)"""
import os, sys, math
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer import prop_layout as PL
from f1sim.viewer.console import protocol as P
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


mp, want = sys.argv[1], sys.argv[2]
runs = [x.split("=", 1) for x in sys.argv[3:]]
i_on, i_hd = CC.index("onsets_prop"), CC.index("hard_prop")
N_POS, SECS = 12, 12.0
line = None
for name, path in runs:
    w = sim_worker.SimWorker(_N(), _N())
    w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
    s = w.build_session(P.SessionConfig(run=path, map_name=mp, seed=0, races=1, cars_per_race=1, device="cuda", compile=False,
                                        collision_mode="soft", procedural=True, episode_s=3600.0, randomize=False), 0)
    env = s["env"]; proc = env.procedural
    cat = PL.catalogue(proc)
    shape = next((e for e in cat if e["label"].endswith(want.replace("x", "×") + " cm")), None)
    if shape is None:
        print("모양 목록:", ", ".join(e["label"] for e in cat)); sys.exit(1)
    hold = PL.LayoutHold(proc, int(proc.C), env.sim.tid[:1]).install()

    def run(pieces, secs):
        hold.set_pieces(pieces)
        s["obs"], _ = env.reset()
        cf = env.sim.contact_frame()[0].float().cpu().numpy(); on0, hd0 = cf[i_on], cf[i_hd]
        xs, laps, lap0, t_last = [], [], None, None
        for k in range(int(secs / 0.025)):
            w._step_once(s)
            st = env.sim.state[0]
            xs.append((float(st[0]), float(st[1]), float(st[2]), float(st[3])))
            lp = int(env.last_result.lap[0])
            if lap0 is None:
                lap0 = lp
            elif lp != lap0:
                t = (k + 1) * 0.025
                if t_last is not None:
                    laps.append(round(t - t_last, 2))
                t_last, lap0 = t, lp
        cf = env.sim.contact_frame()[0].float().cpu().numpy()
        return np.array(xs), laps, int(round(cf[i_on] - on0)), int(round(cf[i_hd] - hd0)), PL.read(proc)

    if line is None:                                      # the solo line: the first run's path with no crate
        X, laps, _, _, _ = run([], 10.0)
        k1 = 120                                          # after the launch
        X = X[k1:k1 + int(round((laps[0] if laps else 6.5) / 0.025))]
        idx = np.linspace(0, len(X) - 1, N_POS, endpoint=False).astype(int)
        line = [(float(X[i, 0]), float(X[i, 1]), float(X[i, 2])) for i in idx]
        print(f"혼자 선: 랩 {laps}, 상자 자리 {N_POS}곳 (모양 {shape['label']})", flush=True)
    tot_c = tot_h = n_bad = 0; lap_all = []
    out = []
    for j, (x, y, yaw) in enumerate(line):
        X, laps, c, h, after = run([{"sid": shape["id"], "x": x, "y": y, "yaw": yaw}], SECS)
        d = np.hypot(X[:, 0] - x, X[:, 1] - y)
        kk = int(np.argmin(d))
        moved = math.hypot(after[0]["x"] - x, after[0]["y"] - y) if after else float("nan")
        out.append(f"{j}:{'접촉' + str(c) + '(세게' + str(h) + ')' if c else '깨끗'} 최소거리 {d.min():.2f} m 그때 속도 {X[kk, 3]:.1f} 상자밀림 {moved:.2f} m 랩 {laps}")
        tot_c += c; tot_h += h; n_bad += int(c > 0); lap_all += laps
    print(f"== {name}: 상자 하나 {N_POS}곳 중 접촉 {n_bad}곳 (접촉 {tot_c}, 세게 {tot_h}), 랩 중앙 {np.median(lap_all) if lap_all else float('nan'):.2f} s", flush=True)
    for o in out:
        print("   ", o, flush=True)
    hold.release()
    del w, s, env
    torch.cuda.empty_cache()
