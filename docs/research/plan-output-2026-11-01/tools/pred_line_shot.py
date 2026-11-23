"""(11-24, 경주 단계) 콘솔 3D 화면에 상대 차 예측 선이 그려지나 -- 화면 캡처 (QT_QPA_PLATFORM=offscreen).

콘솔과 같은 길로: 우리 차는 콘솔의 plan 주행기(PlanRunner, plan대로 정확히)가 몰고, 상대 차는 기준 policy 복사본(배율 0.6)이 몲, 5-15 m
뒤에서 출발. 결정마다 PlanRunner가 남긴 예측(pred_w)을 보고, 예측이 '보임'이고 상대가 가까운 순간들에서 화면 한 장씩: 프레임은 콘솔
worker의 `SimWorker._snapshot`(예측 선을 drv 선으로 넣는 곳)이 만들고, 그리는 것은 콘솔의 ViewportWidget(추격 카메라).
인자: 예측 머리 있는 checkpoint 상대 차 checkpoint 출력 접두어 [장수 3] [초 12]
"""
import os, sys, time, types, collections
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["F1SIM_NO_STAGGER"] = "1"
import numpy as np
import torch
torch.set_grad_enabled(False)

ck_path, ref_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
n_shot = int(sys.argv[4]) if len(sys.argv) > 4 else 3
secs = float(sys.argv[5]) if len(sys.argv) > 5 else 12.0
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
from f1sim.learn.plan_policy import lidar_mount_of, load_actor2, PlanRunner
os.environ["F1SIM_LIDAR_MOUNT_X"] = str(lidar_mount_of(ck))
os.environ["F1SIM_PLAN_LAT_MARGIN"] = str((ck.get("args") or {}).get("plan_lat_margin", 0.95))
from f1sim.learn import plan_rl as PR
from f1sim import plan_steps as PS, maps

W = os.path.join(os.path.dirname(PR.WORLD), "plan_world_korea26scan_race.json")
G = 4
env, lim, a = PR.build(W, 2 * G, 21, dev, overrides={"opp_present": 1.0, "spawn_half_lap": 0.0, "prop_fraction": 0.0, "warps": 0,
                                                       "align_episodes": True})
env.ep_limit[:] = 10 ** 6
runner = PlanRunner(env, ck, dev, mode="ideal")
assert runner.pred_w is not None, "이 checkpoint에는 예측 머리가 없음"
so = PR.SelfOpponents(env, lim, 1.0, seed=2, max_pool=1, p_newest=1.0, pace=(0.6, 0.6, 0.0))
so.add(load_actor2(torch.load(ref_path, map_location=dev, weights_only=False), dev), 500)
lid = env.learner_ids; others = env.sim.other_idx[lid][:, 0]
act0 = torch.zeros(env.B, env.act_dim, device=dev)
T = int(round(secs / PS.DT))
best = []                                                 # (score, decision, race, saved snapshot of the moment)
from f1sim.viewer import sim_worker as SW
stub = types.SimpleNamespace(_overlay={"plan": True, "saliency": False, "internals": False}, gen=0, seq=0, _sim_rate=1.0,
                             _sal_cache=None)
stub._viz_failure = lambda *a_: None


def frame_of(focus):
    session = {"env": env, "cfg": types.SimpleNamespace(max_render_cars=env.B), "focus": focus, "mode": "plan", "act_fn": runner,
               "viz": None, "gg": collections.deque(maxlen=200), "info_line": "", "mtime": time.time(), "controller": None}
    return SW.SimWorker._snapshot(stub, session)


frames = []
for t in range(T):
    so.plan()
    for j in range(PS.SUB):
        runner()
        so.command(j)
        _, _, te, trc, _ = env.step(act0)
        so.stepped(te | trc)
    so.after()
    vis = runner.pred_w[:, -1]
    d = (env.sim.state[others, :2] - env.sim.state[lid, :2]).norm(dim=1)
    for g in range(G):
        if t > 20 and float(vis[g]) >= 0.5 and float(d[g]) < 8.0:
            frames.append((float(vis[g]), t, g, frame_of(int(lid[g]))))
    if t % 20 == 0:
        print(f"  {t * PS.DT:.0f} s: 보임 확률 {[round(float(x), 2) for x in vis]}, 거리 {[round(float(x), 1) for x in d]}", flush=True)
print(f"예측이 '보임'이고 8 m 안: {len(frames)}개 순간", flush=True)
if not frames:
    raise SystemExit("그릴 순간이 없음")
frames.sort(key=lambda x: -x[0])
pick, seen = [], set()
for f in frames:                                          # different moments, at least 1 s apart
    if all(abs(f[1] - p[1]) >= 10 for p in pick):
        pick.append(f)
    if len(pick) >= n_shot:
        break
# the map as the console draws it
from f1sim.viewer.console import app as console_app
from f1sim.viewer.console.viewport import ViewportWidget, TrackGeometry
from f1sim.viewer.console.frames import FrameBuffer
from f1sim.viewer.console.env_editor import _geom_kwargs
from PyQt5 import QtWidgets
app = QtWidgets.QApplication.instance() or console_app.create_app(["shot"])
track_name = str(a.tracks).split(",")[0]
tr = maps.load(track_name)
geom = TrackGeometry(**_geom_kwargs(SW.build_track_geometry(tr, None)))
for k, (vp_, t_, g_, fr) in enumerate(pick):
    vp = ViewportWidget()
    vp.resize(1400, 900)
    vp.set_geometry(geom)
    buf = FrameBuffer(capacity=4)
    buf.set_generation(1)
    fr = dict(fr); fr["gen"] = 1; fr["seq"] = 1
    vp.attach(buf)
    vp.show()
    for _ in range(5):
        app.processEvents(); time.sleep(0.05)
    vp.set_camera("chase")
    vp.plan = fr.get("plan"); vp.plan_pred = fr.get("plan_pred"); vp.drv = fr.get("drv")
    ok = buf.push(fr)
    for _ in range(20):
        vp.update(); app.processEvents(); time.sleep(0.05)
    img = vp.grabFramebuffer()
    fn = f"{out}_{k}.png"
    img.save(fn)
    line = fr.get("drv", {}).get("lines", [None])[-1]
    print(f"화면 {fn}: {t_ * PS.DT:.1f} s, 경주 {g_}, 예측 '보임' {vp_:.2f}, 프레임 받음 {ok}, 예측 선 점 "
          f"{None if line is None else np.round(line['xy'][[0, -1]], 2).tolist()}, 상대 차 지금 "
          f"({fr['x'][int(others[g_])]:.2f}, {fr['y'][int(others[g_])]:.2f})", flush=True)
    vp.close()
