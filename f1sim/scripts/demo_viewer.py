"""Drive the teacher on a track and watch it in the native 3D viewer (or stream to the web monitor)."""
import argparse, torch
from f1sim import Track, Config, Simulator
from f1sim.raceline import Raceline
from f1sim.teacher import RacelineTeacher

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--cars", type=int, default=8)
ap.add_argument("--map", default="", help="catalog name (gen:competition:3, rt:Spielberg, gym:levine) or map yaml; comma-separated for a multi-track sim")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--track", type=int, default=0, help="which track the viewer shows in a multi-track sim")
ap.add_argument("--fast", action="store_true", help="no real-time pacing (as fast as it renders)")
ap.add_argument("--web", action="store_true", help="stream to the browser monitor on :8765 instead of opening a window")
ap.add_argument("--headless-shots", default="", help="render N frames headless and save screenshots to this dir")
ap.add_argument("--no-shadows", action="store_true")
ap.add_argument("--manual", action="store_true", help="drive car 0 yourself (W/S A/D, Shift boost, Space handbrake, R reset); others follow the teacher")
a = ap.parse_args()

from f1sim import maps
tracks = maps.load_many(a.map.split(",")) if a.map else [Track.generate_random(a.seed, style="competition")]
print("building racelines (first time per track: ~10 s, cached afterwards) ...", flush=True)
rl = [Raceline.build_cached(t) for t in tracks]
cfg = Config(); cfg.sim.terminate_on_collision = False
sim = Simulator(tracks, cfg, num_envs=a.cars, device=a.device)
sim.reset(poses=sim.sample_spawn(a.cars, 0.2, 0.1, torch.linspace(0, 6, a.cars).to(sim.device), tid=sim.tid))
teacher = RacelineTeacher(rl, wheelbase=cfg.vehicle.lf + cfg.vehicle.lr, device=sim.device)
import time; t0 = time.time(); print("compiling simulator kernels (first run: 10-20 s) ...", flush=True)
sim.warmup(); print(f"ready in {time.time() - t0:.1f} s", flush=True)
state = {"r": None}

manual = None
def step():
    act = teacher(state["r"].state if state["r"] is not None else sim.state, sim.P, sim.tid)
    if manual is not None:
        model, ramp, viewer = manual
        k = viewer.keys_held()
        if k.get("reset"):
            sim.reset(torch.tensor([0], device=sim.device)); model.reset()
        inp = ramp.update(k.get("left", False), k.get("right", False), k.get("up", False), k.get("down", False),
                          sim.control_dt, boost=k.get("boost", False), handbrake=k.get("handbrake", False))
        v_meas = float(sim.state[0, 3])
        steer, v = model.update(inp, v_meas, sim.control_dt)
        act[0, 0], act[0, 1] = steer, v
        viewer.extra_hud = model.hud(inp, v_meas)
    r = sim.step(act)
    state["r"] = r
    return r

if a.web:
    from f1sim.viewer import Viewer
    viewer = Viewer(sim, raceline=rl)
    while True:
        viewer.publish(step())
        if not a.fast: viewer.sync()
else:
    from f1sim.viewer.native import NativeViewer, MODES
    if a.headless_shots:
        import os; os.makedirs(a.headless_shots, exist_ok=True)
        v = NativeViewer(sim, raceline=rl, headless=True, shadows=not a.no_shadows, track_index=a.track)
        for i in range(240): v.update(step())
        for m in MODES:
            v.mode = MODES.index(m); v.render(); v.render(); v.screenshot(os.path.join(a.headless_shots, f"{m}.png"))
        print("saved", a.headless_shots)
    else:
        v = NativeViewer(sim, raceline=rl, shadows=not a.no_shadows, track_index=a.track)
        if a.manual:
            from f1sim.teleop import DriveModel, KeyRamp, TeleopConfig
            tcfg = TeleopConfig(); manual = (DriveModel(tcfg), KeyRamp(tcfg), v)
        try:
            v.run(step, realtime=not a.fast)
        except KeyboardInterrupt:
            pass
        v.close()
