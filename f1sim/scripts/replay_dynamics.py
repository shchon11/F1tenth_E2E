"""Open-loop replay of real competition driving through the simulator's dynamics.
1 s segments; each starts from the measured speed / yaw rate / steer and is driven by the car's own /drive commands.
Compares simulated yaw rate and speed to the gyro and wheel speed, binned by the real lateral acceleration."""
import sys, glob, json, numpy as np, torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.calib import bagread as br
from f1sim.track import Track
from f1sim.params import Config
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim import dynamics as dyn
from scipy import ndimage
DT = 0.025; SEG = 40; SKIP = 10                          # 1 s segments, score after 0.25 s
VARIANTS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [{}]
# ---- segments from the competition bags ----
segs = []
for b in sorted(glob.glob("/home/shchon11/F1tenth_E2E/real_data/01_competition_0826-0827_map08x23/*")):
    try: d = br.read(b, ["/odom", "/drive", "/sensors/imu/raw", "/sensors/servo_position_command", "/state"])
    except Exception: continue
    if not d.has("/odom", "/drive", "/sensors/imu/raw", "/sensors/servo_position_command"): continue
    t0 = max(d.t["/odom"][0], d.t["/drive"][0]) + 1; t1 = min(d.t["/odom"][-1], d.t["/drive"][-1]) - 2
    T = np.arange(t0, t1, DT)
    v = d.at("/odom", T)[:, 3]; imu = d.at("/sensors/imu/raw", T); r = imu[:, 2]
    ah = np.hypot(imu[:, 3], imu[:, 4]); drv = d.at("/drive", T); sv = d.at("/sensors/servo_position_command", T).ravel()
    pinned = (sv < 0.2355) | (sv > 0.6545)
    bad = np.convolve((ah > 20).astype(float), np.ones(81), "same") > 0          # impacts +-1 s
    for i in range(0, len(T) - SEG - 1, SEG // 2):
        sl = slice(i, i + SEG)
        if v[sl].min() < 1.5 or pinned[sl].any() or bad[sl].any(): continue
        segs.append(dict(v=v[sl], r=r[sl], dcmd=drv[sl, 0], vcmd=drv[sl, 1]))
print("segments", len(segs), flush=True)
S = len(segs)
V = np.stack([s["v"] for s in segs]); R = np.stack([s["r"] for s in segs])
DC = np.stack([s["dcmd"] for s in segs]); VC = np.stack([s["vcmd"] for s in segs])
AY = np.abs(V * R).max(1)
# ---- open floor: 80 m square, walls only at the border ----
res = 0.1; n = 800
occ = np.zeros((n, n), bool); occ[:2] = occ[-2:] = True; occ[:, :2] = occ[:, -2:] = True
edt = ndimage.distance_transform_edt(~occ).astype(np.float32) * res
th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
tr = Track(occupancy=occ, resolution=res, origin=(0.0, 0.0), edt=edt, centerline=np.stack([40 + 30 * np.cos(th), 40 + 30 * np.sin(th)], 1),
           name="open_floor", duct=np.zeros_like(occ), tall=occ.copy())
cfg = Config(); cfg.sim.compile = False; cfg.rand.enabled = False; cfg.sim.terminate_on_collision = False
env = common.make_env([tr], S, "cuda:0", EnvConfig(action_mode="direct", race_size=1), cfg=cfg, seed=1)
env.reset(seed=1); sim = env.sim; dev = sim.state.device
lr = cfg.vehicle.lr
for var in VARIANTS:
    pose = torch.tensor([[40.0, 40.0, 0.0]] * S, device=dev)
    sim.reset(torch.arange(S, device=dev), pose, torch.tensor(V[:, 0], dtype=torch.float32, device=dev))
    for k, val in var.items():
        sim.P[k][:] = val
    r0 = torch.tensor(R[:, 0], dtype=torch.float32, device=dev); d0 = torch.tensor(DC[:, 0], dtype=torch.float32, device=dev)
    sim.state[:, dyn.IR] = r0; sim.state[:, dyn.ISTEER] = d0; sim.state[:, dyn.IVY] = r0 * lr * 0.5
    sim.cmd[:, 0] = d0; sim.cmd_hist[:, :, 0] = d0[:, None]
    rs, vs = [], []
    for j in range(SEG):
        cmd = torch.tensor(np.stack([DC[:, j], VC[:, j]], 1), dtype=torch.float32, device=dev)
        sim.step(cmd)
        rs.append(sim.state[:, dyn.IR].cpu().numpy()); vs.append(sim.state[:, dyn.IVX].cpu().numpy())
    Rs = np.stack(rs, 1); Vs = np.stack(vs, 1)
    a, bsim = R[:, SKIP:], Rs[:, SKIP:]
    rmse = np.sqrt(((a - bsim) ** 2).mean(1)); gain = (a * bsim).sum(1) / np.maximum((a * a).sum(1), 1e-6)
    vr = np.sqrt(((V[:, SKIP:] - Vs[:, SKIP:]) ** 2).mean(1))
    vm = V[:, SKIP:].mean(1); ayi = np.abs(V * R)[:, SKIP:].mean(1)
    gs = [np.median(gain[(vm >= lo) & (vm < hi) & (ayi < 5)]) for lo, hi in ((1.5, 3), (3, 4.5), (4.5, 6), (6, 9.5))]
    print(f"COMPACT {json.dumps(var):52s} rmse {np.median(rmse):.3f} | gain by speed " + " ".join(f"{g:.2f}" for g in gs) + f" | a_y>=8 {np.median(gain[AY >= 8]):.2f}", flush=True)
    if len(VARIANTS) > 4: continue
    print(f"\nvariant {json.dumps(var) or 'default'}")
    print("  real |a_y| max   n    yaw-rate RMSE  sim/real gain(median)  speed RMSE")
    for lo, hi in ((0, 3), (3, 6), (6, 8), (8, 20)):
        m = (AY >= lo) & (AY < hi)
        if m.sum() < 5: continue
        print(f"  {lo:2d}-{hi:2d} m/s^2   {m.sum():4d}   {np.median(rmse[m]):.3f} rad/s      {np.median(gain[m]):.2f}              {np.median(vr[m]):.2f} m/s")
    if not var:
        vm = V[:, SKIP:].mean(1); ayi = np.abs(V * R)[:, SKIP:].mean(1)
        print("  by speed, |a_y| mean < 5 (tyres far from the limit):")
        for lo, hi in ((1.5, 3), (3, 4.5), (4.5, 6), (6, 9.5)):
            m = (vm >= lo) & (vm < hi) & (ayi < 5)
            if m.sum() < 5: continue
            print(f"    v {lo:.1f}-{hi:.1f} m/s  n {m.sum():4d}  sim/real yaw gain {np.median(gain[m]):.2f}")
        # gyro sanity: v*r against the accelerometer's lateral reading is checked separately
