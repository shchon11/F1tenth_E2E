"""Render a track with a driven trajectory and one LiDAR scan to PNG."""
import math, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from f1sim import Track, Config, Simulator
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/tests")
from test_track_lap import centerline_pursuit

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
out = sys.argv[2] if len(sys.argv) > 2 else f"track_{seed}.png"
tr = Track.generate_random(seed)
cfg = Config()
sim = Simulator(tr, cfg, num_envs=1, device="cpu")
sim.reset(poses=sim.sample_spawn(1, 0.0, 0.0, torch.zeros(1)))
traj, odom = [], []
r = None
for i in range(int(sim.track.length / 3.0 / sim.control_dt)):
    r = sim.step(centerline_pursuit(sim, r.state if r else sim.state, speed=3.0))
    traj.append(r.state[0, :2].numpy().copy()); odom.append(r.odom[0, :2].numpy().copy())
    if r.lap[0] >= 1: break
traj, odom = np.array(traj), np.array(odom)
fig, ax = plt.subplots(figsize=(9, 9))
H, W = tr.shape
ax.imshow(~tr.occupancy, cmap="gray", origin="lower",
          extent=[tr.origin[0], tr.origin[0] + W * tr.resolution, tr.origin[1], tr.origin[1] + H * tr.resolution])
ax.plot(tr.centerline[:, 0], tr.centerline[:, 1], "c--", lw=0.6, label="centerline")
ax.plot(traj[:, 0], traj[:, 1], "r-", lw=1.2, label="ground truth")
ax.plot(odom[:, 0], odom[:, 1], "y-", lw=1.0, label="VESC odom (drift)")
st = r.state[0]; meta = sim.scan_meta()
ang = st[2].item() + cfg.lidar.mount_yaw + np.linspace(meta["angle_min"], meta["angle_max"], cfg.lidar.n_beams)
rng = r.scan[0].numpy(); ok = np.isfinite(rng)
lx = st[0].item() + cfg.lidar.mount_x * math.cos(st[2].item()); ly = st[1].item() + cfg.lidar.mount_x * math.sin(st[2].item())
ax.scatter(lx + rng[ok] * np.cos(ang[ok]), ly + rng[ok] * np.sin(ang[ok]), s=1, c="lime", label="scan")
ax.plot(lx, ly, "mo", ms=4)
ax.set_title(f"{tr.name}: len {sim.track.length:.1f} m, {len(traj)} steps, collided={bool(r.collision[0])}")
ax.legend(loc="upper right"); ax.set_aspect("equal")
fig.savefig(out, dpi=130, bbox_inches="tight"); print("saved", out)
