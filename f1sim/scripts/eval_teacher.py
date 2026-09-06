"""Build the raceline for random tracks and drive the teacher (with randomization on)."""
import sys, time, math
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from f1sim import Track, Config, Simulator
from f1sim.raceline import Raceline
from f1sim.teacher import RacelineTeacher

seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3]
B = 16
for seed in seeds:
    tr = Track.generate_random(seed)
    t0 = time.time(); rl = Raceline.build(tr); t_opt = time.time() - t0
    cfg = Config()
    sim = Simulator(tr, cfg, num_envs=B, device="cpu")
    sim.reset(poses=sim.sample_spawn(B, 0.0, 0.0, torch.zeros(B)))
    teacher = RacelineTeacher(rl, wheelbase=cfg.vehicle.lf + cfg.vehicle.lr)
    r = None; traj = []; lap_t = torch.full((B,), float("nan")); vmax = 0.0; lat = []
    max_steps = int(3 * rl.lap_time / sim.control_dt)
    for i in range(max_steps):
        st = r.state if r is not None else sim.state
        r = sim.step(teacher(st, sim.P, sim.tid))
        traj.append(r.state[0, :2].numpy().copy()); vmax = max(vmax, r.state[:, 3].max().item())
        _, le = teacher.project(r.state[:, :2]); lat.append(le)
        done1 = (r.lap >= 1) & torch.isnan(lap_t)
        lap_t[done1] = sim.t
        if (r.lap >= 1).all() or r.collision.all():
            break
    lat = torch.stack(lat)
    print(f"seed {seed}: len {rl.length:.1f} m  raceline opt {t_opt:.1f}s  kinematic lap est {rl.lap_time:.1f}s  "
          f"| sim lap {np.nanmean(lap_t.numpy()):.2f}s (min {np.nanmin(lap_t.numpy()):.2f})  "
          f"collisions {int(r.collision.sum())}/{B}  vmax {vmax:.1f}  lat err mean {lat.mean():.2f} max {lat.max():.2f}")
    traj = np.array(traj)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    H, W = tr.shape
    ext = [tr.origin[0], tr.origin[0] + W * tr.resolution, tr.origin[1], tr.origin[1] + H * tr.resolution]
    ax[0].imshow(~tr.occupancy, cmap="gray", origin="lower", extent=ext)
    ax[0].plot(tr.centerline[:, 0], tr.centerline[:, 1], "c--", lw=0.6, label="centerline")
    sc = ax[0].scatter(rl.xy[:, 0], rl.xy[:, 1], c=rl.v, s=4, cmap="jet", label="raceline (speed)")
    ax[0].plot(traj[:, 0], traj[:, 1], "w-", lw=0.7, alpha=0.8, label="teacher (env 0)")
    plt.colorbar(sc, ax=ax[0], fraction=0.04, label="m/s"); ax[0].legend(); ax[0].set_aspect("equal")
    ax[0].set_title(f"seed {seed}: lap {np.nanmean(lap_t.numpy()):.2f}s, collisions {int(r.collision.sum())}/{B}")
    ax[1].plot(rl.s, rl.v, label="v profile"); ax[1].plot(rl.s, np.abs(rl.kappa) * 10, label="|kappa| x10")
    ax[1].set_xlabel("s [m]"); ax[1].legend()
    fig.savefig(f"/tmp/claude-1000/-home-shchon11-F1tenth/620f3fb5-7a20-477c-957d-12977d5ae04d/scratchpad/teacher_{seed}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
