"""The LiDAR student's car-centred grid (2026-10-30) -- the same cells as the privileged teacher's `state_rl.bev_grid`,
built from what the car itself has: its last F scans and its own odometry over those frames. No map and no map pose.

For every cell (2 x 2 samples, as the teacher's) and every stacked frame: move the cell into that frame's car pose with the
odometry (dead reckoning over (F - 1) x stride steps: 0.3 s at 4 frames x 100 ms), into the LiDAR's nominal mount, look
up the beam pointing at it, and mark it
    free  -- the beam went past it (or had no return within range_max),
    hit   -- the beam ended within `margin` of it,
    neither (unknown: behind the hit, outside the field of view or beyond range).
Channels: (free, hit) per frame, newest first -> (B, 2F, BEV_H, BEV_W). Motion-compensated, a wall or a box stays in the
same cells from frame to frame while a moving car smears -- a stack of raw beam arrays mixes the car's own motion with
the other's (10-29 motion probe: 1 frame -> 10 frames raised the opponent closing-speed R2 only 0.81 -> 0.84).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .state_rl import BEV_H, BEV_RES, BEV_W, BEV_X0


def _cells(device, k=2):
    fx = BEV_X0 + (torch.arange(BEV_H * k, device=device) + 0.5) * (BEV_RES / k)
    fy = (torch.arange(BEV_W * k, device=device) + 0.5) * (BEV_RES / k) - BEV_W * BEV_RES / 2
    gx, gy = torch.meshgrid(fx, fy, indexing="ij")
    return gx.reshape(-1), gy.reshape(-1)


def lidar_grid(scan, rel, angles, mount, range_max, margin=0.15):
    """scan (B, F, N) ranges / range_max (1 = no return), newest first; rel (B, F, 3) each frame's car pose (x, y, yaw)
    in the newest frame's car frame; angles (N,) beam angles in the LiDAR frame; mount (x, y, yaw) of the LiDAR in the
    car frame -> (B, 2F, BEV_H, BEV_W) in [0, 1]."""
    B, F, N = scan.shape
    dev = scan.device
    gx, gy = _cells(dev)                                                  # (P,)
    a0 = float(angles[0]); da = float(angles[-1] - angles[0]) / max(N - 1, 1)
    mx, my, myaw = (float(v) for v in mount)
    outs = []
    for f in range(F):
        tx, ty, tyaw = rel[:, f, 0:1], rel[:, f, 1:2], rel[:, f, 2:3]      # (B, 1)
        c, s = torch.cos(tyaw), torch.sin(tyaw)
        dx, dy = gx[None] - tx, gy[None] - ty
        px, py = dx * c + dy * s, -dx * s + dy * c                         # in frame f's car frame
        qx, qy = px - mx, py - my
        cm, sm = math.cos(myaw), math.sin(myaw)
        lx, ly = qx * cm + qy * sm, -qx * sm + qy * cm                     # in its LiDAR frame
        th = torch.atan2(ly, lx); d = torch.hypot(lx, ly)
        b = torch.round((th - a0) / da).long()
        valid = (b >= 0) & (b < N)
        r_n = torch.gather(scan[:, f].float(), 1, b.clamp(0, N - 1))     # (B, P)
        noret = r_n >= 0.999
        r = r_n * float(range_max)
        free = valid & (((~noret) & (d < r - margin)) | (noret & (d < float(range_max))))
        hit = valid & (~noret) & ((d - r).abs() <= margin)
        outs += [free.float(), hit.float()]
    g = torch.stack(outs, 1).reshape(B, 2 * F, BEV_H, 2, BEV_W, 2).mean((3, 5))
    return g


class OdomHist:
    """The car's own odometry poses at every control step (newest first), aligned with the env's scan stack
    (`scan_hist[:, ::stride]`): call `update()` after every env step (and once after reset). A row whose episode just
    began is filled with its current pose, as the env fills its scan history with the current scan."""

    def __init__(self, env, rows, frames, stride):
        self.env, self.rows = env, rows
        self.frames, self.stride = int(frames), int(stride)
        self.L = (self.frames - 1) * self.stride + 1
        self.buf = None

    def update(self):
        od = self.env.sim.odom.state[self.rows, :3].float()
        if self.buf is None:
            self.buf = od[None].repeat(self.L, 1, 1)
        else:
            self.buf = torch.roll(self.buf, 1, 0); self.buf[0] = od
        fresh = self.env.ep_step[self.rows] <= 1
        if bool(fresh.any()):
            self.buf[:, fresh] = od[fresh][None]

    def rel(self):
        """(B, F, 3): each stacked frame's pose in the newest frame's car frame."""
        p = self.buf[:: self.stride][: self.frames]                         # (F, B, 3)
        p0 = p[0]
        dx, dy = p[..., 0] - p0[:, 0], p[..., 1] - p0[:, 1]
        c, s = torch.cos(p0[:, 2]), torch.sin(p0[:, 2])
        out = torch.stack([dx * c + dy * s, -dx * s + dy * c, p[..., 2] - p0[:, 2]], -1)
        return out.permute(1, 0, 2).contiguous()


class StudentGrid(nn.Module):
    """(scan stack, ego, odometry-relative frame poses) -> (steer, speed) in [-1, 1]: the LiDAR grid (`lidar_grid`) read by
    the same small conv as the teacher's `state_rl.BevNet`, joined with the ego vector."""

    def __init__(self, frames, beams, ego_dim, angles, mount, range_max, feat=256):
        super().__init__()
        self.frames, self.beams = int(frames), int(beams)
        self.register_buffer("angles", torch.as_tensor(angles, dtype=torch.float32).clone())
        self.register_buffer("mount", torch.as_tensor(mount, dtype=torch.float32).clone())
        self.range_max = float(range_max)
        C = 2 * self.frames
        self.conv = nn.Sequential(nn.Conv2d(C, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Flatten())
        n = int(self.conv(torch.zeros(1, C, BEV_H, BEV_W)).shape[1])
        self.proj = nn.Sequential(nn.Linear(n, feat), nn.ReLU())
        self.ego = nn.Sequential(nn.Linear(ego_dim, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(feat + 128, 256), nn.ReLU(), nn.Linear(256, 2), nn.Tanh())

    def grid(self, scan, rel):
        return lidar_grid(scan, rel, self.angles, self.mount.tolist(), self.range_max)

    def forward(self, scan, ego, rel):
        return self.head(torch.cat([self.proj(self.conv(self.grid(scan, rel))), self.ego(ego)], 1))


def lidar_mount(env):
    """the LiDAR's NOMINAL mount (x, y, yaw) in the car frame -- what the car knows, not the randomised truth"""
    return (float(env.cfg.lidar.mount_x), float(env.cfg.lidar.mount_y), float(env.cfg.lidar.mount_yaw))


def make_student(sck_or_none, env, frames, beams, ego_dim, grid, device):
    """a state_student checkpoint's network (or a fresh one): the 1-D scan student or the grid student"""
    from .state_student import Student
    g = bool(sck_or_none.get("grid", False)) if sck_or_none is not None else bool(grid)
    if g:
        sub = int(env.ecfg.scan_subsample)
        m = StudentGrid(frames, beams, ego_dim, env.sim.lidar.angles[::sub].float().cpu(), lidar_mount(env),
                        float(env.cfg.lidar.range_max)).to(device)
    else:
        m = Student(frames, beams, ego_dim).to(device)
    if sck_or_none is not None:
        m.load_state_dict(sck_or_none["student"])
    return m


class StudentRunner:
    """Drives a student in an env: keeps the odometry history a grid student needs. Call `act(obs, ego)` once per env step
    BEFORE stepping (it first records the pose the last step left), every step, also on the steps a decision is held."""

    def __init__(self, env, rows, sck, device):
        self.net = make_student(sck, env, int(sck["frames"]), int(sck["beams"]), int(sck["ego_dim"]), None, device)
        self.net.eval()
        self.grid = bool(sck.get("grid", False))
        self.oh = OdomHist(env, rows, int(sck["frames"]), int(sck["args"].get("stride", 3))) if self.grid else None
        self.rows = rows

    def observe(self):
        if self.oh is not None:
            self.oh.update()

    def __call__(self, obs, ego):
        scan = obs["scan"][self.rows].float()
        return self.net(scan, ego, self.oh.rel()) if self.grid else self.net(scan, ego)
