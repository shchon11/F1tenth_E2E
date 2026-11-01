"""The console's driver for the state_rl teacher and the LiDAR grid student checkpoints (2026-11-01).

These two kinds came after the console's v1 / v2 loader (`learn.model.load_checkpoint`) and are not that format:
- a state_rl teacher (`learn.state_rl`, keys model / args / norm) drives from the TRUE state (`StateObs`: own state,
  the other cars, the props, the track ahead, the corridor and space-time car channel, a wall-and-prop scan) and, with
  `--base`, mixes in its solo teacher where nothing is near (`SoloMix`) -- a privileged oracle, not a car-deployable
  policy;
- a grid student (`learn.state_student`, keys student / frames / beams / ego_keys) drives from what the car has: its
  last scans in a motion-compensated occupancy grid (`learn.lidar_grid`) and its own IMU / speed / last action.
Both command (steer, speed) directly at the teacher's decision rate (every `repeat` control steps), speed with the
reverse extension. The session's env is built to their contract (`env_overrides`), and `StatePolicyRunner` stands
where the console's actor runner stands: called once per control step, it returns every car's action.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch


def state_kind(path: str) -> str:
    """'teacher' for a state_rl checkpoint, 'student' for a state_student one, '' for anything else."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return ""
    if not isinstance(ck, dict):
        return ""
    if "student" in ck and "ego_keys" in ck:
        return "student"
    if "model" in ck and "norm" in ck and isinstance(ck.get("args"), dict):
        return "teacher"
    return ""


def load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def contract(ck: dict, kind: str) -> dict:
    """What the env must be for this checkpoint: beams, scan frames / stride, reverse extension, decision repeat."""
    a = dict(ck.get("args") or {})
    if kind == "teacher":
        return dict(beams=int(a.get("beams", 36)), frames=1, stride=1,
                    reverse_ext=float(a.get("reverse_ext", 0.0) or 0.0), repeat=int(a.get("repeat", 2) or 2))
    # a student is driven at its teacher's rate; its run records the teacher, whose args carry the repeat
    rep = 2
    t = a.get("teacher")
    if t and os.path.isfile(t):
        try:
            rep = int((torch.load(t, map_location="cpu", weights_only=False, mmap=True).get("args") or {}).get("repeat", 2) or 2)
        except Exception:
            rep = 2
    return dict(beams=int(ck["beams"]), frames=int(ck["frames"]), stride=int(a.get("stride", 3)),
                reverse_ext=float(a.get("reverse_ext", 0.0) or 0.0), repeat=rep)


def env_overrides(ck: dict, kind: str) -> dict:
    """EnvConfig fields for this checkpoint's contract (as `state_student.make_env_like` sets them): a direct command
    over [0, 10] m/s from the v1 observation (students) or the privileged state (teachers)."""
    c = contract(ck, kind)
    kw = dict(action_mode="direct", obs_version=1, scan_stack=c["frames"], scan_stride=c["stride"], v_max_policy=10.0,
              reverse_ext=c["reverse_ext"])
    if kind == "teacher":
        kw.update(priv_props=True, priv_time=True)
    return kw


def describe(ck: dict, kind: str, path: str) -> str:
    """one line for the console's header"""
    run = os.path.basename(os.path.dirname(os.path.realpath(path)))
    name = os.path.basename(os.path.realpath(path))
    if kind == "teacher":
        base = " + 혼자 선생 혼합" if (ck.get("args") or {}).get("base") else ""
        return f"참 상태 선생(오라클){base}: {run}/{name} · 직접 행동 20 Hz"
    return f"라이다 격자 학생: {run}/{name} · 반복 {ck.get('iter', '?')} · 스캔 {ck['frames']}장 · 직접 행동 20 Hz"


def _to(x, device):
    """every tensor in a checkpoint's nested dicts / lists on `device`"""
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return type(x)((k, _to(v, device)) for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return type(x)(_to(v, device) for v in x)
    return x


class StatePolicyRunner:
    """(scan, ego) -> (B, 2) every car's action, in the console actor runner's place. The arguments are the console's
    flattened v1 observation and are not used: a teacher builds its own observation from the env, a student reads the
    session's observation dict (`self.session["obs"]`, set by the worker) and keeps its odometry history itself. Called
    once per control step; a new decision every `repeat` calls, held in between, as in training."""

    def __init__(self, env, ck: dict, kind: str, device):
        from ..learn import state_rl
        ck = _to(ck, device)                          # the input norm's buffers load as they were saved
        self.env, self.kind, self.device = env, kind, device
        self.lid = env.learner_ids
        self.repeat = max(1, int(contract(ck, kind)["repeat"]))
        self.k = 0
        self.session: Optional[dict] = None
        B = int(self.lid.numel())
        self.last = torch.zeros(B, 2, device=device)
        if kind == "teacher":
            self.tp = state_rl.TeacherPolicy(env, ck, device)
        else:
            from ..learn.lidar_grid import StudentRunner
            self.sr = StudentRunner(env, self.lid, ck, device)
            self.keys = list(ck.get("ego_keys") or [])

    @property
    def gate(self):
        """the teacher's mixture gate per learner (True = the race policy drives), None for a student"""
        mix = getattr(getattr(self, "tp", None), "mix", None)
        return None if mix is None else mix.gate

    @torch.no_grad()
    def __call__(self, scan=None, ego=None):
        env = self.env
        if self.kind == "student":
            self.sr.observe()                         # the pose the last step left, every step
        if self.k % self.repeat == 0:
            if self.kind == "teacher":
                a = self.tp(self.last)
            else:
                obs = self.session["obs"] if self.session is not None else env._last_obs
                B = int(self.lid.numel())
                eg = torch.cat([obs[k][self.lid].float().reshape(B, -1) for k in self.keys], 1)
                a = self.sr(obs, eg)
            self.last = a.float()
        self.k += 1
        out = torch.zeros(env.B, env.act_dim, device=self.device)
        out[self.lid] = self.last.to(out.dtype)
        return out


class StateIntro:
    """The console's activations view (`ActivationPanel`: "hidden" 256 and "stem" 256) for these networks, by forward
    hooks: the teacher's two actor layers (after each tanh), the student's grid features after the projection ("stem")
    and its head's hidden layer. `paused` while the overlay drives the network on imagined inputs, so the panel shows
    the drive's own forward. No saliency over the scan here -- the overlay's heat map is this view's saliency."""

    def __init__(self, runner: "StatePolicyRunner"):
        self.h: dict = {}
        self.paused = False
        if runner.kind == "teacher":
            pi = runner.tp.net.pi
            if isinstance(pi, torch.nn.Sequential) and len(pi) >= 4:
                pi[1].register_forward_hook(self._hook("stem"))
                pi[3].register_forward_hook(self._hook("hidden"))
        else:
            net = runner.sr.net
            if hasattr(net, "proj") and hasattr(net, "head"):
                net.proj.register_forward_hook(self._hook("stem"))
                net.head[1].register_forward_hook(self._hook("hidden"))

    def _hook(self, key):
        def f(_m, _i, out):
            if not self.paused:
                self.h[key] = out.detach()
        return f

    def saliency(self, *a, **k):
        raise NotImplementedError("no scan saliency for a state teacher / grid student")


# ------------------------------------------------------------------------------------------- the console's picture
# Colours (RGBA). Hue on the trajectory ribbon is speed, as for every driver; these stay off it.
PROP_CELL = (1.00, 0.55, 0.15, 0.95)      # a corridor cell a prop blocks (the teacher's prop channel)
CAR_CELL = (0.95, 0.30, 0.95, 0.95)       # a cell the other car will be in when we get there (space-time channel)
FREE_CELL = (0.70, 0.75, 0.82, 0.35)      # a corridor cell left free
AHEAD_PT = (0.95, 0.95, 0.98, 0.85)       # the track points ahead the teacher steers by
GRID_HIT = (1.00, 0.45, 0.20, 0.90)       # a grid cell where a beam ended (occupied)
HOT = (1.00, 0.95, 0.35, 1.00)            # the cells the action is most sensitive to
#: the student's imagined drive: decisions rolled out, [s] per decision is the env's (repeat x control_dt)
IMAGINE_STEPS = 30
#: the overlay's heavy part (a backward pass, the imagined drive) is refreshed this often [s]; frames in between reuse it
REFRESH_S = 0.2
#: hot cells drawn in 3D (the panel draws the whole heat map)
HOT_CELLS = 24


def _heat_rgba(x):
    """[0, 1] -> an inferno-like ramp (black-purple-orange-yellow) as (..., 3) floats"""
    import numpy as np
    stops = np.array([[0.00, 0.00, 0.02], [0.34, 0.06, 0.43], [0.80, 0.26, 0.24], [0.99, 0.65, 0.04], [0.99, 0.99, 0.64]],
                     np.float32)
    t = np.clip(np.asarray(x, np.float32), 0.0, 1.0) * (len(stops) - 1)
    k = np.minimum(t.astype(np.int32), len(stops) - 2)
    f = (t - k)[..., None]
    return stops[k] * (1.0 - f) + stops[k + 1] * f


class StateViz:
    """The overlay for a car the state teacher or the grid student drives, in `driver_viz.DriverViz`'s interface:
    `driver(row)`, `tensors(row)` (what the snapshot copies to the host), `decode(row, flat, ...)` -> (the trajectory
    ribbon in world [x, y, v], the overlay dict the 3D view and the input panel draw).

    Student -- what it reads and how it reacts:
      * its input, the newest frame of the motion-compensated LiDAR occupancy grid (`lidar_grid`): free cells dark green,
        occupied cells orange, in the car frame, as an image under the panel's scan;
      * where it looks: the cells its (steer, speed) output is most sensitive to, gradient x input summed over the stacked
        frames (an inferno heat map on the grid; the hottest cells as markers in 3D);
      * where it means to go: the policy driven forward `IMAGINE_STEPS` decisions on a kinematic bicycle with the grid
        re-read from each imagined pose -- the scans held still, so the world is taken as frozen (a car ahead is a
        wall here). Not a plan the network outputs: it commands (steer, speed) only; this is that command, repeated.
    Teacher -- its two corridor channels on the 13 x 7 cells along the track it reads them on (props orange, the other
      car where it will be when we get there magenta, free grey), outlined by how strongly its action reacts to each cell
      (gradient magnitude), the track points it steers by, its hit head's P(a hard hit within 1 s), which policy drives,
      and its command rolled out for one second."""

    def __init__(self, runner: StatePolicyRunner):
        from ..learn import state_rl
        self.r = runner
        self.env = runner.env
        v = self.env.cfg.vehicle
        self.wheelbase = float(v.lf + v.lr)
        dev = runner.device
        self.ahead = torch.tensor(state_rl.AHEAD, device=dev)
        self.n_seg, self.n_bin = len(state_rl.AHEAD), state_rl.CORRIDOR_BINS
        # the cells' edges: along the track from 0.2 m (the first dense row is 0.4) to each AHEAD point, across it at
        # every second lateral sample, half a sample spacing outside the outermost
        self.s_edges = torch.cat([torch.tensor([0.2], device=dev), self.ahead])                       # (14,)
        U = torch.tensor(state_rl.CORRIDOR_U, device=dev)
        du = float(U[1] - U[0])
        self.u_edges = torch.cat([U[0::2] - du / 2, (U[-1] + du / 2).reshape(1)])                      # (8,)
        self.bev = (state_rl.BEV_X0, state_rl.BEV_RES, state_rl.BEV_H, state_rl.BEV_W)
        self.dt = float(self.env.sim.control_dt) * float(runner.repeat)
        self._cache, self._t = None, -1e9

    def _idx(self, row: int):
        hit = (self.r.lid == int(row)).nonzero()
        return None if hit.numel() == 0 else int(hit[0, 0])

    def driver(self, row: int):
        if bool(self.env.external_driven[int(row)]):
            return None
        return self.r if self._idx(row) is not None else None

    def tensors(self, row: int) -> list:
        import time
        i = self._idx(row)
        if i is None:
            return []
        now = time.monotonic()
        if self._cache is not None and self._cache[0] == row and now - self._t < REFRESH_S:
            return [t.clone() for t in self._cache[1]]
        intro = getattr(self.r, "intro", None)
        if intro is not None:
            intro.paused = True
        try:
            out = self._teacher(row, i) if self.r.kind == "teacher" else self._student(row, i)
        finally:
            if intro is not None:
                intro.paused = False
        self._cache, self._t = (row, out), now
        return [t.clone() for t in out]

    # ------------------------------------------------------------------ teacher
    def _teacher(self, row, i):
        env, dev, tp = self.env, self.r.device, self.r.tp
        nc = self.n_seg * self.n_bin
        so = tp.sobs
        o = tp.o[i:i + 1].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            y = tp.net.pi(tp.nrm(o))                                   # (1, 2) the race policy's mean, pre-clamp
            g0, = torch.autograd.grad(y[0, 0], o, retain_graph=True)
            g1, = torch.autograd.grad(y[0, 1], o)
        sens = (g0.abs() / (g0.abs()[0, -2 * nc:].max() + 1e-9) + g1.abs() / (g1.abs()[0, -2 * nc:].max() + 1e-9))[0] * 0.5
        o = o.detach()
        car = o[0, -nc:] if so.car_channel else torch.zeros(nc, device=dev)
        k0 = -2 * nc if so.car_channel else -nc
        prop = o[0, k0:k0 + nc] if so.corridor else torch.zeros(nc, device=dev)
        s_car = sens[-nc:] if so.car_channel else torch.zeros(nc, device=dev)
        s_prop = sens[k0:k0 + nc] if so.corridor else torch.zeros(nc, device=dev)
        res = env.last_result
        s, tid = res.s[row:row + 1], env.sim.tid[row:row + 1]
        tr = env.sim.track
        ne = self.s_edges.numel()
        xy, yaw = tr.pose_at_s((s + self.s_edges).reshape(-1), tid.repeat(ne))                       # (14, 2), (14,)
        w = tr.sample_edt(xy[None], tid[:, None].expand(-1, ne))[0]                                  # (14,)
        lat = self.u_edges[None, :] * w[:, None]                                                     # (14, 8) left +
        px = xy[:, 0:1] - lat * torch.sin(yaw)[:, None]
        py = xy[:, 1:2] + lat * torch.cos(yaw)[:, None]
        axy, _ = tr.pose_at_s((s + self.ahead).reshape(-1), tid.repeat(self.n_seg))
        with torch.no_grad():
            hp = torch.sigmoid(tp.net.value_and_hit(tp.nrm(o))[1]).reshape(1)
        gate = tp.mix.gate[i:i + 1].float() if tp.mix is not None else torch.ones(1, device=dev)
        return [prop.float(), car.float(), torch.maximum(s_prop, s_car).float(),
                torch.stack([px, py], -1).reshape(-1).float(), axy.reshape(-1).float(), hp.float(), gate]

    # ------------------------------------------------------------------ student
    def _student(self, row, i):
        env, dev, sr = self.env, self.r.device, self.r.sr
        net = sr.net
        obs = self.r.session["obs"] if self.r.session is not None else env._last_obs
        keys = self.r.keys
        scan = obs["scan"][row:row + 1].float()
        ego0 = torch.cat([obs[k][row:row + 1].float().reshape(1, -1) for k in keys], 1)
        rel0 = sr.oh.rel()[i:i + 1]
        g = net.grid(scan, rel0).detach().requires_grad_(True)                                      # (1, 2F, H, W)
        with torch.enable_grad():
            y = net.head(torch.cat([net.proj(net.conv(g)), net.ego(ego0)], 1))
            g0, = torch.autograd.grad(y[0, 0], g, retain_graph=True)
            g1, = torch.autograd.grad(y[0, 1], g)
        g = g.detach()
        a0 = (g0 * g).abs().sum(1)[0]; a1 = (g1 * g).abs().sum(1)[0]
        heat = 0.5 * (a0 / (a0.max() + 1e-9) + a1 / (a1.max() + 1e-9))                            # (H, W)
        traj = self._imagine(row, scan, ego0, rel0, keys)
        return [g[0, 0].reshape(-1), g[0, 1].reshape(-1), heat.reshape(-1).float(), traj.reshape(-1).float()]

    @torch.no_grad()
    def _imagine(self, row, scan, ego0, rel0, keys):
        """(K, 3) [x, y, v] in the car's frame now: the student driven forward on a kinematic bicycle, re-reading its grid
        from each imagined pose (the scans held still), its last actions and its speed fed back into its ego vector."""
        env, net = self.env, self.r.sr.net
        dev = scan.device
        # where the speed and the action history sit in the ego vector (`StateObs`-free: the v1 observation's keys)
        sizes = {k: int(self.r.session["obs"][k].shape[1]) if self.r.session is not None else 0 for k in keys}
        off, pos = 0, {}
        for k in keys:
            pos[k] = (off, off + sizes[k]); off += sizes[k]
        vmax = float(env.ecfg.v_max_policy)
        ext = float(getattr(env.ecfg, "reverse_ext", 0.0) or 0.0)
        rev = float(getattr(env.ecfg, "reverse_speed", 2.0) or 2.0)
        smax = float(env.s_max)
        L = self.wheelbase
        x = y = yaw = 0.0
        v = float(env.sim.state[row, 3]); d = float(env.sim.state[row, 6])
        ego = ego0.clone()
        hist = ego[0, pos["prev_action"][0]:pos["prev_action"][1]].clone() if "prev_action" in pos else None
        out = []
        dt = self.dt
        for _ in range(IMAGINE_STEPS):
            c, s_ = math.cos(yaw), math.sin(yaw)
            dx, dy = rel0[..., 0] - x, rel0[..., 1] - y
            rel = torch.stack([dx * c + dy * s_, -dx * s_ + dy * c, rel0[..., 2] - yaw], -1)
            if "speed" in pos:
                ego[0, pos["speed"][0]] = v / vmax
            if hist is not None:
                ego[0, pos["prev_action"][0]:pos["prev_action"][1]] = hist
            a = net(scan, ego, rel)[0]
            st_n, sp_n = float(a[0].clamp(-1, 1)), float(a[1])
            v_cmd = (sp_n + 1.0) * 0.5 * vmax if sp_n >= -1.0 else -rev * (-(sp_n + 1.0) / max(ext, 1e-6))
            d_cmd = st_n * smax
            if hist is not None:
                hist = torch.cat([a.reshape(-1), hist[:-2]]) if hist.numel() > 2 else a.reshape(-1)
            for _sub in range(5):                                     # 5 substeps per decision
                h = dt / 5
                v += max(-8.0 * h, min(5.0 * h, v_cmd - v))
                d += max(-4.0 * h, min(4.0 * h, d_cmd - d))
                yaw += v * math.tan(d) / L * h
                x += v * math.cos(yaw) * h; y += v * math.sin(yaw) * h
            out.append((x, y, v))
        return torch.tensor(out, device=dev, dtype=torch.float32)

    # ------------------------------------------------------------------ host side
    def decode(self, row: int, flat, *, pose, mount, scan_m, cmd):
        import numpy as np
        from .driver_viz import command_arc
        flat = np.asarray(flat, np.float32)
        steer, v = float(cmd[0]), float(cmd[1])
        px, py, pyaw = (float(t) for t in pose)
        cs, sn = math.cos(pyaw), math.sin(pyaw)
        lines, points = [], []
        if self.r.kind == "teacher":
            nc = self.n_seg * self.n_bin
            i = 0
            prop = flat[i:i + nc] > 0.5; i += nc
            car = flat[i:i + nc] > 0.5; i += nc
            sens = flat[i:i + nc]; i += nc
            ne, nu = self.n_seg + 1, self.n_bin + 1
            P = flat[i:i + ne * nu * 2].reshape(ne, nu, 2); i += ne * nu * 2
            axy = flat[i:i + 2 * self.n_seg].reshape(-1, 2); i += 2 * self.n_seg
            hp, gate = float(flat[i]), float(flat[i + 1])
            # each cell's four corners (k = along, j = across), world frame
            q = np.stack([P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]], 2).reshape(nc, 4, 2)
            base = np.where(car[:, None], np.array(CAR_CELL, np.float32)[None],
                            np.where(prop[:, None], np.array(PROP_CELL, np.float32)[None],
                                     np.array([0.55, 0.60, 0.70, 0.10], np.float32)[None]))
            fill = base.copy()
            fill[:, 3] = np.where(car | prop, 0.55, 0.06)
            sn_ = np.clip(sens / max(1e-6, float(sens.max())), 0, 1) ** 1.5      # the hottest cell = 1
            edge = np.concatenate([_heat_rgba(0.45 + 0.55 * sn_), sn_[:, None]], 1)
            quads = {"xy": q, "fill": fill, "edge": edge}
            cen = q.mean(1)
            hot = np.argsort(-sens)[:8]
            points += [{"xy": cen[prop], "rgba": PROP_CELL, "px": 8.0, "panel": False},
                       {"xy": cen[car], "rgba": CAR_CELL, "px": 9.0, "panel": False},
                       {"xy": cen[hot], "rgba": HOT, "px": 5.0, "panel": False},
                       {"xy": axy, "rgba": AHEAD_PT, "px": 4.0}]
            who = "경주 정책이 몲" if gate > 0.5 else "혼자 선생이 몲(근처에 차·상자 없음)"
            legend = (f"참 상태 선생 · {who} · 충돌 예측(1 s 안) {hp:.2f} · 칸 = 트랙을 따라 13 x 7 통로: 주황 = 상자가 막음, 분홍 = 내가 "
                      f"닿을 때 상대가 있을 곳 · 칸 테두리 밝기 = 행동이 그 칸에 얼마나 민감한가 · 흰 점 = 따라가는 앞 트랙 점 · "
                      f"색 띠 = 명령(조향 {np.degrees(steer):+.0f}도, {v:.1f} m/s)을 1초 굴린 궤적")
            title = "참 상태 선생"
            body = {"quads": quads}
            traj = command_arc(pose, steer, v, self.wheelbase)
        else:
            x0, res, H, W = self.bev
            n = H * W
            free = flat[:n].reshape(H, W); hit = flat[n:2 * n].reshape(H, W)
            heat = flat[2 * n:3 * n].reshape(H, W)
            tr_ = flat[3 * n:].reshape(-1, 3)
            occ = hit > 0.25
            fr_ = (free > 0.5) & ~occ
            # the panel's image: free / occupied as the network reads them, the heat map over both
            img = np.zeros((H, W, 4), np.float32)
            img[fr_] = (0.07, 0.26, 0.22, 0.75)
            img[occ] = (0.95, 0.95, 0.98, 0.95)
            ht = np.clip(heat / max(1e-6, float(np.quantile(heat, 0.995))), 0.0, 1.0) ** 1.3
            hc = _heat_rgba(0.3 + 0.7 * ht)
            ht = np.clip((ht - 0.25) / 0.75, 0.0, 1.0)                  # the faint tail stays off the free cells
            al = (0.9 * ht)[..., None]
            img[..., :3] = img[..., :3] * (1 - al) + hc * al
            img[..., 3] = np.maximum(img[..., 3], 0.9 * ht)
            ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            bx = x0 + (ii + 0.5) * res; by = (jj + 0.5) * res - W * res / 2
            wx = px + bx * cs - by * sn; wy = py + bx * sn + by * cs
            hot = np.argsort(-heat.reshape(-1))[:HOT_CELLS]
            points += [{"xy": np.stack([wx[occ], wy[occ]], 1), "rgba": GRID_HIT, "px": 4.0, "panel": False},
                       {"xy": np.stack([wx.reshape(-1)[hot], wy.reshape(-1)[hot]], 1), "rgba": HOT, "px": 7.0, "panel": False}]
            # the imagined drive, world frame, as the speed ribbon (the 3D view and the panel draw `plan` coloured by speed)
            traj = np.stack([px + tr_[:, 0] * cs - tr_[:, 1] * sn, py + tr_[:, 0] * sn + tr_[:, 1] * cs, tr_[:, 2]], 1)
            traj = np.concatenate([np.array([[px, py, tr_[0, 2] if len(tr_) else 0.0]], np.float32), traj], 0).astype(np.float32)
            legend = (f"라이다 격자 학생 · 입력 = 최근 스캔 4장(100 ms)을 자기 오도메트리로 맞춘 점유 격자({H * res:.0f} x {W * res:.0f} m, "
                      f"{res:.2f} m 칸) 최신 장: 짙은 청록 = 빈 곳, 흰색 = 점유 · 열지도 = 조향·속도가 그 칸에 얼마나 반응하나(기울기 x 입력) · "
                      f"색 띠 = 정책을 {IMAGINE_STEPS * self.dt:.1f} s 굴린 상상 궤적(세상이 멈췄다고 보고 격자를 다시 읽음) · "
                      f"지금 명령 조향 {np.degrees(steer):+.0f}도, {v:.1f} m/s")
            title = "라이다 격자 학생"
            body = {"grid": {"x0": x0, "res": res, "H": H, "W": W, "rgba": (np.clip(img, 0, 1) * 255).astype(np.uint8)}}
        return traj, dict({"kind": self.r.kind, "title": title, "legend": legend, "lines": lines, "points": points}, **body)
