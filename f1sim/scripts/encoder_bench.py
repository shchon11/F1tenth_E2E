"""Which LiDAR encoder can tell where the other car is and how it moves? Supervised, offline, held-out tracks.

The policy's own encoder carries the nearest opponent's position at R^2 0.07-0.24 and its relative velocity at
-0.06-0.19 (linear probe, docs/research/motion-memory-2026-09-14.md) -- but it was never asked to. This asks
directly: the same targets, the same data, trained with supervision, for three encoders:

  1d      the policy's resnet ScanStem on the raw 6-frame stack (what the car runs now)
  bev     past scans warped into the CURRENT ego frame (wheel speed + yaw rate, `learn.aligned`) and rasterised to
          an egocentric grid (0.1 m, x -2..10 m, y -4..4 m), one channel per frame, small 2D CNN
  1d+bev  both, concatenated
  bevmem  a persistent egocentric BEV memory (the user's idea, 2026-09-30): every 2 steps the memory is warped into
          the new ego frame (grid_sample on the same wheel-speed + yaw-rate motion) and a ConvGRU folds the new scan's
          raster into it -- mapping while driving, so what is static accumulates and what moves stands out. 0.6 s
          window (8 updates, every 3 steps, 8 channels, bf16: what fits beside two trainings) from an empty memory; the encoder reads memory + the current raster.

Targets: `F1VecEnv.future_labels` of this instant -- the nearest opponent's (lon, lat) position and (vlon, vlat)
relative velocity in the ego frame and its presence. R^2 and MAE on present rows of held-out tracks.

    python scripts/encoder_bench.py collect OUT.pt [--steps 1500]
    python scripts/encoder_bench.py train OUT.pt [--epochs 8]
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN = ["real:icra2022", "real:icra2022~rev", "real:blackbox2021_1", "real:blackbox2022_1~mir", "real:blackbox2021_3",
         "gen:competition:1001", "gen:control:1401", "gen:circuit:1201", "gen:hallway:1101", "gen:competition:1005~rev"]
TEST = ["real:korea_2025_iccas", "real:blackbox2022_3", "real:map12x16", "gen:competition:0"]
SLOTS = '[{"kind_mix":["forzaeth","forzaeth_pred","lane_switch","interactive"],"speed_scale":[0.55,0.95]},' \
        '{"kind_mix":["forzaeth","forzaeth_pred","lane_switch","interactive"],"speed_scale":[0.55,0.95]}]'
FRAMES = (0, 2, 4, 6)                    # BEV: steps back (0, 50, 100, 150 ms at 40 Hz)
GRID = dict(x0=-2.0, x1=10.0, y0=-4.0, y1=4.0, cell=0.1)
STACK = 6
MEM_WINDOW, MEM_EVERY, MEM_CH = 24, 3, 8


def collect(names, steps, envs, seed, device):
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    from f1sim.opponent_slots import parse_slots
    trs, rls = common.load_tracks(names, racelines=True, objective="min_curvature", a_lat=7.0, a_acc=6.5, a_brake=4.0)
    ecfg = EnvConfig(action_mode="plan", race_size=3, opponent="slots", opponent_slots=parse_slots(SLOTS),
                     collision_mode="soft", speed_cap=9.0, scan_stack=STACK, hist_len=20, spawn_runway=3.0,
                     compile_tracker=False)
    from f1sim.params import Config
    cfg = Config(); cfg.sim.compile = False                          # eager: compiling beside two trainings takes longer than collecting
    env = common.make_env(trs, envs * 3, device, ecfg, cfg=cfg, seed=seed, rls=rls,
                          teacher_limits=dict(a_lat=7.0, a_acc=6.5, a_brake=4.0))
    teacher = common.make_teacher(rls, env, a_lat=7.0, a_acc=6.5, a_brake=4.0)
    obs, _ = env.reset(seed=seed)
    lid = env.learner_ids
    rec = {k: [] for k in ("scan", "v", "w", "lab", "car", "done")}
    t0 = time.time()
    for t in range(steps):
        r = env.last_result
        st = r.state[lid]
        rec["scan"].append(obs["scan"][lid, 0].half().cpu())          # newest frame, normalised range
        rec["v"].append(st[:, 3].float().cpu()); rec["w"].append(st[:, 5].float().cpu())
        rec["lab"].append(env.future_labels()[lid].float().cpu())
        rec["car"].append(env.opponent_beam_mask()[lid].bool().cpu())
        with torch.no_grad():
            act = env.teacher_label(teacher)
        obs, _r, term, trunc, _i = env.step(act)
        rec["done"].append((term | trunc)[lid].cpu())
        if t % 250 == 0:
            print(f"  {t}/{steps} steps, {time.time() - t0:.0f} s", flush=True)
    out = {k: torch.stack(v) for k, v in rec.items()}               # (T, L, ...)
    out["dt"] = float(env.sim.control_dt); out["n_beams"] = int(env.n_beams); out["fov"] = float(env.cfg.lidar.fov)
    out["range_max"] = float(getattr(env.cfg.lidar, "range_max", 10.0))
    return out


# ------------------------------------------------------------------ inputs
def windows(d, need):
    """(t, l) index pairs whose last `need` steps are one episode."""
    T, L = d["done"].shape
    ok = torch.ones(T, L, dtype=torch.bool)
    ok[:need] = False
    dn = d["done"].float()
    for k in range(1, need + 1):                                  # a done at t-k means t-k+1.. is a new episode
        ok[k:] &= dn[:-k] == 0
    t, l = torch.nonzero(ok, as_tuple=True)
    return t, l


def stack_1d(d, t, l):
    idx = t[:, None] - torch.arange(STACK)[None]                    # newest first, as the env stacks
    return d["scan"][idx, l[:, None]].float()                      # (B, 6, N)


def bev(d, t, l, angles, device):
    from f1sim.learn.aligned import step_increment, compose_stack
    g = GRID; W = int(round((g["x1"] - g["x0"]) / g["cell"])); H = int(round((g["y1"] - g["y0"]) / g["cell"]))
    B = t.shape[0]; N = angles.shape[0]
    out = torch.zeros(B, len(FRAMES), W, H, device=device)
    ca, sa = torch.cos(angles), torch.sin(angles)
    for c, k in enumerate(FRAMES):
        r = d["scan"][t - k, l].float().to(device) * d["range_max"]            # (B, N) metres
        hit = r < d["range_max"] * 0.999
        px, py = r * ca, r * sa
        if k > 0:
            j = t[:, None] - torch.arange(k, 0, -1)[None]                     # steps t-k .. t-1, oldest first
            v0 = d["v"][j, l[:, None]].to(device); v1 = d["v"][j + 1, l[:, None]].to(device)
            w0 = d["w"][j, l[:, None]].to(device); w1 = d["w"][j + 1, l[:, None]].to(device)
            p, dy = step_increment(v0.T, w0.T, v1.T, w1.T, d["dt"])         # (k, B, 2), (k, B)
            A, b = compose_stack(p, dy)
            px, py = (A[:, 0, 0, None] * px + A[:, 0, 1, None] * py + b[:, 0, None],
                      A[:, 1, 0, None] * px + A[:, 1, 1, None] * py + b[:, 1, None])
        ix = ((px - g["x0"]) / g["cell"]).floor().long(); iy = ((py - g["y0"]) / g["cell"]).floor().long()
        keep = hit & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        bi = torch.arange(B, device=device)[:, None].expand(B, N)
        out[bi[keep], c, ix[keep], iy[keep]] = 1.0
    return out


def _grid_dims():
    g = GRID
    return int(round((g["x1"] - g["x0"]) / g["cell"])), int(round((g["y1"] - g["y0"]) / g["cell"]))


def raster(r_m, hit, px, py, device):
    g = GRID; W, H = _grid_dims(); B, N = px.shape
    out = torch.zeros(B, 1, W, H, device=device)
    ix = ((px - g["x0"]) / g["cell"]).floor().long(); iy = ((py - g["y0"]) / g["cell"]).floor().long()
    keep = hit & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    bi = torch.arange(B, device=device)[:, None].expand(B, N)
    out[bi[keep], 0, ix[keep], iy[keep]] = 1.0
    return out


def warp_grid(A, b, device):
    """grid_sample grid taking a map in the OLD ego frame to the NEW one, p_new = A p_old + b."""
    g = GRID; W, H = _grid_dims()
    xs = g["x0"] + (torch.arange(W, device=device) + 0.5) * g["cell"]
    ys = g["y0"] + (torch.arange(H, device=device) + 0.5) * g["cell"]
    X, Y = torch.meshgrid(xs, ys, indexing="ij")                       # (W, H) cell centres, new frame
    dx, dy = X[None] - b[:, 0, None, None], Y[None] - b[:, 1, None, None]
    ox = A[:, 0, 0, None, None] * dx + A[:, 1, 0, None, None] * dy      # A^T (p - b)
    oy = A[:, 0, 1, None, None] * dx + A[:, 1, 1, None, None] * dy
    nx = (ox - g["x0"]) / (g["x1"] - g["x0"]) * 2 - 1                    # tensor dim 2 (x) -> grid[..., 1]
    ny = (oy - g["y0"]) / (g["y1"] - g["y0"]) * 2 - 1                    # tensor dim 3 (y) -> grid[..., 0]
    return torch.stack([ny, nx], -1)


class ConvGRU(nn.Module):
    def __init__(self, cin, ch):
        super().__init__()
        self.zr = nn.Conv2d(cin + ch, 2 * ch, 3, padding=1)
        self.h = nn.Conv2d(cin + ch, ch, 3, padding=1)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.zr(torch.cat([x, h], 1))).chunk(2, 1)
        return (1 - z) * h + z * torch.tanh(self.h(torch.cat([x, r * h], 1)))


def mem_inputs(d, t, l, angles, device):
    """The per-update rasters and the frame-to-frame warps of the memory window, oldest first."""
    from f1sim.learn.aligned import step_increment, compose_stack
    ca, sa = torch.cos(angles), torch.sin(angles)
    taus = list(range(MEM_WINDOW, -1, -MEM_EVERY))                          # steps back, oldest first
    rasters, warps = [], []
    for i, k in enumerate(taus):
        r = d["scan"][t - k, l].float().to(device) * d["range_max"]
        rasters.append(raster(r, r < d["range_max"] * 0.999, r * ca, r * sa, device))
        if i > 0:
            kp = taus[i - 1]                                                  # previous update, kp steps back
            j = t[:, None] - torch.arange(kp, k, -1)[None]                    # steps t-kp .. t-k-1
            v0 = d["v"][j, l[:, None]].to(device); v1 = d["v"][j + 1, l[:, None]].to(device)
            w0 = d["w"][j, l[:, None]].to(device); w1 = d["w"][j + 1, l[:, None]].to(device)
            p, dy = step_increment(v0.T, w0.T, v1.T, w1.T, d["dt"])
            A, b = compose_stack(p, dy)
            warps.append(warp_grid(A, b, device))
    return rasters, warps


class MemNet(nn.Module):
    def __init__(self, out=256):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(1, MEM_CH, 3, padding=1), nn.GELU())
        self.gru = ConvGRU(MEM_CH, MEM_CH)
        self.read = BEVNet(MEM_CH + 1, out)

    def forward(self, rasters, warps):
        h = torch.zeros(rasters[0].shape[0], MEM_CH, *rasters[0].shape[2:], device=rasters[0].device)
        for i, x in enumerate(rasters):
            if i > 0:
                h = F.grid_sample(h, warps[i - 1], mode="bilinear", padding_mode="zeros", align_corners=False)
            h = self.gru(self.inp(x), h)
        return self.read(torch.cat([h, rasters[-1]], 1))


# ------------------------------------------------------------------ encoders
class BEVNet(nn.Module):
    def __init__(self, cin, out=256):
        super().__init__()
        ch = [cin, 32, 64, 96, 128]
        layers = []
        for a, b_ in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(a, b_, 3, stride=2, padding=1), nn.GroupNorm(8, b_), nn.GELU(),
                       nn.Conv2d(b_, b_, 3, padding=1), nn.GroupNorm(8, b_), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.fc = nn.LazyLinear(out)

    def forward(self, x):
        return F.gelu(self.fc(self.net(x).flatten(1)))


class Model(nn.Module):
    def __init__(self, arm, n_beams):
        super().__init__()
        from f1sim.learn.model import ScanStem
        self.arm = arm
        self.s1 = ScanStem(STACK, n_beams, out=256, scan_stem="resnet") if "1d" in arm else None
        self.s2 = (MemNet() if arm == "bevmem" else BEVNet(len(FRAMES))) if "bev" in arm else None
        width = 256 * (("1d" in arm) + ("bev" in arm))
        self.head = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 5))

    def forward(self, x1, x2):
        f = []
        if self.s1 is not None:
            f.append(self.s1(x1))
        if self.s2 is not None:
            f.append(self.s2(*x2) if self.arm == "bevmem" else self.s2(x2))
        return self.head(torch.cat(f, 1))


def second(arm, d, t, l, angles, device):
    return mem_inputs(d, t, l, angles, device) if arm == "bevmem" else bev(d, t, l, angles, device)


def evaluate(model, d, t, l, angles, device, bs=96):
    preds, labs = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, t.shape[0], bs):
            tt, ll = t[i:i + bs], l[i:i + bs]
            x1 = stack_1d(d, tt, ll).to(device) if model.s1 is not None else None
            x2 = second(model.arm, d, tt, ll, angles, device) if model.s2 is not None else None
            preds.append(model(x1, x2).cpu()); labs.append(d["lab"][tt, ll])
    model.train()
    P, Y = torch.cat(preds), torch.cat(labs)
    pres = Y[:, 6] > 0.5
    from f1sim.gym_env import PRIV_OPP_DIST_SCALE as S
    res = {"present_rows": int(pres.sum()),
           "presence_acc": float(((P[:, 4] > 0) == pres).float().mean())}
    for c, name in enumerate(["lon", "lat", "vlon", "vlat"]):
        y, p = Y[pres, c], P[pres, c]
        res[f"r2_{name}"] = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-9))
        res[f"mae_{name}"] = float((y - p).abs().mean() * S)
    return res


def train(path, epochs, device, arms, seed):
    D = torch.load(path, weights_only=False)
    tr, te = D["train"], D["test"]
    need = max(max(FRAMES), STACK, MEM_WINDOW + 1)
    t_tr, l_tr = windows(tr, need); t_te, l_te = windows(te, need)
    from f1sim.learn.aligned import beam_angles
    angles = beam_angles(tr["n_beams"], tr["fov"], device=device)
    print(f"train rows {t_tr.shape[0]} (present {int((tr['lab'][t_tr, l_tr][:, 6] > 0.5).sum())}), "
          f"test rows {t_te.shape[0]}", flush=True)
    results = {}
    for arm in arms:
        torch.manual_seed(seed)
        m = Model(arm, tr["n_beams"]).to(device)
        with torch.no_grad():                                         # materialise the lazy layer
            i = torch.arange(4)
            m(stack_1d(tr, t_tr[i], l_tr[i]).to(device) if m.s1 is not None else None,
              second(m.arm, tr, t_tr[i], l_tr[i], angles, device) if m.s2 is not None else None)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        n = t_tr.shape[0]; bs = min(32 if arm == "bevmem" else 256, n); steps = max(1, epochs * (n // bs))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
        t0 = time.time(); k = 0
        for ep in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n - bs + 1, bs):
                sel = perm[i:i + bs]; tt, ll = t_tr[sel], l_tr[sel]
                y = tr["lab"][tt, ll].to(device)
                x1 = stack_1d(tr, tt, ll).to(device) if m.s1 is not None else None
                x2 = second(m.arm, tr, tt, ll, angles, device) if m.s2 is not None else None
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(arm == "bevmem")):
                    p = m(x1, x2).float()
                pres = y[:, 6]
                loss = (((p[:, :4] - y[:, :4]) ** 2).sum(1) * pres).sum() / pres.sum().clamp_min(1) \
                    + F.binary_cross_entropy_with_logits(p[:, 4], pres)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step(); k += 1
            print(f"  {arm} epoch {ep + 1}/{epochs} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
        results[arm] = {"test": evaluate(m, te, t_te, l_te, angles, device),
                        "train": evaluate(m, tr, t_tr[:20000], l_tr[:20000], angles, device),
                        "params": sum(p.numel() for p in m.parameters())}
        print(arm, json.dumps(results[arm]), flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("collect", "train"))
    ap.add_argument("path")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--arms", default="1d,bev,1d+bev,bevmem")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.mode == "collect":
        D = {"train": collect(TRAIN, a.steps, a.envs, a.seed, a.device),
             "test": collect(TEST, a.steps // 2, a.envs, a.seed + 1, a.device)}
        torch.save(D, a.path); print("saved", a.path, flush=True)
    else:
        res = train(a.path, a.epochs, a.device, a.arms.split(","), a.seed)
        with open(a.path.replace(".pt", ".json"), "w") as f:
            json.dump(res, f, indent=1)
