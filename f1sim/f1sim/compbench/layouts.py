"""Box-layout banks for the suites (the exam bank format: per_track[name] = {sid (K, C), pose (K, C, 3), lap (K,), length}).

Two ways to make one, both written to /data/compbench/banks/<id>/ and hashed into the suite:
  - `rekey`: an existing bank's layouts at the same world coordinates under another track's name and lap length (a
    hose variant of the same map: the boxes sit on the racing line, away from the hoses; boxes that end up within
    0.15 m of a wall there are counted and reported)
  - `generate`: the prior developer's `exam_line_bank.py` (3 boxes a layout on a reference line, one per third of the
    lap, lateral N(0, 0.15) cut at 0.4 m, judged passable by the min-time line generator with 0.10 m clearance), run
    unmodified with its hard-coded seed replaced (numpy's first default_rng call) so that fresh layouts come out
"""
import os
import sys
import json
import hashlib

import numpy as np


def bank_sha256(bank_dir: str) -> str:
    files = sorted(f for f in os.listdir(bank_dir) if f.endswith(".pt"))
    h = hashlib.sha256()
    for f in files:
        with open(os.path.join(bank_dir, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def track_length(track_name: str) -> float:
    """the lap length the env's bank check compares against (`TrackSet.length` of that one track)"""
    from f1sim.learn import common
    trs, _ = common.load_tracks([track_name], racelines=False)
    from f1sim.track import TrackTensors
    import torch
    ts = TrackTensors(trs, torch.device("cpu"))
    return float(ts.length[0]), trs[0]


def empty_bank(track_name: str, out_dir: str, C: int = 18) -> dict:
    """one layout without a box, keyed under the track: lets exam_record run a box-free session on a map that has no
    bank yet (the reference drive a new map's layouts are generated from)"""
    import torch
    L, tr = track_length(track_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(dict(per_track={tr.name: dict(sid=torch.full((1, C), -1, dtype=torch.long), pose=torch.zeros(1, C, 3),
                                             lap=torch.zeros(1), length=L)}, C=C,
                    compbench=dict(empty=True, track=track_name)), os.path.join(out_dir, "bank.pt"))
    return dict(track=track_name, key=tr.name, length=L, sha256=bank_sha256(out_dir))


def refline_from_centerline(track_name: str, out_npz: str) -> dict:
    """a reference line for a map without a driven one: the lane's centerline smoothed over 0.5 m
    (`layout_line.smooth_line`, what the bank judge starts from), resampled every 0.1 m (xy, s, L)"""
    from f1sim.layout_line import smooth_line
    L, tr = track_length(track_name)
    xy = smooth_line(np.asarray(tr.centerline, float))
    seg = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    Ls = float(seg.sum())
    q = np.arange(0.0, Ls, 0.1)
    x = np.interp(q, np.concatenate([s, [Ls]]), np.concatenate([xy[:, 0], xy[:1, 0]]))
    y = np.interp(q, np.concatenate([s, [Ls]]), np.concatenate([xy[:, 1], xy[:1, 1]]))
    np.savez(out_npz, xy=np.stack([x, y], 1), s=q, L=np.array(Ls))
    return dict(track=track_name, L=Ls, n=len(q))


def rekey(src_bank: str, track_name: str, out_dir: str) -> dict:
    import torch
    from scipy.ndimage import distance_transform_edt
    files = sorted(f for f in os.listdir(src_bank) if f.endswith(".pt"))
    src = torch.load(os.path.join(src_bank, files[0]), map_location="cpu", weights_only=False)
    key0 = list(src["per_track"].keys())[0]
    ent = dict(src["per_track"][key0])
    L, tr = track_length(track_name)
    ent["length"] = L
    dst = dict(src)
    dst["per_track"] = {tr.name: ent}
    dst["compbench"] = dict(rekeyed_from=os.path.abspath(src_bank), source_key=key0, track=track_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(dst, os.path.join(out_dir, "bank.pt"))
    pose = ent["pose"].numpy().reshape(-1, 3)
    sid = ent["sid"].numpy().reshape(-1)
    xy = pose[sid >= 0, :2]
    occ = np.asarray(tr.occupancy)
    free = occ == 0 if occ.max() <= 1 else occ < 50
    edt = distance_transform_edt(free) * float(tr.resolution)
    org = np.asarray(tr.origin)[:2]
    c = np.floor((xy[:, 0] - org[0]) / tr.resolution).astype(int)
    r = np.floor((xy[:, 1] - org[1]) / tr.resolution).astype(int)
    ok = (r >= 0) & (r < edt.shape[0]) & (c >= 0) & (c < edt.shape[1])
    d = np.where(ok, edt[r.clip(0, edt.shape[0] - 1), c.clip(0, edt.shape[1] - 1)], 0)
    info = dict(track=track_name, key=tr.name, length=L, layouts=int(ent["sid"].shape[0]), boxes=int(len(xy)),
                near_wall_015=int((d < 0.15).sum()), near_wall_025=int((d < 0.25).sum()), sha256=bank_sha256(out_dir))
    json.dump(info, open(os.path.join(out_dir, "info.json"), "w"), indent=1)
    return info


def generate(world: str, refline: str, record: str, out_dir: str, n: int, seed: int, bank_tracks: str = "",
             line_mode: str = "") -> None:
    """exam_line_bank.py unmodified, seed replaced. Runs on the CPU (the 4070 stays free)."""
    import runpy
    from . import TOOLS
    sys.path.insert(0, TOOLS)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if bank_tracks:
        os.environ["BANK_TRACKS"] = bank_tracks
    if line_mode:
        os.environ["BANK_LINE"] = line_mode
    orig = np.random.default_rng
    state = {"done": False}

    def rng_seeded(s=None, *a, **k):
        if not state["done"]:
            state["done"] = True
            print(f"[compbench] exam_line_bank seed {s} -> {seed}", flush=True)
            return orig(seed)
        return orig(s, *a, **k)

    np.random.default_rng = rng_seeded
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "bank.pt")
    sys.argv = [os.path.join(TOOLS, "exam_line_bank.py"), world, refline, record, out, str(int(n)),
                os.path.join(out_dir, "fig")]
    try:
        runpy.run_path(os.path.join(TOOLS, "exam_line_bank.py"), run_name="__main__")
    except Exception as e:                                       # the figures at the end need a boxed record
        if not os.path.exists(out):
            raise
        print(f"[compbench] bank saved; figures skipped ({type(e).__name__}: {e})", flush=True)
    json.dump(dict(world=world, refline=refline, n=n, seed=seed, bank_tracks=bank_tracks, line_mode=line_mode,
                   sha256=bank_sha256(out_dir)), open(os.path.join(out_dir, "info.json"), "w"), indent=1)



def grid_record(track_name: str, out_npz: str) -> dict:
    """a stand-in for a session record on a map nobody has driven yet: only the walls-only distance grid and the car
    size that exam_line_bank.py reads from its record (`rec_lib.Rec.wall_dist`); no sessions (its figures are skipped)"""
    import torch
    from f1sim.learn import common
    from f1sim.track import TrackTensors
    trs, _ = common.load_tracks([track_name], racelines=False)
    tt = TrackTensors(trs, torch.device("cpu"))
    np.savez(out_npz, edt=tt.edt.float().numpy(), t_off=tt.t_off.numpy(), t_H=tt.t_H.numpy(), t_W=tt.t_W.numpy(),
             t_res=tt.t_res.numpy(), t_origin=tt.t_origin.numpy(), car=np.array([0.58, 0.31]), dt=np.array(0.1),
             rs=np.zeros((1, 1), bool), cnt=np.zeros((1, 1, 6), np.float32), st=np.zeros((2, 1, 4), np.float32),
             on=np.zeros((2, 1), bool), sid=np.full((2, 1, 1), -1, np.int16), pose=np.zeros((2, 1, 1, 3), np.float32),
             name=np.array(f"grid of {track_name}"))
    return dict(track=track_name, out=out_npz)


def generate_bounded(world: str, refline: str, out_dir: str, n: int, seed: int, bank_tracks: str, max_rounds: int = 12) -> dict:
    """exam_line_bank.py's design, ported so that it always ends (11-28: on lab12x16 the original looped for > 10 minutes on
    the last layout -- one stretch of that floor has no passable place for a box within 0.4 m of the centerline, and the
    original redraws until every one of its N x 3 stretches holds a passable box). Same rules: the lap cut into N x 3
    stretches, one box in each, layout i's three boxes about a third of a lap apart; lateral N(0, 0.15) cut at 0.4 m;
    the venue's measured box sizes; judged by the same min-time line generator (`layout_line.LayoutLines`, 300 iterations,
    the car body 0.10 m clear of every box and wall, curvature within the steering limit); a failed layout redraws the
    boxes in the way, after 4 rounds moving them inside their own stretch. NEW: after `max_rounds` rounds a box still in
    the way is dropped (the layout keeps the others) and the drop is counted in info.json."""
    import argparse
    import time
    import torch
    from scipy.ndimage import uniform_filter1d
    from f1sim.learn import state_rl as SR
    from f1sim import layout_line as LL
    from f1sim.raceline import curvature
    from f1sim import props as _props
    from types import SimpleNamespace
    NEAR, SIG, N_BOX, NB, ITERS, CLR_PROP, CLR_WALL = 0.40, 0.15, 3, 32, 300, 0.10, 0.10
    rng = np.random.default_rng(seed)
    dev = torch.device("cpu")
    a = argparse.Namespace(**json.load(open(world)))
    a.envs, a.seed, a.device = 2, seed, "cpu"
    a.prop_place, a.prop_count, a.prop_fraction, a.prop_bank = "uniform", "fixed", 1.0, ""
    a.warps, a.track_scales = 0, None
    a.tracks = bank_tracks
    env = SR.build_env(a, dev)
    track = env.sim.track.tracks[0]
    rls = [SimpleNamespace(xy=LL.smooth_line(np.asarray(track.centerline, float)))]
    builder = LL.LayoutLines(track, rls[0], dev, veh_width=float(env.cfg.vehicle.width), vehicle=env.cfg.vehicle, iters=ITERS)
    kmax = builder.kmax
    C = env.procedural.C
    edt = env.sim.track.edt.float().numpy()
    off, H, W = int(env.sim.track.t_off[0]), int(env.sim.track.t_H[0]), int(env.sim.track.t_W[0])
    res, org = float(env.sim.track.t_res[0]), env.sim.track.t_origin[0].numpy()
    grid = edt[off: off + H * W].reshape(H, W)

    def wall_dist(pts):
        c = np.floor((pts[:, 0] - org[0]) / res).astype(int)
        r = np.floor((pts[:, 1] - org[1]) / res).astype(int)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        out = np.zeros(len(pts))
        out[ok] = grid[r[ok], c[ok]]
        return out

    ref = np.load(refline)
    refp, refs, L = ref["xy"], ref["s"], float(ref["L"])
    tan = np.roll(refp, -1, 0) - np.roll(refp, 1, 0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    po = env.procedural
    pool = po.line_pool.cpu().numpy()
    FP = {int(k): np.asarray(_props.build(po.shapes[k].style, seed=0, **dict(po.shapes[k].dims)).envelope.footprint, float)
          for k in pool}

    def poly(k, x, y, yaw):
        q = FP[k]
        c, s_ = np.cos(yaw), np.sin(yaw)
        return np.stack([x + q[:, 0] * c - q[:, 1] * s_, y + q[:, 0] * s_ + q[:, 1] * c], 1)

    def lateral():
        while True:
            d = rng.normal(0.0, SIG)
            if abs(d) <= NEAR:
                return d

    def place(s):
        j = int(np.argmin(np.abs(((refs - s + 0.5 * L) % L) - 0.5 * L)))
        k = int(rng.choice(pool))
        for _ in range(50):
            d = lateral()
            yaw = rng.uniform(0.0, np.pi)
            x, y = refp[j] + d * nrm[j]
            if wall_dist(np.vstack([poly(k, x, y, yaw), [[x, y]]])).min() >= CLR_WALL:
                return (k, x, y, yaw, refs[j], d)
        return None

    K = n * N_BOX
    phase0 = rng.uniform(0.0, L / K)
    st = dict(judged=0, failed=0, moved=0, redrawn=0, dropped=0)

    def stretch(k):
        return (phase0 + (k + rng.uniform(0.2, 0.8)) * L / K) % L

    def fill(lay, jb):
        for _ in range(40):
            b = place(lay["s"][jb])
            if b is not None:
                lay["boxes"][jb] = b
                return
            st["moved"] += 1
            lay["s"][jb] = stretch(lay["slot"][jb])
        lay["boxes"][jb] = None                                 # no wall-clear place at all in this stretch

    def near_perm(w):
        return np.argsort(np.arange(n) + rng.uniform(-w, w, n))

    p1, p2 = near_perm(n / 8), near_perm(n / 8)
    lays = []
    for i in range(n):
        slot = np.array([i, n + p1[i], 2 * n + p2[p1[i]]])
        lays.append(dict(slot=slot, s=np.array([stretch(k) for k in slot]), boxes=[None] * N_BOX, rnd=0))
    for lay in lays:
        for jb in range(N_BOX):
            fill(lay, jb)
    pend, laps_ok, t0 = list(range(n)), {}, time.time()
    while pend:
        batch, pend = pend[:NB], pend[NB:]
        lines, laps, clr, _ = builder.build([[poly(*b[:4]) for b in lays[i]["boxes"] if b is not None] for i in batch])
        for i, xy, lp, (cp, cw) in zip(batch, lines, laps, clr):
            st["judged"] += 1
            kap = np.abs(uniform_filter1d(curvature(xy), 5, mode="wrap"))
            if cp >= CLR_PROP and cw >= CLR_WALL and kap.max() <= kmax and np.isfinite(lp):
                laps_ok[i] = float(lp)
                continue
            st["failed"] += 1
            lay = lays[i]
            lay["rnd"] += 1
            live = [jb for jb in range(N_BOX) if lay["boxes"][jb] is not None]
            if not live:                                        # the empty floor fails: nothing a box can be blamed for
                laps_ok[i] = float(lp) if np.isfinite(lp) else float("nan")
                continue
            P = [poly(*lay["boxes"][jb][:4]) for jb in live]
            cpi = np.array([LL.check_line(xy, [p_], track)[0] for p_ in P])
            rm = cpi < CLR_PROP
            hot = xy[kap > kmax]
            if len(hot):
                cen = np.array([[lay["boxes"][jb][1], lay["boxes"][jb][2]] for jb in live])
                rm |= np.linalg.norm(cen[:, None] - hot[None], axis=-1).min(1) < 1.0
            if not rm.any():
                rm[int(np.argmin(cpi))] = True
            for q in np.nonzero(rm)[0]:
                jb = live[q]
                if lay["rnd"] > max_rounds:
                    lay["boxes"][jb] = None
                    st["dropped"] += 1
                    continue
                st["redrawn"] += 1
                if lay["rnd"] > 4:
                    st["moved"] += 1
                    lay["s"][jb] = stretch(lay["slot"][jb])
                fill(lay, jb)
            pend.append(i)
        print(f"  [{(time.time() - t0) / 60:4.1f} min] passed {len(laps_ok)}/{n}, judged {st['judged']}, failed {st['failed']}, "
              f"redrawn {st['redrawn']}, moved {st['moved']}, dropped {st['dropped']}", flush=True)
    sid = torch.full((n, C), -1, dtype=torch.long)
    pose = torch.zeros(n, C, 3)
    for i in range(n):
        c = 0
        for b in lays[i]["boxes"]:
            if b is None:
                continue
            sid[i, c] = b[0]
            pose[i, c] = torch.tensor([b[1], b[2], b[3]])
            c += 1
    os.makedirs(out_dir, exist_ok=True)
    torch.save(dict(per_track={track.name: dict(sid=sid, pose=pose, lap=torch.tensor([laps_ok[i] for i in range(n)]),
                                                length=float(env.sim.track.length[0]))},
                    C=C, judge=dict(clr_prop=CLR_PROP, clr_wall=CLR_WALL, kmax=kmax, iters=ITERS, near=NEAR, sig=SIG, n_box=N_BOX,
                                    refline=os.path.abspath(refline), seed=seed, max_rounds=max_rounds,
                                    generator="compbench.layouts.generate_bounded")), os.path.join(out_dir, "bank.pt"))
    nb = (sid >= 0).sum(1)
    info = dict(world=world, refline=refline, n=n, seed=seed, bank_tracks=bank_tracks, stats=st,
                boxes_per_layout={int(k): int((nb == k).sum()) for k in range(N_BOX + 1)}, sha256=bank_sha256(out_dir))
    json.dump(info, open(os.path.join(out_dir, "info.json"), "w"), indent=1)
    return info


if __name__ == "__main__":
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    cmd = sys.argv[1]
    if cmd == "rekey":
        print(json.dumps(rekey(sys.argv[2], sys.argv[3], sys.argv[4]), indent=1))
    elif cmd == "generate":
        # world refline record out_dir n seed [bank_tracks] [line_mode]
        generate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]),
                 sys.argv[8] if len(sys.argv) > 8 else "", sys.argv[9] if len(sys.argv) > 9 else "")
    elif cmd == "sha":
        print(bank_sha256(sys.argv[2]))
    elif cmd == "empty":
        print(json.dumps(empty_bank(sys.argv[2], sys.argv[3]), indent=1))
    elif cmd == "bounded":
        # world refline out_dir n seed bank_tracks
        print(json.dumps(generate_bounded(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]), indent=1))
    elif cmd == "gridrec":
        print(json.dumps(grid_record(sys.argv[2], sys.argv[3]), indent=1))
    elif cmd == "refline":
        print(json.dumps(refline_from_centerline(sys.argv[2], sys.argv[3]), indent=1))

