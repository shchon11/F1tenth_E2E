"""Real /scan vs the simulator's LiDAR cast at the particle-filter pose on the venue map (korea_2026_competition,
the /map the car localised against). Plus per-beam static noise from standstill scans."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim import maps                                   # noqa: E402
from f1sim.lidar import Lidar, HIT_DUCT, HIT_TALL, HIT_GROUND, HIT_NONE   # noqa: E402
from f1sim.params import Config                          # noqa: E402
from f1sim.randomization import ParamSet                 # noqa: E402
from f1sim.track import TrackTensors                     # noqa: E402

import sig                                               # noqa: E402

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
BAGS = ["20260827-115713", "20260826-192819", "20260826-193250"]
SENT = 65.533


def valid_real(s):
    return (s > 0.02) & (s < 29.9) & np.isfinite(s)


def main():
    tr = maps.load("real:korea_2026_competition")
    tt = TrackTensors(tr, "cpu")
    cfg = Config()
    ps = ParamSet(cfg, 1, torch.device("cpu"), torch.Generator().manual_seed(0))
    li = Lidar(tt, cfg.lidar.n_beams, cfg.lidar.fov, "cpu", compile=False)
    res = {}
    allr = []
    for b in BAGS:
        f = [x for x in sig.files("comp") if b in x][0]
        Z = np.load(f)
        st, sc = Z["scan_t"], Z["scan"]
        pf = Z["pf"]; core = Z["core"]
        v = np.interp(st, core[:, 0], core[:, 2]) / 4204.0
        yaw_u = np.unwrap(pf[:, 4])
        # lag between the scan's record time and the pf pose that best explains it: grid on a moving subset
        mv = np.flatnonzero(v > 2.0)
        sub_idx = mv[::max(1, len(mv) // 150)]
        best = None
        for lag in np.arange(-0.06, 0.061, 0.01):
            pr = cast(li, ps, tt, pf, yaw_u, st[sub_idx] + lag, noisy=False)
            rr = sc[sub_idx]
            both = valid_real(rr) & (pr[0] < 29.9)
            med = np.median(np.abs(rr[both] - pr[0][both]))
            if best is None or med < best[0]:
                best = (med, lag)
        lag = best[1]
        idx = np.arange(0, len(st), 4)                 # every 4th scan (10 Hz)
        idx = idx[(st[idx] > pf[0, 0] + 0.1) & (st[idx] < pf[-1, 0] - 0.1)]
        r_sim_true, r_sim, typ, cosinc = [], [], [], []
        for c0 in range(0, len(idx), 200):
            ii = idx[c0:c0 + 200]
            a, bnoisy, t_, ci = cast(li, ps, tt, pf, yaw_u, st[ii] + lag, noisy=True)
            r_sim_true.append(a); r_sim.append(bnoisy); typ.append(t_); cosinc.append(ci)
        r_sim_true = np.concatenate(r_sim_true); r_sim = np.concatenate(r_sim); typ = np.concatenate(typ)
        cosinc = np.concatenate(cosinc)
        rr = sc[idx]; vv = v[idx]
        rv, sv = valid_real(rr), r_sim < 29.9
        both = rv & sv
        d = rr - r_sim_true
        out = dict(lag=float(lag), n_scans=int(len(idx)),
                   real_valid=float(rv.mean()), sim_valid=float(sv.mean()),
                   real_only=float((rv & ~sv).mean()), sim_only=float((sv & ~rv).mean()),
                   med_abs_res=float(np.median(np.abs(d[both]))),
                   within_5cm=float(np.mean(np.abs(d[both]) < 0.05)), within_10cm=float(np.mean(np.abs(d[both]) < 0.10)),
                   real_shorter_50cm=float(np.mean(d[both] < -0.5)), real_longer_50cm=float(np.mean(d[both] > 0.5)),
                   real_gt10m=float(np.mean(rv & (rr > 10))))
        # by speed
        for lo, hi in ((0, 0.05), (2, 5), (5, 10)):
            m = (vv >= lo) & (vv < hi)
            if m.sum() > 10:
                bb = both[m]
                out[f"med_abs_res_v{lo}-{hi}"] = float(np.median(np.abs(d[m][bb])))
                out[f"real_valid_v{lo}-{hi}"] = float(rv[m].mean()); out[f"sim_valid_v{lo}-{hi}"] = float(sv[m].mean())
        # no-return rate vs incidence on duct hits (sim-predicted surface), real vs sim
        duct = (typ == HIT_DUCT) & (r_sim_true < 8.0)
        bins = [0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.01]
        inc = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = duct & (cosinc >= lo) & (cosinc < hi)
            if m.sum() > 50:
                inc.append((lo, hi, int(m.sum()), float((~rv[m]).mean()), float((~sv[m]).mean())))
        out["noreturn_by_cos"] = inc
        # by range on duct hits
        rng = []
        for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10)):
            m = duct & (r_sim_true >= lo) & (r_sim_true < hi)
            if m.sum() > 50:
                rng.append((lo, hi, int(m.sum()), float((~rv[m]).mean()), float((~sv[m]).mean()),
                            float(np.median(np.abs(d[m & both])))))
        out["by_range"] = rng
        res[b] = out
        print(b, json.dumps(out), flush=True)
        if b == BAGS[0]:
            k = len(idx) // 2
            for kk in (k, k + 40):
                allr.append(dict(t=float(st[idx[kk]]), real=rr[kk].tolist(), sim=r_sim[kk].tolist(), v=float(vv[kk])))
        # static per-beam noise on standstill windows of this bag
        still = v < 0.005
        runs, cur = [], []
        for i in range(len(st)):
            if still[i]:
                cur.append(i)
            else:
                if len(cur) > 80:
                    runs.append(cur)
                cur = []
        if len(cur) > 80:
            runs.append(cur)
        stds, mads, means = [], [], []
        for rn in runs:
            for c0 in range(0, len(rn) - 40, 40):
                blk = sc[rn[c0:c0 + 40]]
                ok = valid_real(blk).all(0)
                if ok.sum() < 100:
                    continue
                x = blk[:, ok]
                stds.append(x.std(0)); mads.append(np.median(np.abs(x - np.median(x, 0)), 0)); means.append(x.mean(0))
        if stds:
            stds, mads, means = np.concatenate(stds), np.concatenate(mads), np.concatenate(means)
            tab = []
            for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 6), (6, 10), (10, 30)):
                m = (means >= lo) & (means < hi)
                if m.sum() > 50:
                    tab.append((lo, hi, int(m.sum()), float(np.median(stds[m])), float(np.median(mads[m])),
                                float(np.percentile(stds[m], 90)), float(cfg.lidar.noise_std + cfg.lidar.noise_std_rel * (lo + hi) / 2)))
            res[b]["static_noise"] = tab
            print("  static noise (lo, hi, n, med std, med MAD, p90 std, sim sigma)", tab, flush=True)
    json.dump(res, open(os.path.join(HERE, "lidar_cmp.json"), "w"), indent=1)
    json.dump(allr, open(os.path.join(HERE, "lidar_examples.json"), "w"))


def cast(li, ps, tt, pf, yaw_u, ts, noisy):
    """sim scan at base_link pose interpolated from the particle filter at times ts (motion over one period)."""
    def pose_at(t):
        return np.stack([np.interp(t, pf[:, 0], pf[:, 2]), np.interp(t, pf[:, 0], pf[:, 3]), np.interp(t, pf[:, 0], yaw_u)], 1)
    p1 = torch.tensor(pose_at(ts), dtype=torch.float32)
    p0 = torch.tensor(pose_at(ts - 0.025), dtype=torch.float32)
    B = len(ts)
    P = {k: v.expand(B).clone() for k, v in ps.P.items()}
    r, rt, typ = li.scan(p1, p0, P, motion_distortion=True, noisy=True, att=torch.zeros(B, 2), att_prev=torch.zeros(B, 2),
                         compiled=False)
    # incidence on the predicted surface: |cos| between the horizontal beam and the duct distance-field gradient
    origin, dh, k = li.rays(p1, torch.zeros(B, 2), P, per_beam=False)
    hit_xy = origin[..., :2] + dh * (rt / torch.sqrt(1.0 + k * k))[..., None]
    nrm = tt.edt_gradient(hit_xy, torch.zeros_like(k, dtype=torch.long), tt.edt_duct)
    cos_inc = (nrm * dh).sum(-1).abs()
    rt_np = rt.numpy().copy(); rt_np[typ.numpy() == HIT_NONE] = SENT
    if not noisy:
        return (rt_np,)
    return rt_np, r.numpy(), typ.numpy(), cos_inc.numpy()


if __name__ == "__main__":
    main()
