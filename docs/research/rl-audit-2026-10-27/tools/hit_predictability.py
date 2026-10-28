"""Can a hard hit be seen coming, and from which inputs? (2026-10-29)

The race teacher's hit head (P(hard hit within 20 decisions)) rises only 0.5 s before its hits (median P 0.02 / 0.05 /
0.26 / 0.44 at 2 / 1 / 0.5 / 0 s, crash_scenes.py) and its actions do not change when the props are erased. Either the
hits are not predictable 1-2 s ahead from what the teacher observes (an observation problem: add the missing input),
or they are and the teacher has not learned it (a learning-signal problem: fix the signal, not the input).

This rolls a state_rl checkpoint's MEAN action out in its training world, logs every decision's observation plus
candidate extra inputs, and fits classifiers for "a hard wall/prop hit" and "a hard car contact" within 0.5 / 1 / 2 s
(10 / 20 / 40 decisions) on held-out cars:
  T       the teacher's own observation (state block + its 108-beam scan)
  T+geo   + per prop slot its extent (farthest face from the centre) and speed (a shove moves it), the opponent's
          acceleration (its velocity change over one decision, ego frame)
  T x3    the teacher observation at t, t-1, t-2 decisions (motion of everything, implicitly)
  T+scan  + the full 1081-beam noise-free scan through a small conv encoder (the student's resolution)
Falsifier for "an observation problem": T+geo / T x3 / T+scan reach the same held-out AUC as T at 1 and 2 s.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl  # noqa: E402
from f1sim.learn.state_rl import ActorCritic, RunningNorm, StateObs  # noqa: E402

HOR = (10, 20, 40)


def auc(score, y):
    """rank AUC (ties ignored)"""
    o = torch.argsort(score)
    r = torch.empty_like(o, dtype=torch.float64); r[o] = torch.arange(1, len(o) + 1, dtype=torch.float64, device=o.device)
    n1 = float(y.sum()); n0 = float(len(y) - n1)
    if n1 < 5 or n0 < 5:
        return float("nan")
    return float((r[y > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--stage", default="race")
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--warps", type=int, default=16)
    ap.add_argument("--decisions", type=int, default=2400)
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=93)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    t_beams = int(ck["args"].get("beams", 36)); t_scan = bool(ck["args"].get("scan_obs", 0))
    a.beams = 1081
    env = state_rl.build_env(a, device)
    from f1sim.learn.graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    base = StateObs(env)
    env.reset(seed=int(a.seed))
    lid = env.learner_ids; B = int(lid.numel())
    ang_s = env.sim.lidar.angles.float()
    ang_t = torch.linspace(float(ang_s[0]), float(ang_s[-1]), t_beams, device=device)
    pos = (ang_t - ang_s[0]) / (ang_s[-1] - ang_s[0]) * (ang_s.numel() - 1)
    i0 = pos.floor().long().clamp(0, ang_s.numel() - 2); wgt = (pos - i0.float())[None, :]

    def tobs(last):
        o_ = base(last)
        if not t_scan:
            return o_
        sc = env.last_result.scan_true[lid].float()
        return torch.cat([o_, ((sc[:, i0] * (1 - wgt) + sc[:, i0 + 1] * wgt) / 10.0).clamp(0.0, 1.0)], 1)

    last = torch.zeros(B, 2, device=device)
    o = tobs(last)
    model = ActorCritic(o.shape[1]).to(device); model.load_state_dict(ck["model"], strict=False); model.eval()
    norm = RunningNorm(o.shape[1], device); norm.load_state_dict(ck["norm"])
    act_full = torch.zeros(env.B, env.act_dim, device=device)
    N = int(a.decisions)
    X = torch.zeros(N, B, o.shape[1], dtype=torch.float16, device=device)
    S = torch.zeros(N, B, 1081, dtype=torch.float16, device=device)
    G = torch.zeros(N, B, 9, device=device)                 # extent x3, prop speed x3, opponent accel (ax, ay), |a|
    HW = torch.zeros(N, B, dtype=torch.bool, device=device); HC = torch.zeros(N, B, dtype=torch.bool, device=device)
    DN = torch.zeros(N, B, dtype=torch.bool, device=device)
    touching = torch.zeros(B, dtype=torch.bool, device=device)
    po = env.procedural
    opp_v_prev = None
    t0 = time.time()
    with torch.no_grad():
        for t in range(N):
            X[t] = o.half(); S[t] = (env.last_result.scan_true[lid].float() / 10.0).clamp(0, 1).half()
            st = env.last_result.state
            g = torch.zeros(B, 9, device=device)
            if po is not None and po.C > 0:
                ste = st[lid]
                d = po.p_poses[lid, :, :2] - ste[:, None, :2]
                c, s_ = torch.cos(ste[:, 2])[:, None], torch.sin(ste[:, 2])[:, None]
                dx = d[..., 0] * c + d[..., 1] * s_
                dist = d.norm(dim=-1)
                live = (po.p_sid[lid] >= 0) & (dx > -1.0) & (dist < 12.0)
                key = torch.where(live, dist, torch.full_like(dist, 1e6))
                j = key.topk(3, dim=1, largest=False).indices
                ext = torch.where(torch.isfinite(po.p_d[lid]), po.p_d[lid], torch.full_like(po.p_d[lid], -1e9)).amax(-1)
                pv = po.p_vel[lid].norm(dim=-1)
                pres = live.gather(1, j).float()
                g[:, 0:3] = ext.gather(1, j).clamp(0, 1) * pres
                g[:, 3:6] = pv.gather(1, j).clamp(0, 5) * pres
            if env.M > 1:
                dx_, dy_, dyaw_, dvx_, dvy_, dist_, present_ = env._opp_rel(st)
                k_ = torch.where(present_, dist_, torch.full_like(dist_, 1e6)).argmin(1, keepdim=True)
                v_now = torch.stack([dvx_.gather(1, k_)[:, 0], dvy_.gather(1, k_)[:, 0]], 1)[lid]
                if opp_v_prev is not None:
                    acc = (v_now - opp_v_prev) / (env.sim.control_dt * 2)
                    g[:, 6:8] = acc.clamp(-20, 20) / 10.0; g[:, 8] = acc.norm(dim=1).clamp(0, 20) / 10.0
                opp_v_prev = v_now
            G[t] = g
            act = model.pi(norm(o)).clamp(-1, 1)
            act_full[lid] = act
            hw = torch.zeros(B, dtype=torch.bool, device=device); hc = torch.zeros(B, dtype=torch.bool, device=device)
            dn = torch.zeros(B, dtype=torch.bool, device=device)
            for r in range(2):
                _, r_, te, trc, info = env.step(act_full)
                res = env.last_result
                on, imp = res.contact_onsets[lid], res.contact_impacts[lid]
                hw |= (on[:, 0] & (imp[:, 0] >= 1.5)) | (on[:, 1] & (imp[:, 1] >= 1.5))
                car = info["reward_components"]["car_contact"][lid].float(); tch = car < 0
                hc |= tch & ~touching
                d_ = (te | trc)[lid]; touching = tch & ~d_
                dn |= d_
            HW[t] = hw; HC[t] = hc; DN[t] = dn
            last = torch.where(dn[:, None], torch.zeros_like(act), act)
            if dn.any():
                opp_v_prev = None if env.M > 1 and bool(dn.all()) else opp_v_prev
            o = tobs(last)
            if t % 400 == 0:
                print(f"[예측성] 굴림 {t}/{N} 결정, {time.time() - t0:.0f} s, 벽+상자 {int(HW[:t + 1].sum())} 차 {int(HC[:t + 1].sum())}", flush=True)
    # labels: a hit within (t, t+K] decisions of the same episode; masks: complete windows, and not within 10 decisions
    # after a hit (the seconds after a hit replay it)
    ep = torch.cumsum(DN.long(), 0) - DN.long()            # episode index per (t, row): increments after a done
    Y = {}
    valid = torch.ones(N, B, dtype=torch.bool, device=device)
    any_hit = HW | HC
    since = torch.full((B,), 999, device=device, dtype=torch.long)
    for t in range(N):
        valid[t] &= since >= 10
        since = torch.where(any_hit[t], torch.zeros_like(since), since + 1)
        since = torch.where(DN[t], torch.full_like(since, 999), since)
    for K in HOR:
        for nm, H in (("ob", HW), ("car", HC)):
            y = torch.zeros(N, B, dtype=torch.bool, device=device); ok = torch.zeros(N, B, dtype=torch.bool, device=device)
            for t in range(N - K):
                fut = H[t:t + K]                                # hits at decisions t..t+K-1 (after the action at t)
                same = (ep[t:t + K] == ep[t][None])
                y[t] = (fut & same).any(0)
                ok[t] = True
            Y[(nm, K)] = (y, ok & valid)
    print(f"[예측성] 굴림 끝: {N * B} 결정, 벽+상자 {int(HW.sum())}회, 차 {int(HC.sum())}회", flush=True)
    rows = torch.arange(B, device=device)
    test_rows = rows % 5 == 0
    # stacked teacher obs (t, t-1, t-2), the same episode only (else repeat the oldest valid)
    Xf = X.float()
    X1 = torch.cat([Xf[:1], Xf[:-1]], 0); X2 = torch.cat([X1[:1], X1[:-1]], 0)
    e1 = torch.cat([ep[:1], ep[:-1]], 0); e2 = torch.cat([e1[:1], e1[:-1]], 0)
    X1 = torch.where((e1 == ep)[..., None], X1, Xf); X2 = torch.where((e2 == ep)[..., None], X2, X1)
    feats = {"T": Xf, "T+geo": torch.cat([Xf, G], -1), "Tx3": torch.cat([Xf, Xf - X1, Xf - X2], -1)}
    keys = [(nm, K) for K in HOR for nm in ("ob", "car")]
    Yall = torch.stack([Y[k][0] for k in keys], -1).float()      # (N, B, 6)
    Mall = torch.stack([Y[k][1] for k in keys], -1).float()

    class ScanNet(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(1, 16, 9, stride=4), nn.ReLU(), nn.Conv1d(16, 32, 7, stride=3), nn.ReLU(),
                                      nn.Conv1d(32, 32, 5, stride=2), nn.ReLU(), nn.AdaptiveAvgPool1d(16), nn.Flatten())
            self.mlp = nn.Sequential(nn.Linear(d_in + 512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, len(keys)))

        def forward(self, x, s):
            return self.mlp(torch.cat([x, self.conv(s[:, None, :])], 1))

    def fit(name, F_, use_scan=False):
        Dm = F_.shape[-1]
        tr = ~test_rows; te = test_rows
        Ftr = F_[:, tr].reshape(-1, Dm); Ytr = Yall[:, tr].reshape(-1, len(keys)); Mtr = Mall[:, tr].reshape(-1, len(keys))
        Fte = F_[:, te].reshape(-1, Dm); Yte = Yall[:, te].reshape(-1, len(keys)); Mte = Mall[:, te].reshape(-1, len(keys))
        Str = S[:, tr].reshape(-1, 1081) if use_scan else None; Ste = S[:, te].reshape(-1, 1081) if use_scan else None
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-3
        torch.manual_seed(0)
        net = ScanNet(Dm).to(device) if use_scan else nn.Sequential(nn.Linear(Dm, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                                                     nn.Linear(512, len(keys))).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        pw = ((Mtr * (1 - Ytr)).sum(0) / (Mtr * Ytr).sum(0).clamp_min(1)).clamp(1, 200)
        n = Ftr.shape[0]
        for it in range(int(a.iters)):
            j = torch.randint(0, n, (4096,), device=device)
            x = ((Ftr[j] - mu) / sd).clamp(-8, 8)
            lg = net(x, Str[j].float()) if use_scan else net(x)
            loss = (nn.functional.binary_cross_entropy_with_logits(lg, Ytr[j], pos_weight=pw, reduction="none") * Mtr[j]).sum() / Mtr[j].sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, Fte.shape[0], 65536):
                x = ((Fte[i:i + 65536] - mu) / sd).clamp(-8, 8)
                out.append(net(x, Ste[i:i + 65536].float()) if use_scan else net(x))
        lg = torch.cat(out, 0)
        res = {}
        for q, k in enumerate(keys):
            m = Mte[:, q] > 0
            res[k] = auc(lg[m, q], Yte[m, q])
        print(f"[예측성] {name:7s} | 시험 AUC  벽+상자 0.5/1/2 s: " + " / ".join(f"{res[('ob', K)]:.3f}" for K in HOR)
              + " | 차 0.5/1/2 s: " + " / ".join(f"{res[('car', K)]:.3f}" for K in HOR), flush=True)
        return res

    # the teacher's own hit head (trained on its exploration-noise rollouts), as a reference on the same test samples
    with torch.no_grad():
        ph = []
        Xt = Xf[:, test_rows].reshape(-1, Xf.shape[-1])
        for i in range(0, Xt.shape[0], 65536):
            ph.append(torch.sigmoid(model.value_and_hit(norm(Xt[i:i + 65536]))[1]))
        ph = torch.cat(ph)
        Yte = Yall[:, test_rows].reshape(-1, len(keys)); Mte = Mall[:, test_rows].reshape(-1, len(keys))
        r = {k: auc(ph[Mte[:, q] > 0], Yte[Mte[:, q] > 0, q]) for q, k in enumerate(keys)}
        print(f"[예측성] 선생의 충돌 예측 머리 | 시험 AUC 벽+상자 0.5/1/2 s: " + " / ".join(f"{r[('ob', K)]:.3f}" for K in HOR)
              + " | 차 0.5/1/2 s: " + " / ".join(f"{r[('car', K)]:.3f}" for K in HOR), flush=True)
    for nm, F_ in feats.items():
        fit(nm, F_)
    fit("T+scan", Xf, use_scan=True)


if __name__ == "__main__":
    main()
