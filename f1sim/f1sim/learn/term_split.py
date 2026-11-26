"""(11-26, RQ2; plan_rl --term-split) Do the reward's terms cancel in the policy gradient, or is the signal too small?

A measurement on a FROZEN checkpoint, no update made. plan_rl collects one rollout of the normal size with the checkpoint's
own arguments and world (after `--term-split-warm` warm-up rollouts, so the episodes are mid-way and the memory prefix is
there), then this module:

* splits each decision's reward into R_lap (the lap bonus and the distance shaping) and R_contact (every contact cost: the
  contacts, the plan-hit price, the plan-car cost); anything else (wrong-way, commitment) is reported as a remainder;
* fits one value function per term on the rollout's critic inputs (the checkpoint critic's scan encoder and input statistics,
  frozen; a copy of its MLP each -- R_contact's starts from a zero output layer), a few hundred Adam steps on TD(lambda) targets
  re-made every round with the run's gamma / lambda;
* makes GAE advantages per term exactly as the trainer does (a restart bootstraps from the value before it, held decisions run
  lambda-free), normalised per minibatch as training normalises them, and raw;
* takes the policy-gradient vector of each term on the actor's policy weights for every minibatch of the update (the trainer's
  own `batches`, `actor_on`, `lp_scored`: the ratio-1 loss -(logp x A x made) / n) -> norms, cos(mean G_lap, mean G_contact),
  agreement between minibatches (mean pairwise cosine), the noise scale B_noise (McCandlish et al. 2018, as plan_rl measures
  it), and the share of the gradient carried by decisions within `NEAR_S` before a box pass (box slots and the car's distance
  to go along the walls-only field: env truth, `plan_rl.box_along`).

Reading (the plan, 11-26): H_opt = the terms cancel (cos <= -0.5, sizes alike) -> the reward's shape; H_starve = both far below
the noise, cos ~ 0 -> exploration / credit."""
import copy
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

NEAR_S = 1.5                  # [s] a decision this close before a box pass is "near"
PASS_REACH = 3.0              # [m] a pass: the box 0..3 m ahead at one decision, 0..3 m behind at the next


def gae(R: torch.Tensor, V: torch.Tensor, v_last: torch.Tensor, D: torch.Tensor, FM: torch.Tensor, gamma: float,
        lam: float) -> torch.Tensor:
    """plan_rl's GAE, one term: (T, B) rewards, values, restarts (1 = the row restarted in the decision), new-plan mask;
    a restart bootstraps from the value before it, inside a held plan lambda is 1"""
    T = R.shape[0]
    adv = torch.zeros_like(R)
    g = torch.zeros_like(R[0])
    for t in reversed(range(T)):
        v_next = v_last if t == T - 1 else V[t + 1]
        v_next = torch.where(D[t] > 0, V[t], v_next)
        delta = R[t] + gamma * v_next - V[t]
        lam_t = lam if t == T - 1 else torch.where(FM[t + 1] > 0, torch.full_like(g, lam), torch.ones_like(g))
        g = delta + gamma * lam_t * (1.0 - D[t]) * g
        adv[t] = g
    return adv


def near_passes(ds: torch.Tensor, live: torch.Tensor, pos: torch.Tensor, D: torch.Tensor, k: int):
    """(T, B, C) along-track distance ahead to each box slot at every decision's start, live, the slot's (x, y); (T, B)
    restarts. -> (passes (T, B) bool: a box passed during decision t, near (T, B) bool: decision t is at most k-1 decisions
    before a pass in the same episode, i.e. within k x 0.1 s of it)"""
    T = ds.shape[0]
    ds = ds.float(); pos = pos.float()
    same = ((pos[1:] - pos[:-1]).abs().sum(-1) <= 0.05) & live[1:] & live[:-1]
    crossed = same & (ds[:-1] > 0.0) & (ds[:-1] <= PASS_REACH) & (ds[1:] <= 0.0) & (ds[1:] > -PASS_REACH)
    p = torch.zeros(T, ds.shape[1], dtype=torch.bool, device=ds.device)
    p[:-1] = crossed.any(-1) & (D[:-1] <= 0)
    near = torch.zeros_like(p)
    # walk back from each pass until k decisions or a restart (a decision that restarted ends the episode before the pass)
    cur = torch.zeros(ds.shape[1], dtype=torch.long, device=ds.device)      # decisions left of the window
    for t in reversed(range(T)):
        cur = torch.where(D[t] > 0, torch.zeros_like(cur), cur)
        cur = torch.where(p[t], torch.full_like(cur, k), cur)
        near[t] = cur > 0
        cur = (cur - 1).clamp_min(0)
    return p, near


class _Acc:
    """per-minibatch gradient vectors of one variant, accumulated without keeping them: the sum, the sum of squared norms,
    the sum of unit vectors, the norms"""

    def __init__(self):
        self.S = None; self.Q = 0.0; self.U = None; self.norms = []; self.n = 0; self.dec = 0.0

    def add(self, g: torch.Tensor, n_dec: float) -> None:
        nrm = float(g.norm())
        self.S = g.clone() if self.S is None else self.S + g
        u = g / max(nrm, 1e-30)
        self.U = u.clone() if self.U is None else self.U + u
        self.Q += nrm * nrm; self.norms.append(nrm); self.n += 1; self.dec += float(n_dec)

    def mean(self) -> torch.Tensor:
        return self.S / max(self.n, 1)

    def agree(self) -> float:
        """mean cosine between two different minibatches' vectors"""
        n = self.n
        return float((self.U.pow(2).sum() - n) / max(n * (n - 1), 1))

    def b_noise(self) -> Dict[str, float]:
        """McCandlish et al.: |G|^2 and tr(Sigma) from the minibatch size B_s and the whole rollout B_b (plan_rl's probe)"""
        n = self.n
        bs = self.dec / n; bb = self.dec
        gb2 = float(self.mean().pow(2).sum()); gs2 = self.Q / n
        G2 = (bb * gb2 - bs * gs2) / (bb - bs)
        tr = (gs2 - gb2) / (1.0 / bs - 1.0 / bb)
        return {"G2": G2, "tr": tr, "B_noise": (tr / G2) if G2 > 0 else float("inf"), "bs": bs, "bb": bb}


def signal_cos(cos_mean: float, n_a: float, n_b: float, mb_dots, bs: float, bb: float, G2a: float, G2b: float) -> float:
    """the cosine of the two terms' SIGNALS (their expected gradients), the minibatch noise taken out as McCandlish's |G|^2 does:
    E[g_a.g_b] at batch B = G_a.G_b + C_ab / B, so from the rollout mean's dot (B = bb) and the minibatches' mean dot (B = bs)
    G_a.G_b = (bb dot_mean - bs mean(dot_mb)) / (bb - bs), over sqrt(|G_a|^2 |G_b|^2). Noise the two terms share (their
    advantages correlate) inflates the raw cos of the means"""
    d_sig = (bb * cos_mean * n_a * n_b - bs * float(np.mean(mb_dots))) / (bb - bs)
    return d_sig / math.sqrt(G2a * G2b) if G2a > 0 and G2b > 0 else float("nan")


def _flat(gs, params) -> torch.Tensor:
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])


def _norm_adv(a: torch.Tensor, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """the trainer's per-minibatch normalisation (over the decisions made)"""
    mu = (a * m).sum() / n
    return (a - mu) / (((a - mu).pow(2) * m).sum() / n).sqrt().add(1e-8)


def fit_values(critic, F: torch.Tensor, f_last: torch.Tensor, terms: Dict[str, torch.Tensor], D: torch.Tensor,
               FM: torch.Tensor, gamma: float, lam: float, steps: int, rounds: int = 4, mb: int = 8192, lr: float = 1e-3,
               seed: int = 0, log=print):
    """one value function per term: a copy of the critic's MLP on its frozen inputs `F` (N = T B, F_in), TD(lambda) targets
    re-made each round. -> {term: (V (T, B), v_last (B,), explained variance)}"""
    T, B = D.shape
    N = F.shape[0]
    g = torch.Generator(device=F.device); g.manual_seed(seed)
    out = {}
    for name, R in terms.items():
        head = copy.deepcopy(critic.mlp)
        if name != "lap":                                  # a term the total's value says little about: from zero
            with torch.no_grad():
                head[-1].weight.zero_(); head[-1].bias.zero_()
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        per = max(1, steps // rounds)
        for r in range(rounds):
            with torch.no_grad():
                V = torch.cat([head(F[i:i + 65536])[:, 0] for i in range(0, N, 65536)]).view(T, B)
                vl = head(f_last)[:, 0]
                ret = (gae(R, V, vl, D, FM, gamma, lam) + V).reshape(-1)
            for _ in range(per):
                idx = torch.randint(0, N, (mb,), device=F.device, generator=g)
                loss = 0.5 * (head(F[idx])[:, 0] - ret[idx]).pow(2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            V = torch.cat([head(F[i:i + 65536])[:, 0] for i in range(0, N, 65536)]).view(T, B)
            vl = head(f_last)[:, 0]
            ret = gae(R, V, vl, D, FM, gamma, lam) + V
            ev = 1.0 - float((ret - V).var() / ret.var().clamp_min(1e-30))
        log(f"[term-split] value {name}: explained variance {ev:.3f} ({steps} steps)")
        out[name] = (V, vl, ev)
    return out


def measure(*, actor, critic, pol_params, batches, actor_on, lp_scored, bX, bS, bPV, bU, bFM, bA, V_ckpt, v_last_ckpt,
            R_all, R_lap, R_con, D, FM, box_ds, box_live, box_pos, last_inputs, gamma, lam, steps, out_path, info,
            log=print):
    """the whole RQ2 measurement on the rollout just collected; writes `out_path` (.txt) and its .npz"""
    t0 = time.time()
    dev = bS.device
    T, B = D.shape
    N = T * B
    R_oth = R_all - R_lap - R_con
    # --- the critic's frozen inputs and two value functions -------------------------------------------------------------------
    with torch.no_grad():
        F = torch.cat([torch.cat([critic.enc(bX[i:i + 8192]), critic.norm(torch.cat([bS[i:i + 8192], bPV[i:i + 8192]], 1))], 1)
                       for i in range(0, N, 8192)])
        sc_l, s_l, pv_l = last_inputs
        f_last = torch.cat([critic.enc(sc_l), critic.norm(torch.cat([s_l, pv_l], 1))], 1)
        ret_ck = gae(R_all, V_ckpt, v_last_ckpt, D, FM, gamma, lam) + V_ckpt
        ev_ck = 1.0 - float((ret_ck - V_ckpt).var() / ret_ck.var().clamp_min(1e-30))
    vals = fit_values(critic, F, f_last, {"lap": R_lap, "contact": R_con}, D, FM, gamma, lam, steps, log=log)
    del F
    with torch.no_grad():
        A_lap = gae(R_lap, vals["lap"][0], vals["lap"][1], D, FM, gamma, lam).reshape(-1)
        A_con = gae(R_con, vals["contact"][0], vals["contact"][1], D, FM, gamma, lam).reshape(-1)
        passes, near = near_passes(box_ds, box_live, box_pos, D, int(round(NEAR_S / 0.1)))
        near_f = near.reshape(-1).float()
    t_fit = time.time() - t0
    # --- per-minibatch policy gradients -----------------------------------------------------------------------------------
    params = [p for p in pol_params if p.requires_grad]
    names = [n_ for n_, p_ in actor.named_parameters() if p_.requires_grad and any(p_ is q for q in params)]
    sizes = [p.numel() for p in params]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    gru_sl = [slice(int(offs[i]), int(offs[i + 1])) for i, n_ in enumerate(names) if n_.startswith("gru.")]
    VAR = ("lap_n", "con_n", "sum_n", "train_n", "lap_r", "con_r", "sum_r", "lap_n_near", "con_n_near", "sum_n_near",
           "con_r_near", "lap_r_near")
    acc = {k: _Acc() for k in VAR}
    acc_gru = {k: _Acc() for k in ("lap_n", "con_n", "sum_n", "train_n")}
    pair_cos = {"n": [], "r": []}
    pair_dot = {"n": [], "r": []}                          # g_lap . g_contact of each minibatch
    n_mb = 0
    for sel in batches():
        mean, log_std, _, mb, _ = actor_on(sel)
        lp = lp_scored(bU[mb], mean, log_std, None)
        m = bFM[mb]; n = m.sum().clamp_min(1.0)
        a_l, a_c, a_t = A_lap[mb], A_con[mb], bA[mb]
        nr = near_f[mb]
        w = {"lap_n": _norm_adv(a_l, m, n) * m / n, "con_n": _norm_adv(a_c, m, n) * m / n,
             "sum_n": _norm_adv(a_l + a_c, m, n) * m / n, "train_n": _norm_adv(a_t, m, n) * m / n,
             "lap_r": a_l * m / n, "con_r": a_c * m / n}
        for k_ in ("lap_n", "con_n", "sum_n", "lap_r", "con_r"):
            w[k_ + "_near"] = w[k_] * nr
        keys = [k_ for k_ in w if k_ in VAR]
        gs = {}
        for j, k_ in enumerate(keys):
            gs[k_] = _flat(torch.autograd.grad(-(lp * w[k_].detach()).sum(), params, retain_graph=j < len(keys) - 1,
                                               allow_unused=True), params)
        gs["sum_r"] = gs["lap_r"] + gs["con_r"]
        for k_ in VAR:
            acc[k_].add(gs[k_], float(n))
        for k_ in acc_gru:
            acc_gru[k_].add(torch.cat([gs[k_][s_] for s_ in gru_sl]) if gru_sl else gs[k_][:0], float(n))
        cs = lambda a_, b_: float(nn.functional.cosine_similarity(a_, b_, dim=0))
        pair_cos["n"].append(cs(gs["lap_n"], gs["con_n"])); pair_cos["r"].append(cs(gs["lap_r"], gs["con_r"]))
        pair_dot["n"].append(float(gs["lap_n"] @ gs["con_n"])); pair_dot["r"].append(float(gs["lap_r"] @ gs["con_r"]))
        del mean, log_std, lp, gs, w
        n_mb += 1
    t_grad = time.time() - t0 - t_fit
    # --- numbers ------------------------------------------------------------------------------------------------------
    cs = lambda a_, b_: float(nn.functional.cosine_similarity(a_, b_, dim=0))
    M = {k: acc[k].mean() for k in VAR}
    num = {}
    for k in VAR:
        a = acc[k]
        num[k] = {"mean_norm": float(M[k].norm()), "mb_norm": float(np.mean(a.norms)), "agree": a.agree(), **a.b_noise()}
    for k, a in acc_gru.items():
        num["gru_" + k] = {"mean_norm": float(a.mean().norm()), "mb_norm": float(np.mean(a.norms)) if a.norms else 0.0,
                           "agree": a.agree() if a.n > 1 and a.norms[0] > 0 else float("nan"), **(a.b_noise() if a.norms and a.norms[0] > 0 else {})}
    share = {}
    for k in ("lap_n", "con_n", "sum_n", "lap_r", "con_r"):
        full, nr_ = M[k], M[k + "_near"]
        share[k] = {"proj": float((nr_ @ full) / full.pow(2).sum().clamp_min(1e-30)), "norm": float(nr_.norm() / full.norm().clamp_min(1e-30)),
                    "mb_norm": float(np.mean(acc[k + "_near"].norms) / max(np.mean(acc[k].norms), 1e-30))}
    with torch.no_grad():
        fmb = bFM > 0
        dec_near = float(near_f[fmb].mean())
        rl, rc, ro = R_lap.reshape(-1), R_con.reshape(-1), R_oth.reshape(-1)
        stats = {"R_lap_sum": float(rl.sum()), "R_con_sum": float(rc.sum()), "R_other_abs_sum": float(ro.abs().sum()),
                 "R_con_near_share": float((rc * near_f).sum() / rc.sum()) if float(rc.sum()) != 0 else float("nan"),
                 "A_lap_sd": float(A_lap[fmb].std()), "A_con_sd": float(A_con[fmb].std()), "A_train_sd": float(bA[fmb].std()),
                 "A_corr": float(torch.corrcoef(torch.stack([A_lap[fmb], A_con[fmb]]))[0, 1]),
                 "A_con_sd_near": float(A_con[fmb & (near_f > 0)].std()) if bool((fmb & (near_f > 0)).any()) else float("nan"),
                 "A_con_sd_far": float(A_con[fmb & (near_f == 0)].std()),
                 "passes": int(passes.sum()), "dec_near": dec_near,
                 "contact_decisions": int((rc < 0).sum()), "ev_ckpt": ev_ck, "ev_lap": vals["lap"][2], "ev_con": vals["contact"][2]}
    cos_n, cos_r = cs(M["lap_n"], M["con_n"]), cs(M["lap_r"], M["con_r"])
    cos_tr_lap, cos_tr_con = cs(M["train_n"], M["lap_n"]), cs(M["train_n"], M["con_n"])
    sig = {v: signal_cos(c_, num[f"lap_{v}"]["mean_norm"], num[f"con_{v}"]["mean_norm"], pair_dot[v], num[f"lap_{v}"]["bs"],
                         num[f"lap_{v}"]["bb"], num[f"lap_{v}"]["G2"], num[f"con_{v}"]["G2"]) for v, c_ in (("n", cos_n), ("r", cos_r))}
    # --- the report ---------------------------------------------------------------------------------------------------
    f = lambda x: "inf" if not math.isfinite(x) else (f"{x:.3g}" if abs(x) < 1e4 else f"{x:.3e}")
    L = []
    L.append(f"RQ2 보상 항목별 policy gradient 나눔 (plan_rl --term-split) -- {info.get('ckpt')}")
    L.append(f"rollout: 차 {B}대 x {T}결정 = {N}결정 (warm-up rollout {info.get('warm')}번 뒤), gamma {gamma:g} lambda {lam:g}, "
             f"minibatch {n_mb}개 (평균 {acc['lap_n'].dec / n_mb:.0f}결정), actor 가중치 {int(offs[-1])}개 (GRU {sum(s_.stop - s_.start for s_ in gru_sl)}개)")
    L.append(f"보상 합: R_lap {stats['R_lap_sum']:.2f} 바퀴, R_contact {stats['R_con_sum']:.2f} 바퀴 (접촉 있는 결정 {stats['contact_decisions']}개), "
             f"나머지 |합| {stats['R_other_abs_sum']:.3g}")
    L.append(f"value 맞춤 설명 분산: checkpoint critic(전체) {stats['ev_ckpt']:.3f} | V_lap {stats['ev_lap']:.3f} | V_contact {stats['ev_con']:.3f} "
             f"({steps} step, 처음 MLP = checkpoint critic, contact는 출력층 0에서)")
    L.append(f"advantage 표준편차 (raw, 바퀴): A_lap {f(stats['A_lap_sd'])}, A_contact {f(stats['A_con_sd'])} (상자 앞 {NEAR_S:g} s 안 "
             f"{f(stats['A_con_sd_near'])} / 밖 {f(stats['A_con_sd_far'])}), checkpoint critic A {f(stats['A_train_sd'])}; corr(A_lap, A_contact) {stats['A_corr']:+.3f}")
    L.append(f"상자 통과 {stats['passes']}번, 통과 앞 {NEAR_S:g} s 안 결정 {100 * dec_near:.2f} %, 그 안에 든 접촉 비용 {100 * stats['R_con_near_share']:.1f} %")
    L.append("")
    L.append("gradient (actor policy 가중치 전부; _n = minibatch마다 정규화한 advantage(학습과 같음), _r = raw 바퀴 단위)")
    L.append(f"{'항목':<10}{'|평균 G|':>12}{'minibatch |g|':>15}{'일치도':>9}{'B_noise':>12}{'|G|^2':>12}{'tr(Σ)':>12}")
    for k, lab in (("lap_n", "lap"), ("con_n", "contact"), ("sum_n", "합"), ("train_n", "학습(ckpt)"), ("lap_r", "lap raw"),
                   ("con_r", "contact raw"), ("sum_r", "합 raw")):
        v = num[k]
        L.append(f"{lab:<10}{f(v['mean_norm']):>12}{f(v['mb_norm']):>15}{v['agree']:>+9.4f}{f(v['B_noise']):>12}{f(v['G2']):>12}{f(v['tr']):>12}")
    L.append("")
    L.append(f"cos(평균 G_lap, 평균 G_contact): 정규화 {cos_n:+.3f}, raw {cos_r:+.3f} | 같은 minibatch 안 cos 평균: 정규화 "
             f"{np.mean(pair_cos['n']):+.3f}, raw {np.mean(pair_cos['r']):+.3f}")
    L.append(f"잡음 뺀 신호끼리 cos (signal_cos: 두 항의 minibatch 잡음이 같이 움직이는 몫을 뺌): 정규화 {sig['n']:+.3f}, raw {sig['r']:+.3f}")
    L.append(f"학습 gradient(ckpt critic)와: cos(G_train, G_lap) {cos_tr_lap:+.3f}, cos(G_train, G_contact) {cos_tr_con:+.3f}")
    L.append(f"크기 비 |G_contact| / |G_lap| (평균 G): 정규화 {num['con_n']['mean_norm'] / max(num['lap_n']['mean_norm'], 1e-30):.3f}, "
             f"raw {num['con_r']['mean_norm'] / max(num['lap_r']['mean_norm'], 1e-30):.3f}")
    L.append(f"rollout 하나({acc['lap_n'].dec:.0f}결정) 대비 B_noise: lap {num['lap_n']['B_noise'] / acc['lap_n'].dec:.2f}배, "
             f"contact {num['con_n']['B_noise'] / acc['lap_n'].dec:.2f}배, 합 {num['sum_n']['B_noise'] / acc['lap_n'].dec:.2f}배 "
             f"(1 넘으면 rollout 하나 평균도 잡음이 신호보다 큼)")
    L.append("")
    L.append(f"상자 통과 앞 {NEAR_S:g} s 안 결정이 가진 gradient 몫 (결정 몫 {100 * dec_near:.2f} %): 투영 = (G_near·G)/|G|^2, 크기 = |G_near|/|G|")
    for k, lab in (("con_n", "contact"), ("sum_n", "합"), ("lap_n", "lap"), ("con_r", "contact raw"), ("lap_r", "lap raw")):
        s = share[k]
        L.append(f"  {lab:<12} 투영 {100 * s['proj']:6.1f} %  크기 {100 * s['norm']:6.1f} %  minibatch 크기 평균 {100 * s['mb_norm']:6.1f} %")
    L.append("")
    L.append("GRU 가중치만")
    for k, lab in (("lap_n", "lap"), ("con_n", "contact"), ("sum_n", "합"), ("train_n", "학습(ckpt)")):
        v = num["gru_" + k]
        L.append(f"  {lab:<10} |평균 G| {f(v['mean_norm'])}  minibatch |g| {f(v['mb_norm'])}  일치도 {v['agree']:+.4f}  B_noise {f(v.get('B_noise', float('nan')))}")
    L.append("")
    L.append(f"시간: value 맞춤 {t_fit:.0f} s, gradient {t_grad:.0f} s")
    txt = "\n".join(L)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(txt + "\n")
    np.savez(out_path.rsplit(".", 1)[0] + ".npz", **{f"{k}_{q}": np.array(v_) for k, d_ in num.items() for q, v_ in d_.items()},
             **{f"share_{k}_{q}": np.array(v_) for k, d_ in share.items() for q, v_ in d_.items()},
             **{f"norms_{k}": np.array(acc[k].norms) for k in VAR}, pair_cos_n=np.array(pair_cos["n"]),
             pair_cos_r=np.array(pair_cos["r"]), cos_n=np.array(cos_n), cos_r=np.array(cos_r),
             pair_dot_n=np.array(pair_dot["n"]), pair_dot_r=np.array(pair_dot["r"]), signal_cos_n=np.array(sig["n"]),
             signal_cos_r=np.array(sig["r"]),
             **{k: np.array(v_) for k, v_ in stats.items()})
    log(txt)
    return num, share, stats
