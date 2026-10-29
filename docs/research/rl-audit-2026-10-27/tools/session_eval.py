"""대회 형식 평가 (2026-10-29, 사용자: 혼자 세션 + 1대1 세션, 둘 다 고정 위치 장애물로 달리다가 시간이 지나면 치움 -> 치우면 다시
가장 빠른 라인으로).

rlc6 평가 세계(`--like` race JSON: 상자 밀도 1.5 최대 10, 주행선 위 허용, 40 s 에피소드)에서 에피소드마다 고정 배치로 달리다
`--remove-at` 초에 상자를 모두 치움(`state_rl.PropRemoval`, 같은 경주의 두 차 동시에). 혼자(상대 없음)와 1대1 둘 다.
보고: 치우기 전 / 후의 랩 시간(그 구간 안에서 시작하고 끝난 랩), 진행 속도, 강한 충돌/km. 그림: 차 몇 대의 치우기 전 경로(파랑,
그때의 상자 주황)와 치운 뒤 경로(초록)를 겹침 -> 치운 뒤 가장 빠른 선으로 돌아가는지.
사용: python session_eval.py CKPT --like RACE_JSON --out DIR [--remove-at 20] [--envs 32]
CKPT = state_rl 선생 또는 state_student 학생.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl  # noqa: E402
from f1sim.learn.state_rl import ActorCritic, PropRemoval, RunningNorm, StateObs, Unwedge  # noqa: E402
from f1sim.learn.state_student import Student, make_env_like  # noqa: E402


def prop_polys(po, row):
    out = []
    for c in (po.p_sid[row] >= 0).nonzero().flatten().tolist():
        n, d = po.p_n[row, c].cpu().numpy(), po.p_d[row, c].cpu().numpy()
        ok = np.isfinite(d); n, d = n[ok], d[ok]
        o = np.argsort(np.arctan2(n[:, 1], n[:, 0])); n, d = n[o], d[o]
        cs = []
        for i in range(len(n)):
            A = np.array([n[i], n[(i + 1) % len(n)]])
            if abs(np.linalg.det(A)) > 1e-9:
                cs.append(np.linalg.solve(A, np.array([d[i], d[(i + 1) % len(n)]])))
        if len(cs) >= 3:
            cs = np.array(cs); px, py, pyaw = po.p_poses[row, c].cpu().numpy()
            cc, ss = math.cos(pyaw), math.sin(pyaw)
            out.append(cs @ np.array([[cc, ss], [-ss, cc]]) + np.array([px, py]))
    return out


def run(a, ck, solo, device):
    is_teacher = "model" in ck
    n_beams = int(ck["args"].get("beams", 36)) if is_teacher else int(ck["beams"])
    frames = 1 if is_teacher else int(ck["frames"])
    stride = 1 if is_teacher else int(ck["args"].get("stride", 3))
    ov = dict(max_steps=int(a.steps) + 10)
    if float(ck["args"].get("v_min", 0.0) or 0.0):
        ov.update(v_min_policy=float(ck["args"]["v_min"]))
    if solo:
        from f1sim.gym_env import EnvConfig
        ov.update(race_size=1, opponent=EnvConfig().opponent, opponent_slots=None)
    env, cfg, ecfg, md = make_env_like(a.like, device, n_beams, frames, stride, True, envs=int(a.envs), override=ov)
    obs, _ = env.reset(seed=int(md["seed"]))
    lid = env.learner_ids; B = int(lid.numel())
    rem = PropRemoval(env, 1.0, float(a.remove_at), float(a.remove_at), seed=1)
    uw = Unwedge(env, lid, mode=int(a.unwedge)) if int(a.unwedge) else None
    R = int(ck["args"].get("repeat", 2))
    if is_teacher:
        sobs = StateObs(env, scan=bool(ck["args"].get("scan_obs", 0)), corridor=bool(ck["args"].get("corridor_obs", 0)), corridor_cars=bool(ck["args"].get("corridor_cars", 0)), car_channel=bool(ck["args"].get("car_channel", 0)), bev=bool(ck["args"].get("bev", 0)))
        last = torch.zeros(B, 2, device=device)
        o = sobs(last)
        net = ActorCritic(o.shape[1], bev=ck.get("bev")).to(device); net.load_state_dict(ck["model"], strict=False); net.eval()
        nrm = RunningNorm(o.shape[1], device); nrm.load_state_dict(ck["norm"])
    else:
        st = Student(int(ck["frames"]), int(ck["beams"]), int(ck["ego_dim"])).to(device); st.load_state_dict(ck["student"]); st.eval()
        keys = ck.get("ego_keys") or sorted(k for k in obs if k != "scan")
        egov = lambda ob: torch.cat([ob[k][lid].float().reshape(B, -1) for k in keys], 1)
    act_full = torch.zeros(env.B, env.act_dim, device=device)
    dt = float(env.sim.control_dt); t_rm = float(a.remove_at)
    path = np.zeros((int(a.steps), B, 2), np.float32)
    props_before = [prop_polys(env.procedural, int(lid[i])) for i in range(min(B, 4))] if env.procedural is not None else []
    laps = {"전": [], "후": []}; prog = {"전": 0.0, "후": 0.0}; hits = {"전": 0, "후": 0}; secs = {"전": 0.0, "후": 0.0}
    lap_start = torch.zeros(B, device=device)
    touching = torch.zeros(B, dtype=torch.bool, device=device)
    # the race result (10-30, the user: speed is judged solo; a 1 vs 1 is judged by who is ahead): the arc gap to the
    # opponent at the start (signed, + = we lead; we start behind) plus the progress both make
    opp = env.sim.other_idx[lid][:, 0] if (not solo and env.sim.other_idx is not None) else None
    if opp is not None:
        g0 = env.signed_gaps(env.last_result.s, env.sim.tid)[lid][:, 0].neg()           # + = we are ahead
        p_me = torch.zeros(B, device=device); p_op = torch.zeros(B, device=device)
    with torch.no_grad():
        for t in range(int(a.steps)):
            if t % R == 0:
                if is_teacher:
                    act = net.pi(nrm(sobs(last))).clamp(-1, 1); last = act
                else:
                    act = st(obs["scan"][lid].float(), egov(obs))
                act_full[lid] = act
            if uw is not None:
                uw.before(act_full)
            obs, r_, te, trc, info = env.step(act_full)
            if uw is not None:
                uw.after()
            rem.step()
            ph = "전" if t * dt < t_rm else "후"
            path[t] = env.sim.state[lid, :2].cpu().numpy()
            prog[ph] += float(info["progress"][lid].float().sum()); secs[ph] += B * dt
            if opp is not None:
                p_me += info["progress"][lid].float(); p_op += info["progress"][opp].float()
            res = env.last_result
            on, imp = res.contact_onsets[lid], res.contact_impacts[lid]
            hits[ph] += int(((on[:, 0] & (imp[:, 0] >= 1.5)) | (on[:, 1] & (imp[:, 1] >= 1.5))).sum())
            car = info["reward_components"]["car_contact"][lid].float() if "car_contact" in info["reward_components"] else torch.zeros(B, device=device)
            tch = car < 0; hits[ph] += int((tch & ~touching).sum()); touching = tch
            if info["lap_times"].numel():
                ids = info["lap_ids"]; lm = env.learner[ids]
                for j, lt in zip(ids[lm].tolist(), info["lap_times"][lm].tolist()):
                    i = int((lid == j).nonzero()[0]) if (lid == j).any() else None
                    if i is None:
                        continue
                    t_end = (t + 1) * dt; t_beg = t_end - lt
                    if t_end <= t_rm:
                        laps["전"].append(lt)
                    elif t_beg >= t_rm:
                        laps["후"].append(lt)
    out = {}
    if opp is not None:
        lead = (g0 + p_me - p_op).cpu().numpy()
        out["결과"] = {"앞선 비율": float((lead > 0).mean()), "끝 차이 중앙 m": float(np.median(lead)),
                      "출발 차이 중앙 m": float(np.median(g0.cpu().numpy()))}
    for ph in ("전", "후"):
        km = max(prog[ph], 1e-6) / 1000
        out[ph] = {"랩 중앙": float(np.median(laps[ph])) if laps[ph] else None, "랩 수": len(laps[ph]),
                   "진행 m/s": prog[ph] / max(secs[ph], 1e-6), "강한/km": hits[ph] / km}
    return out, path, props_before, env.sim.track.tracks[0], int(t_rm / dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--like", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--remove-at", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--unwedge", type=int, default=1, help="the old tracker's wedge recovery (rlc6 had it)")
    a = ap.parse_args()
    device = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
    name = os.path.basename(os.path.dirname(a.ckpt)) + "/" + os.path.basename(a.ckpt)
    summary = {}
    fig, axs = plt.subplots(2, 4, figsize=(20, 11), dpi=90)
    for row_i, solo in enumerate((True, False)):
        tag = "혼자" if solo else "1대1"
        res, path, props, track, k_rm = run(a, ck, solo, device)
        summary[tag] = res
        if "결과" in res:
            r_ = res["결과"]
            print(f"[대회 형식] {name} {tag} 결과: 뒤에서 출발(중앙 {r_['출발 차이 중앙 m']:+.1f} m) -> {a.steps * 0.025:.0f} s 뒤 상대보다 앞선 "
                  f"경주 {r_['앞선 비율'] * 100:.0f} %, 끝 차이 중앙 {r_['끝 차이 중앙 m']:+.1f} m", flush=True)
        print(f"[대회 형식] {name} {tag}: 치우기 전 랩 {res['전']['랩 중앙']} ({res['전']['랩 수']}개) 진행 {res['전']['진행 m/s']:.2f} "
              f"강한 {res['전']['강한/km']:.2f}/km | 치운 뒤 랩 {res['후']['랩 중앙']} ({res['후']['랩 수']}개) 진행 {res['후']['진행 m/s']:.2f} "
              f"강한 {res['후']['강한/km']:.2f}/km", flush=True)
        for j in range(min(4, path.shape[1])):
            ax = axs[row_i, j]
            occ = track.occupancy; rs = track.resolution; ox, oy = track.origin
            ax.imshow(occ, origin="lower", cmap="Greys", extent=(ox, ox + occ.shape[1] * rs, oy, oy + occ.shape[0] * rs), vmin=0, vmax=1.5)
            if j < len(props):
                for P in props[j]:
                    Q = np.vstack([P, P[:1]]); ax.fill(Q[:, 0], Q[:, 1], color="darkorange", alpha=0.9)
            b = path[40:k_rm, j]; f = path[k_rm + 40:, j]
            ax.plot(b[:, 0], b[:, 1], "-", color="royalblue", lw=1.0, alpha=0.8, label="상자 있을 때")
            ax.plot(f[:, 0], f[:, 1], "-", color="limegreen", lw=1.0, alpha=0.9, label="치운 뒤")
            m = np.isfinite(path[:, j, 0])
            ax.set_xlim(path[m, j, 0].min() - 1, path[m, j, 0].max() + 1); ax.set_ylim(path[m, j, 1].min() - 1, path[m, j, 1].max() + 1)
            ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=7)
            ax.set_title(f"{tag} 차 {j}: 파랑 = 상자 있을 때(주황), 초록 = {a.remove_at:g} s에 치운 뒤", fontsize=8)
    fig.suptitle(f"{name}: 대회 형식(고정 상자 -> 치움) | 혼자 랩 전 {summary['혼자']['전']['랩 중앙']} / 후 {summary['혼자']['후']['랩 중앙']} s, "
                 f"1대1 진행 전 {summary['1대1']['전']['진행 m/s']:.2f} / 후 {summary['1대1']['후']['진행 m/s']:.2f} m/s", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "session.png")); plt.close(fig)
    json.dump(summary, open(os.path.join(a.out, "session.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
