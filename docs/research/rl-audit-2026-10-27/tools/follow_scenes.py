"""1대1 따라가기 장면 (2026-10-31): 느린 상대 뒤에서 왜 안 앞지르나 — 그림 먼저.

대회 형식 1대1(`--like` 경주 JSON의 세계, 상대 속도 배율 0.6~1.0)에서 상자를 처음부터 치우고(상자 없는 1대1: 치운 뒤 세션과 같음) 정책을 몬다.
매 스텝 두 차의 위치·속도·트랙 옆 위치(중앙선 기준)·트랙 반폭, 내 명령, (혼합 선생이면) 몬 정책을 기록.
따라가기 구간 = 상대가 트랙 따라 0~4 m 앞에 1 s 넘게 계속 있는 구간. 수치: 따라간 시간 비율, 구간 수, 그 구간에서 앞지른 수, 구간 동안 상대 속도 /
내 혼자 속도(같은 위치의 `--solo-ref` 프로파일 없으면 생략). 그림: 가장 긴 구간 6개 — 왼쪽 = 그 구간의 두 차 경로(내 차 속도 색, 상대 보라)와 트랙,
오른쪽 = 시간에 따라 간격(상대가 앞 +), 두 차 속도, 두 차 옆 위치와 트랙 반폭, 내 속도 명령, 몬 정책(경주 1 / 혼자 0).
사용: python follow_scenes.py CKPT --like RACE_JSON --out DIR [--envs 32 --steps 1600]
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
from f1sim.learn.state_rl import PropRemoval  # noqa: E402
from f1sim.learn.state_student import make_env_like  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--like", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--envs", type=int, default=32)
ap.add_argument("--steps", type=int, default=1600)
ap.add_argument("--images", type=int, default=6)
a0 = ap.parse_args()
os.makedirs(a0.out, exist_ok=True)
dev = torch.device("cuda")
ck = torch.load(a0.ckpt, map_location=dev, weights_only=False)
is_teacher = "model" in ck
n_beams = int(ck["args"].get("beams", 36)) if is_teacher else int(ck["beams"])
frames = 1 if is_teacher else int(ck["frames"])
stride = 1 if is_teacher else int(ck["args"].get("stride", 3))
ov = dict(max_steps=int(a0.steps) + 10)
if float(ck["args"].get("reverse_ext", 0.0) or 0.0):
    ov.update(reverse_ext=float(ck["args"]["reverse_ext"]))
env, cfg, ecfg, md = make_env_like(a0.like, dev, n_beams, frames, stride, True, envs=int(a0.envs), override=ov)
obs, _ = env.reset(seed=int(md["seed"]))
lid = env.learner_ids; B = int(lid.numel())
rem = PropRemoval(env, 1.0, 0.0, 0.0, seed=1)                  # no props: the 1 vs 1 after removal
R = int(ck["args"].get("repeat", 2))
if is_teacher:
    tp = state_rl.TeacherPolicy(env, ck, dev)
else:
    from f1sim.learn.lidar_grid import StudentRunner
    st_run = StudentRunner(env, lid, ck, dev)
    keys = ck.get("ego_keys") or sorted(k for k in obs if k != "scan")
    egov = lambda ob: torch.cat([ob[k][lid].float().reshape(B, -1) for k in keys], 1)
opp = env.sim.other_idx[lid][:, 0]
act = torch.zeros(env.B, env.act_dim, device=dev); last = torch.zeros(B, 2, device=dev)
N = int(a0.steps)
rec = {k: np.zeros((N, B), np.float32) for k in ("gap", "v", "vo", "lat", "lato", "hw", "cmd", "gate", "x", "y", "xo", "yo", "contact")}
gate = torch.ones(B, device=dev)
tr = env.sim.track
touching = torch.zeros(B, dtype=torch.bool, device=dev)
with torch.no_grad():
    for t in range(N):
        if not is_teacher:
            st_run.observe()
        if t % R == 0:
            if is_teacher:
                a_ = tp(last)
                if tp.mix is not None:
                    gate = tp.mix.gate.float()
            else:
                a_ = st_run(obs, egov(obs))
            act[lid] = a_; last = a_
        obs, r_, te, trc, info = env.step(act)
        rem.step()
        res = env.last_result; st = env.sim.state
        g = env.signed_gaps(res.s, env.sim.tid)[lid][:, 0]          # + = the opponent is ahead
        xy_c, _ = tr.pose_at_s(res.s[lid], env.sim.tid[lid])
        rec["gap"][t] = g.cpu().numpy()
        rec["v"][t] = st[lid, 3].cpu().numpy(); rec["vo"][t] = st[opp, 3].cpu().numpy()
        rec["lat"][t] = res.lateral[lid].cpu().numpy(); rec["lato"][t] = res.lateral[opp].cpu().numpy()
        rec["hw"][t] = tr.sample_edt(xy_c[:, None], env.sim.tid[lid][:, None])[:, 0].cpu().numpy()
        rec["cmd"][t] = (a_[:, 1].clamp(-1, 1) + 1).cpu().numpy() / 2 * float(env.ecfg.v_max_policy)
        rec["gate"][t] = gate.cpu().numpy()
        rec["x"][t] = st[lid, 0].cpu().numpy(); rec["y"][t] = st[lid, 1].cpu().numpy()
        rec["xo"][t] = st[opp, 0].cpu().numpy(); rec["yo"][t] = st[opp, 1].cpu().numpy()
        car = info["reward_components"]["car_contact"][lid].float() if "car_contact" in info["reward_components"] else torch.zeros(B, device=dev)
        rec["contact"][t] = (car < 0).float().cpu().numpy()
dt = float(env.sim.control_dt)
# following stretches: the opponent 0..4 m ahead along the track for more than 1 s
segs = []
for i in range(B):
    f = (rec["gap"][:, i] > 0) & (rec["gap"][:, i] < 4.0)
    t = 40
    while t < N:
        if f[t]:
            t1 = t
            while t1 < N and f[t1]:
                t1 += 1
            if (t1 - t) * dt > 1.0:
                passed = t1 < N and rec["gap"][t1, i] <= 0 and rec["gap"][min(N - 1, t1 + 20), i] < 0
                segs.append(dict(i=i, t0=t, t1=t1, dur=(t1 - t) * dt, passed=bool(passed),
                                 v=float(rec["v"][t:t1, i].mean()), vo=float(rec["vo"][t:t1, i].mean()),
                                 gate=float(rec["gate"][t:t1, i].mean()), contact=float(rec["contact"][t:t1, i].max())))
            t = t1
        else:
            t += 1
total_follow = sum(s["dur"] for s in segs)
n_pass = sum(s["passed"] for s in segs)
lead_end = float((rec["gap"][-1] < 0).mean())
print(f"[따라가기] {os.path.basename(os.path.dirname(a0.ckpt))}/{os.path.basename(a0.ckpt)} | {B}경기 x {N * dt:.0f} s, 상자 없음 | "
      f"상대 0~4 m 뒤를 1 s 넘게 따라간 시간 {total_follow / (B * N * dt) * 100:.0f} %, 구간 {len(segs)}개, 그중 앞지름 {n_pass}개 | "
      f"끝에 앞선 경주 {lead_end * 100:.0f} %", flush=True)
if segs:
    print(f"[따라가기] 구간 평균: 내 속도 {np.mean([s['v'] for s in segs]):.2f} m/s, 상대 {np.mean([s['vo'] for s in segs]):.2f} m/s, "
          f"길이 중앙 {np.median([s['dur'] for s in segs]):.1f} s, 경주 정책이 몬 비율 {np.mean([s['gate'] for s in segs]) * 100:.0f} %, "
          f"접촉 있던 구간 {np.mean([s['contact'] for s in segs]) * 100:.0f} %", flush=True)
# where the time goes: the opponent's arc gap in bands, with both cars' speeds and who drove (from 2 s on)
g_all = rec["gap"][80:]; bands = {}
for lab, lo, hi in (("상대가 20 m 넘게 앞", 20, 1e9), ("10~20 m 앞", 10, 20), ("4~10 m 앞", 4, 10), ("0~4 m 앞", 0, 4), ("내가 앞", -1e9, 0)):
    m = (g_all > lo) & (g_all <= hi)
    if m.any():
        bands[lab] = dict(share=float(m.mean()), v=float(rec["v"][80:][m].mean()), vo=float(rec["vo"][80:][m].mean()),
                          race_policy=float(rec["gate"][80:][m].mean()))
        print(f"[간격 대역] {lab}: 시간 {m.mean() * 100:.0f} %, 내 속도 {bands[lab]['v']:.2f} / 상대 {bands[lab]['vo']:.2f} m/s, "
              f"경주 정책이 몬 비율 {bands[lab]['race_policy'] * 100:.0f} %", flush=True)
print(f"[간격] 출발 간격 중앙 {np.median(rec['gap'][1]):+.1f} m -> 끝 중앙 {np.median(rec['gap'][-1]):+.1f} m (상대가 앞 +)", flush=True)
json.dump(dict(follow_share=total_follow / (B * N * dt), segments=segs, lead_end=lead_end, bands=bands),
          open(os.path.join(a0.out, "follow.json"), "w"), indent=1)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
track = tr.tracks[0]
# the races over time: gap, both speeds, who drove, and where on the track (lap fraction of my car)
tt_all = np.arange(N) * dt
fig, axs = plt.subplots(6, 1, figsize=(16, 17), dpi=75, sharex=True)
for r, i in enumerate(range(0, min(B, 6))):
    ax = axs[r]
    ax.plot(tt_all, rec["gap"][:, i], color="k", lw=1.2, label="간격 m (상대가 앞 +)")
    ax.plot(tt_all, rec["v"][:, i], color="tab:blue", lw=1.0, label="내 속도")
    ax.plot(tt_all, rec["vo"][:, i], color="purple", lw=1.0, label="상대 속도")
    ax.plot(tt_all, rec["cmd"][:, i], color="tab:blue", lw=0.6, ls=":", label="내 속도 명령")
    ax.fill_between(tt_all, -1, 0, where=rec["gate"][:, i] > 0.5, color="orange", alpha=0.3, label="경주 정책이 몲")
    ax.axhspan(4, 10, color="red", alpha=0.06)
    ax.axhline(0, color="k", lw=0.4); ax.set_ylim(-3, 14)
    ax.set_title(f"경기 {i}: 끝 간격 {rec['gap'][-1, i]:+.1f} m | 빨간 띠 = 상대 4~10 m 앞", fontsize=9)
    if r == 0:
        ax.legend(fontsize=8, ncol=5, loc="upper right")
axs[-1].set_xlabel("시간 s")
fig.tight_layout(); fig.savefig(os.path.join(a0.out, "races.png")); plt.close(fig)
for k, s in enumerate(sorted(segs, key=lambda z: -z["dur"])[:int(a0.images)]):
    i, t0, t1 = s["i"], max(0, s["t0"] - 40), min(N, s["t1"] + 80)
    fig = plt.figure(figsize=(17, 8), dpi=80)
    ax = fig.add_subplot(1, 2, 1)
    occ = track.occupancy; rs = track.resolution; ox, oy = track.origin
    ax.imshow(occ, origin="lower", cmap="Greys", extent=(ox, ox + occ.shape[1] * rs, oy, oy + occ.shape[0] * rs), vmin=0, vmax=1.5)
    sc = ax.scatter(rec["x"][t0:t1:4, i], rec["y"][t0:t1:4, i], c=rec["v"][t0:t1:4, i], cmap="viridis", vmin=0, vmax=9, s=8)
    ax.plot(rec["xo"][t0:t1, i], rec["yo"][t0:t1, i], "-", color="purple", lw=1.2, label="상대")
    ax.plot(rec["x"][t0, i], rec["y"][t0, i], "b^", ms=8); ax.plot(rec["xo"][t0, i], rec["yo"][t0, i], "s", color="purple", ms=8)
    xs = np.r_[rec["x"][t0:t1, i], rec["xo"][t0:t1, i]]; ys = np.r_[rec["y"][t0:t1, i], rec["yo"][t0:t1, i]]
    ax.set_xlim(xs.min() - 2, xs.max() + 2); ax.set_ylim(ys.min() - 2, ys.max() + 2); ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, fraction=0.04, label="내 속도 m/s"); ax.legend(fontsize=8)
    ax.set_title(f"경기 {i}: {s['t0'] * dt:.1f}~{s['t1'] * dt:.1f} s 따라감 {s['dur']:.1f} s, 앞지름 {'예' if s['passed'] else '아니오'} | "
                 f"내 {s['v']:.1f} / 상대 {s['vo']:.1f} m/s (세모 = 시작)", fontsize=9)
    tt = np.arange(t0, t1) * dt
    for r, (keys_, lab) in enumerate(((("gap",), "간격 m (상대가 앞 +)"), (("v", "vo", "cmd"), "속도 m/s: 나 / 상대 / 내 명령"),
                                      (("lat", "lato", "hw"), "옆 위치 m (왼쪽 +): 나 / 상대 / 트랙 반폭 ±"), (("gate", "contact"), "경주 정책이 몲 / 차 접촉"))):
        axr = fig.add_subplot(4, 2, 2 * r + 2)
        for kk, col in zip(keys_, ("tab:blue", "purple", "tab:gray")):
            y = rec[kk][t0:t1, i]
            axr.plot(tt, y, color=col, lw=1.2, label=kk)
            if kk == "hw":
                axr.plot(tt, -y, color=col, lw=1.2)
        axr.axvspan(s["t0"] * dt, s["t1"] * dt, color="orange", alpha=0.12)
        axr.axhline(0, color="k", lw=0.4); axr.set_ylabel(lab, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a0.out, f"follow_{k:02d}.png")); plt.close(fig)
print("saved", a0.out)
