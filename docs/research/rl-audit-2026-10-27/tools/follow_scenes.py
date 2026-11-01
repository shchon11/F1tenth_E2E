"""1대1 따라가기 장면 (2026-10-31): 느린 상대 뒤에서 왜 안 앞지르나 — 그림 먼저.

대회 형식 1대1(`--like` 경주 JSON의 세계, 상대 속도 배율 0.6~1.0)에서 상자를 처음부터 치우고(상자 없는 1대1: 치운 뒤 세션과 같음) 정책을 몬다.
매 스텝 두 차의 위치·속도·트랙 옆 위치(중앙선 기준)·트랙 반폭, 내 명령, (혼합 선생이면) 몬 정책을 기록.
따라가기 구간 = 상대가 트랙 따라 0~4 m 앞에 1 s 넘게 계속 있는 구간. 수치: 따라간 시간 비율, 구간 수, 그 구간에서 앞지른 수, 구간 동안 상대 속도 /
내 혼자 속도(같은 위치의 `--solo-ref` 프로파일 없으면 생략). 그림: 가장 긴 구간 6개 — 왼쪽 = 그 구간의 두 차 경로(내 차 속도 색, 상대 보라)와 트랙,
오른쪽 = 시간에 따라 간격(상대가 앞 +), 두 차 속도, 두 차 옆 위치와 트랙 반폭, 내 속도 명령, 몬 정책(경주 1 / 혼자 0).
사용: python follow_scenes.py CKPT --like RACE_JSON --out DIR [--envs 32 --steps 1600]
CKPT 대신 planner:KIND(forzaeth / forzaeth_pred / lane_switch / raceline)면 그 플래너가 내 차를 몬다(11-01: 이 세계에서 앞지르기가
가능한지 — 참 상태(GT) 오라클, MAP 추종기; `--tracker pp`면 순수 추종). 앞지름 수 = 간격 부호가 5 m 안에서 +에서 −로 바뀐 횟수(따라가기 구간 없이 바로 지나간 것 포함).
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
ap.add_argument("--stuck-images", type=int, default=0, help="pictures of the first stretch per race where my car stands (|v| < 0.5 m/s) > 3 s")
ap.add_argument("--tracker", default="map", help="planner:KIND only: the planner's path follower, map or pp")
ap.add_argument("--opp-slot", default="", help="JSON merged into the world's opponent slot (e.g. the school world's slow cars)")
a0 = ap.parse_args()
os.makedirs(a0.out, exist_ok=True)
dev = torch.device("cuda")
planner = a0.ckpt[len("planner:"):] if a0.ckpt.startswith("planner:") else ""
ck = (dict(model=None, args=dict(beams=108, repeat=2)) if planner
      else torch.load(a0.ckpt, map_location=dev, weights_only=False))
is_teacher = "model" in ck
n_beams = int(ck["args"].get("beams", 36)) if is_teacher else int(ck["beams"])
frames = 1 if is_teacher else int(ck["frames"])
stride = 1 if is_teacher else int(ck["args"].get("stride", 3))
ov = dict(max_steps=int(a0.steps) + 10)
if float(ck["args"].get("reverse_ext", 0.0) or 0.0):
    ov.update(reverse_ext=float(ck["args"]["reverse_ext"]))
if a0.opp_slot:
    from f1sim import opponent_slots as _osl
    _slot = dict(json.load(open(a0.like))["nominal"]["metadata"]["env_config"]["opponent_slots"][0])
    _slot.update(json.loads(a0.opp_slot))
    ov.update(opponent_slots=_osl.parse_slots(json.dumps([_slot])))
env, cfg, ecfg, md = make_env_like(a0.like, dev, n_beams, frames, stride, True, envs=int(a0.envs), override=ov)
env.sim.measure_impacts = True                                  # car impacts (as crash_scenes / evalx)
obs, _ = env.reset(seed=int(md["seed"]))
lid = env.learner_ids; B = int(lid.numel())
rem = PropRemoval(env, 1.0, 0.0, 0.0, seed=1)                  # no props: the 1 vs 1 after removal
R = int(ck["args"].get("repeat", 2))
if planner:
    from f1sim.learn import common as _common
    mc = env._make_call
    drv = _common.build_ego_driver(env, planner, rls=mc["rls"], limits=mc["teacher_limits"], estimated=False,
                                   stack_params=dict(tracker=a0.tracker))
    env.set_ego_driver(drv)
    tp = lambda last_: torch.zeros(B, 2, device=dev)
elif is_teacher:
    tp = state_rl.TeacherPolicy(env, ck, dev)
else:
    from f1sim.learn.lidar_grid import StudentRunner
    st_run = StudentRunner(env, lid, ck, dev)
    keys = ck.get("ego_keys") or sorted(k for k in obs if k != "scan")
    egov = lambda ob: torch.cat([ob[k][lid].float().reshape(B, -1) for k in keys], 1)
opp = env.sim.other_idx[lid][:, 0]
act = torch.zeros(env.B, env.act_dim, device=dev); last = torch.zeros(B, 2, device=dev)
N = int(a0.steps)
rec = {k: np.zeros((N, B), np.float32) for k in ("gap", "v", "vo", "lat", "lato", "hw", "cmd", "gate", "x", "y", "xo", "yo", "contact", "steer", "yaw")}
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
                if getattr(tp, "mix", None) is not None:
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
        rec["steer"][t] = np.degrees((a_[:, 0].clamp(-1, 1) * float(env.s_max)).cpu().numpy()); rec["yaw"][t] = st[lid, 2].cpu().numpy()
        rec["x"][t] = st[lid, 0].cpu().numpy(); rec["y"][t] = st[lid, 1].cpu().numpy()
        rec["xo"][t] = st[opp, 0].cpu().numpy(); rec["yo"][t] = st[opp, 1].cpu().numpy()
        # a hard car hit: a contact ONSET with the simulator's impact >= 1.5 m/s, as crash_scenes --all-car and evalx count
        # (11-01: this read the car_contact reward before, which the eval world prices at 0 -- every table's "접촉 0" was that)
        cc = res.car_collision[lid] if res.car_collision is not None else torch.zeros(B, dtype=torch.bool, device=dev)
        onset = cc & ~touching; touching = cc.clone()
        imp_c = res.car_impact[lid] if res.car_impact is not None else torch.zeros(B, device=dev)
        rec["contact"][t] = (onset & (imp_c >= 1.5)).float().cpu().numpy()
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
# every overtake / being overtaken: the gap changes sign within 5 m (a sign change at half a lap is the wrap, not a pass)
gp = rec["gap"][40:]
cross = (np.abs(gp[1:]) < 5.0) & (np.abs(gp[:-1]) < 5.0)
n_over = int(((gp[:-1] > 0) & (gp[1:] <= 0) & cross).sum()); n_under = int(((gp[:-1] <= 0) & (gp[1:] > 0) & cross).sum())
km = float(np.abs(rec["v"]).sum() * float(env.sim.control_dt)) / 1000.0
print(f"[앞지름] 전체: 내가 앞지름 {n_over}회, 앞지름 당함 {n_under}회 | 차 강한 충돌 {int(rec['contact'].sum())}회 / {km:.1f} km = "
      f"{rec['contact'].sum() / max(km, 1e-6):.2f}/km | 평균 속도 {rec['v'].mean():.2f} m/s", flush=True)
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
json.dump(dict(follow_share=total_follow / (B * N * dt), segments=segs, lead_end=lead_end, bands=bands, overtakes=n_over,
               overtaken=n_under, car_hits=float(rec["contact"].sum()), km=km),
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
# where my car stands: the first stretch per race (from 1 s on) with |v| < 0.5 m/s for more than 3 s -- the place, the pose,
# the opponent, and what the policy commanded there
w3 = int(3.0 / dt); n_stuck = 0; t_stuck = 0.0
for i in range(B):
    sl_ = np.abs(rec["v"][40:, i]) < 0.5
    k = 0
    while k < len(sl_):
        if sl_[k]:
            k1 = k
            while k1 < len(sl_) and sl_[k1]:
                k1 += 1
            if k1 - k > w3:
                n_stuck += 1; t_stuck += (k1 - k) * dt
            k = k1
        else:
            k += 1
print(f"[서 있음] 전체: 3 s 넘게 선 구간 {n_stuck}개, 합 {t_stuck:.1f} s / {B}경기 x {N * dt:.0f} s", flush=True)
n_st = 0
for i in range(B):
    if n_st >= int(a0.stuck_images):
        break
    slow = np.abs(rec["v"][:, i]) < 0.5
    t = 40
    while t < N and not (slow[t] and slow[t:t + int(3.0 / dt)].all() and t + int(3.0 / dt) <= N):
        t += 1
    if t >= N:
        continue
    t1 = t
    while t1 < N and slow[t1]:
        t1 += 1
    t0 = max(0, t - int(2.0 / dt))
    fig = plt.figure(figsize=(17, 8), dpi=80)
    ax = fig.add_subplot(1, 2, 1)
    occ = track.occupancy; rs = track.resolution; ox, oy = track.origin
    ax.imshow(occ, origin="lower", cmap="Greys", extent=(ox, ox + occ.shape[1] * rs, oy, oy + occ.shape[0] * rs), vmin=0, vmax=1.5)
    ax.plot(rec["x"][t0:t1, i], rec["y"][t0:t1, i], "-", color="tab:blue", lw=1.0)
    for tq, col in ((t, "tab:blue"), (min(t1, N - 1), "red")):
        x_, y_, h_ = rec["x"][tq, i], rec["y"][tq, i], rec["yaw"][tq, i]
        c, s_ = np.cos(h_), np.sin(h_)
        box = np.array([[0.32, 0.15], [0.32, -0.15], [-0.22, -0.15], [-0.22, 0.15], [0.32, 0.15]])
        ax.plot(x_ + box[:, 0] * c - box[:, 1] * s_, y_ + box[:, 0] * s_ + box[:, 1] * c, color=col, lw=1.5)
    ax.plot(rec["xo"][t0:t1, i], rec["yo"][t0:t1, i], "-", color="purple", lw=0.8)
    ax.plot(rec["xo"][t, i], rec["yo"][t, i], "s", color="purple", ms=6)
    cx, cy = rec["x"][t, i], rec["y"][t, i]
    ax.set_xlim(cx - 3, cx + 3); ax.set_ylim(cy - 3, cy + 3); ax.set_aspect("equal")
    ax.set_title(f"경기 {i}: {t * dt:.1f}~{t1 * dt:.1f} s 서 있음 ({(t1 - t) * dt:.1f} s) | 파랑 = 서기 시작, 빨강 = 끝, 보라 = 상대", fontsize=9)
    tt = np.arange(t0, min(t1 + int(1.0 / dt), N)) * dt; sl = slice(t0, min(t1 + int(1.0 / dt), N))
    for r, (keys_, lab) in enumerate(((("v", "cmd"), "속도 m/s: 나 / 내 명령"), (("steer",), "조향 명령 도 (왼쪽 +)"),
                                      (("lat", "hw"), "옆 위치 m (왼쪽 +) / 트랙 반폭 ±"), (("gate", "contact"), "경주 정책이 몲 / 차 접촉"))):
        axr = fig.add_subplot(4, 2, 2 * r + 2)
        for kk, col in zip(keys_, ("tab:blue", "tab:gray")):
            y = rec[kk][sl, i]
            axr.plot(tt, y, color=col, lw=1.2, label=kk)
            if kk == "hw":
                axr.plot(tt, -y, color=col, lw=1.2)
        axr.axvspan(t * dt, t1 * dt, color="red", alpha=0.08)
        axr.axhline(0, color="k", lw=0.4); axr.set_ylabel(lab, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a0.out, f"stuck_{n_st:02d}.png")); plt.close(fig)
    print(f"[서 있음] 경기 {i}: {t * dt:.1f}~{t1 * dt:.1f} s, 자리 ({cx:.2f}, {cy:.2f}), 명령 속도 중앙 {np.median(rec['cmd'][t:t1, i]):.2f} m/s, "
          f"조향 명령 중앙 {np.median(rec['steer'][t:t1, i]):+.0f}도 (범위 {rec['steer'][t:t1, i].min():+.0f}~{rec['steer'][t:t1, i].max():+.0f})", flush=True)
    n_st += 1
print("saved", a0.out)
