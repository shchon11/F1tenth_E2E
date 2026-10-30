"""계획기만의 시험 (2026-10-30): 강화학습 없이 `learn.path_plan.PathPlanner`의 길이 좋은지.

학습 차를 상대 차들이 쓰는 주행선 선생(`teacher.RacelineTeacher`: 지연 보정 순수 추종, 주행선을 옆으로 옮긴 선을 따라감)이 몬다. 옮기는 양 =
계획기가 고른 길의 주행선 offset(선생이 내다보는 거리 앞에서), 속도 = 선생의 주행선 속도와, 계획 길이 막혀 있으면 그 앞에서 4 m/s²로 멈출 수 있는
속도 중 작은 것. 같은 rlc6 평가 세계(`--like` 경주 JSON). 상자·벽 충돌이 적으면 계획 길이 좋다는 뜻(강화학습이 이 길을 따라가게 하는 것은 그다음).
수치: 강한(1.5 m/s 이상) 충돌/km 벽·상자·차, 진행, 랩; 같은 세계의 계획기 없는 주행선 선생(`--no-plan 1`)과 비교.
사용: python plan_teacher_eval.py --like RACE_JSON [--no-plan 1] [--steps 2400]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import common, state_rl  # noqa: E402
from f1sim.learn.path_plan import PathPlanner  # noqa: E402
from f1sim.learn.state_student import make_env_like  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--like", required=True)
ap.add_argument("--steps", type=int, default=2400)
ap.add_argument("--no-plan", type=int, default=0)
a0 = ap.parse_args()
dev = torch.device("cuda")
env, cfg, ecfg, md = make_env_like(a0.like, dev, 36, 1, 1, True)
env.sim.measure_impacts = True
_, rls = common.load_tracks([state_rl.KOREA], racelines=True, **dict(common.teacher_limits(7.0, 6.5, 4.0), objective="min_time"))
env.reset(seed=int(md["seed"]))
lid = env.learner_ids; B = int(lid.numel())
pl = PathPlanner(env, lid, rls)
teacher = env.teacher
act = torch.zeros(env.B, env.act_dim, device=dev)
P_l = {k: (v[lid] if torch.is_tensor(v) and v.shape[:1] == (env.B,) else v) for k, v in env.sim.P.items()}
prog = 0.0; hits = {"벽": 0, "상자": 0, "차": 0}; touching = torch.zeros(B, dtype=torch.bool, device=dev); laps = []
for t in range(int(a0.steps)):
    if t % 2 == 0:
        st = env.sim.state[lid]; tid = env.sim.tid[lid]
        off = None
        if not a0.no_plan:
            pl.update()
            s_car = env.last_result.s[lid]
            ld = (teacher.k_ld * st[:, 3].abs()).clamp(teacher.ld_min, teacher.ld_max)
            s_la = (s_car + ld)[:, None]
            off = (pl.offset(s_la, tid) - pl.rl_offset(s_la, tid[:, None]))[:, 0]
        # the teacher carries per-car state for every row of the env (its grip labels): ask it for all rows, keep ours
        off_all = None
        if off is not None:
            off_all = torch.zeros(env.B, device=dev); off_all[lid] = off
        cmd = teacher(env.sim.state, P=env.sim.P, tid=env.sim.tid, offset=off_all)[lid]
        v = cmd[:, 1]
        if not a0.no_plan:
            v_stop = torch.sqrt(2 * 4.0 * (pl.free - 1.0).clamp_min(0.0))
            v = torch.where(pl.free < 15.0, torch.minimum(v, v_stop), v)
        act[lid] = env.teacher_action_to_normalized(torch.stack([cmd[:, 0], v], 1))
    _, r_, te, trc, info = env.step(act)
    res = env.last_result
    on, imp = res.contact_onsets[lid], res.contact_impacts[lid]
    hits["벽"] += int((on[:, 0] & (imp[:, 0] >= 1.5)).sum()); hits["상자"] += int((on[:, 1] & (imp[:, 1] >= 1.5)).sum())
    if res.car_collision is not None:
        cc = res.car_collision[lid]; ci = res.car_impact[lid] if res.car_impact is not None else torch.zeros(B, device=dev)
        hits["차"] += int((cc & ~touching & (ci >= 1.5)).sum()); touching = cc
    prog += float(info["progress"][lid].float().clamp_min(0).sum())
    if info["lap_times"].numel():
        laps += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
km = prog / 1000.0; secs = int(a0.steps) * B * float(env.sim.control_dt)
tag = "주행선 선생만" if a0.no_plan else "주행선 선생 + 계획기"
print(f"[계획기 시험] {tag} | {a0.like.split('/')[-1]} | {km:.1f} km, 진행 {prog / secs:.2f} m/s, 랩 중앙 "
      f"{np.median(laps) if laps else float('nan'):.2f} s ({len(laps)}개) | 강한 충돌/km 벽 {hits['벽'] / km:.2f} 상자 {hits['상자'] / km:.2f} "
      f"차 {hits['차'] / km:.2f} (합 {sum(hits.values()) / km:.2f})", flush=True)
