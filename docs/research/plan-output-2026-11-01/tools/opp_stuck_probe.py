"""학습 세계의 상대 차가 상자에 걸려 서나 (11-11: u50·u75 학습 세계 영상에서 상대 차가 상자에 박혀 0 m/s로 서 있음 -- 대회의 상대 차는
상자를 피함). 학습 세계(plan_rl.build, 경주의 반은 상자를 10-30 s에 치움), 학습 조건, policy 평균 행동, 고정 시드, 통째 에피소드.
상대 차가 있는 결정마다: 상대 차가 섰나(0.3 m/s 아래), 그때 가장 가까운 상자가 1 m 안인가, 상대 차가 상자에 닿은 횟수, 우리 차가 선 상대
차 3 m 안에 있나. 경주에 상자가 있을 때 / 없을 때로 나눔.
인자: checkpoint [환경 수, 기본 512] [초, 기본 40] [상대 차 설정 JSON(학습 세계의 opp_slot 대신, 예 '{"speed_scale": [0.4, 1.2],
"kind_mix": ["layout"]}', 빈 문자열 = 그대로)] [선 상대 차를 다시 출발시키는 시간 s (0 = 안 함, 기본)]"""
import os, sys
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn import state_rl as SR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_reward import PlanReward
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim import plan_steps as PS

torch.set_grad_enabled(False)
ck_path = sys.argv[1]
n_env = int(sys.argv[2]) if len(sys.argv) > 2 else 512
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
SEED = 4242
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
if (ck.get("args") or {}).get("plan_lat_margin") is not None:
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(ck["args"]["plan_lat_margin"])
OV = {"episode_s": secs + 5.0}
if len(sys.argv) > 4 and sys.argv[4]:
    OV["opp_slot"] = sys.argv[4]                                 # a different opponent set, the rest of the world unchanged
import time
t_b = time.time()
env, lim, world = PR.build(PR.WORLD, n_env, SEED, dev, fmt="v2", overrides=OV)
print(f"세계 만들기 {time.time() - t_b:.0f} s, 상대 차 설정 {getattr(world, 'opp_slot', '')}", flush=True)
lid = env.learner_ids; Bl = int(lid.numel())
actor = load_actor2(ck, dev)
pr = PlanReward(env, float(ck["args"].get("gamma", 0.995)), dev)
obs = PlanObs2(env, lim, pr.goal)
drv = PR.Driver(env, lim, pr, obs)
po = env.procedural
rm_p = float(getattr(world, "removal_p", 0.0))
remover = SR.PropRemoval(env, rm_p, *[float(v) for v in getattr(world, "removal_s", (10.0, 30.0))], seed=SEED) if rm_p > 0 else None
restarter = SR.OppRestart(env, float(sys.argv[5]), PS.DT) if len(sys.argv) > 5 and float(sys.argv[5]) > 0 else None
if env.M < 2:
    raise SystemExit("이 세계에는 상대 차 자리가 없음")
others = env.sim.other_idx[lid][:, 0]
eid = env.sim.eid[lid]
absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)
i_op = CC.index("onsets_prop")
acc = {k: torch.zeros(6, device=dev) for k in ("box", "nobox")}   # decisions, stopped, stopped near a box, stopped > 2 s, ours near a stopped one, opp-box contacts
stop_run = torch.zeros(Bl, device=dev)
# by kind and by speed scale, races with boxes only (11-11: the four kinds are all meant to see boxes)
KN = list(getattr(env, "alt_teacher_kinds", ())) + ["raceline"]
kacc = torch.zeros(len(KN), 3, device=dev)                         # opponent decisions, stopped, box contacts
SB = (0.6, 0.8, 1.0)
sacc = torch.zeros(len(SB) + 1, 3, device=dev)
# the scripted / reactive sideways offset is ADDED to the planner's own (opponent_planner.plan_action) and clamped against the
# walls only: is an opponent with an offset on the one that hits the boxes?
eacc = torch.zeros(2, 3, device=dev)                               # offset off / on: decisions, stopped, box contacts
# the planners' own state (spliner / lane-switch: RACING 0, TRAILING 1, OVERTAKING|CHANGING 2): TRAILING hands the line back to the
# raceline and leaves the speed to the env's follow cap -- which brakes for CARS only (gym_env.follow_cap)
ALT = dict(zip(getattr(env, "alt_teacher_kinds", ()), getattr(env, "alt_teachers", ())))
macc = torch.zeros(len(KN), 3, 3, device=dev)                      # kind x mode x (decisions, stopped, box contacts)
prev_c = env.sim.contact_frame()[others, i_op].float().clone()
t_run = time.time()
for kd in range(int(secs / PS.DT)):
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    out, _, _ = drv.decision(torch.tanh(mean), p)
    if remover is not None:
        remover.step()
    if restarter is not None:
        restarter.step()
    so = env.sim.state[others].float(); st = env.sim.state[lid].float()
    on = ~absent[others]
    live = po.p_sid[eid] >= 0
    has_box = live.any(1)
    d2 = ((po.p_poses[eid, :, :2].float() - so[:, None, :2]) ** 2).sum(2)
    d2 = torch.where(live, d2, torch.full_like(d2, 1e9))
    near_box = d2.min(1).values < 1.0
    stopped = on & (so[:, 3].abs() < 0.3)
    stop_run = torch.where(stopped, stop_run + PS.DT, torch.zeros_like(stop_run))
    ours_near = stopped & ((st[:, :2] - so[:, :2]).norm(dim=1) < 3.0)
    c = env.sim.contact_frame()[others, i_op].float()
    dc = (c - prev_c).clamp(min=0.0); prev_c = c.clone()
    rs = out["restarted"]
    stop_run[rs] = 0.0
    code = env.slot_kind_code[others] if hasattr(env, "slot_kind_code") else torch.full((Bl,), -1, device=dev, dtype=torch.long)
    code = torch.where(code < 0, torch.full_like(code, len(KN) - 1), code)
    scale = env.teacher.speed_scale[others] if torch.is_tensor(getattr(env.teacher, "speed_scale", None)) else torch.ones(Bl, device=dev)
    mb = (on & has_box & ~rs).float()
    row = torch.stack([mb, stopped.float() * mb, dc * mb], 1)
    kacc.index_add_(0, code, row)
    sacc.index_add_(0, torch.bucketize(scale, torch.tensor(SB, device=dev)), row)
    ev = getattr(env, "events", None)
    off = ev.lateral_offset() if ev is not None else None
    ev_on = (off[others].abs() > 0.02).long() if off is not None else torch.zeros(Bl, dtype=torch.long, device=dev)
    eacc.index_add_(0, ev_on, row)
    for j, kname in enumerate(KN):
        t = ALT.get(kname)
        ls = getattr(t, "last_state", None) if t is not None else None
        if not torch.is_tensor(ls):
            continue
        if ls.shape[0] != env.B:                                       # planned for its rows only (`plan_rows`, env._alt_rows)
            rws = getattr(env, "_alt_rows", {}).get(kname)
            if rws is None or rws.numel() != ls.shape[0]:
                continue
            full = torch.zeros(env.B, dtype=torch.long, device=dev); full[rws] = ls.long(); ls = full
        mo = ls[others].long().clamp(0, 2)
        mk = (code == j).float()[:, None]
        macc[j].index_add_(0, mo, row * mk)
    for k, m in (("box", has_box), ("nobox", ~has_box)):
        mm = (on & m & ~rs).float()
        acc[k] += torch.stack([mm.sum(), (stopped.float() * mm).sum(), ((stopped & near_box).float() * mm).sum(),
                               ((stop_run > 2.0).float() * mm).sum(), (ours_near.float() * mm).sum(), (dc * mm).sum()])
name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
print(f"주행 {time.time() - t_run:.0f} s ({int(secs / PS.DT)}결정, 결정당 {(time.time() - t_run) / int(secs / PS.DT) * 1000:.0f} ms, 측정 포함)", flush=True)
print(f"{name}: 학습 세계, 학습 조건, 학습 차 {Bl}대 x {secs:.0f} s, 상대 차가 있는 결정만"
      + (f" | {restarter.after_s:g} s 동안 {restarter.move_m:g} m도 못 간 상대 차 다시 출발 {restarter.count}번" if restarter is not None else ""), flush=True)
for k, lab in (("box", "경주에 상자 있음"), ("nobox", "상자 없음(처음부터 없거나 치운 뒤)")):
    v = acc[k]; n = max(float(v[0]), 1.0); mins = float(v[0]) * PS.DT / 60.0
    print(f"  {lab}: 상대 차 {mins:.0f}대·분 | 상대 차가 서 있음 {float(v[1]) / n * 100:.1f} % (그중 상자 1 m 안 "
          f"{float(v[2]) / max(float(v[1]), 1.0) * 100:.0f} %) | 2 s 넘게 서 있음 {float(v[3]) / n * 100:.1f} % | "
          f"우리 차가 선 상대 차 3 m 안 {float(v[4]) / n * 100:.1f} % | 상대 차가 상자에 닿음 {float(v[5]) / max(mins, 1e-6):.2f}/분", flush=True)

for title, A_, labs in (("종류별", kacc, KN), ("속도 배율별", sacc, [f"{SB[0]:g} 아래"] + [f"{SB[i]:g}~{SB[i + 1]:g}" for i in range(len(SB) - 1)] + [f"{SB[-1]:g} 넘게"])):
    print(f"  [상자 있는 경주, {title}]", flush=True)
    for i, lab in enumerate(labs):
        n = float(A_[i, 0])
        if n < 1:
            continue
        mins = n * PS.DT / 60.0
        print(f"    {lab}: 상대 차 {mins:.0f}대·분 | 서 있음 {float(A_[i, 1]) / n * 100:.1f} % | 상자에 닿음 {float(A_[i, 2]) / mins:.2f}/분", flush=True)

print("  [상자 있는 경주, 이벤트의 옆 이동(2 cm 넘게)이 걸려 있나]", flush=True)
for i, lab in enumerate(("옆 이동 없음", "옆 이동 걸림")):
    n = float(eacc[i, 0])
    if n < 1:
        continue
    mins = n * PS.DT / 60.0
    print(f"    {lab}: 상대 차 {mins:.0f}대·분 ({n / max(float(eacc[:, 0].sum()), 1.0) * 100:.0f} %) | 서 있음 {float(eacc[i, 1]) / n * 100:.1f} % | "
          f"상자에 닿음 {float(eacc[i, 2]) / mins:.2f}/분", flush=True)

print("  [상자 있는 경주, planner 상태별 (0 레이싱 라인 / 1 따라가기 = 비킬 수 없다고 봄 / 2 비켜 가기)]", flush=True)
for j, kname in enumerate(KN):
    tot = float(macc[j, :, 0].sum())
    if tot < 1:
        continue
    parts = []
    for m in range(3):
        n = float(macc[j, m, 0]); mins = n * PS.DT / 60.0
        parts.append(f"{m}: 시간 {n / tot * 100:.0f} %, 서 있음 {float(macc[j, m, 1]) / max(n, 1.0) * 100:.0f} %, 상자 접촉 "
                     f"{float(macc[j, m, 2]) / max(mins, 1e-6):.1f}/분")
    print(f"    {kname}: " + " | ".join(parts), flush=True)
