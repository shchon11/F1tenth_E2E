"""(11-24, 경주 단계) 상대 차 예측 머리(plan_rl --pred-opp)를 붙인 fork가 update 0에서 부모와 똑같이 달리나.

fork를 만드는 길(plan_rl.init_actor: trunk 첫 층에 예측 입력 열을 0으로 붙이고, 예측 머리의 스캔 encoder는 policy 것을 복사) 그대로
부모 checkpoint에서 fork actor를 만들고, 경주 세계(상대 차 = 복사본, 배율 여러 가지 -> 예측 값이 실제로 움직이는 상황)에서 부모가 모는
대로 달리며 결정마다 같은 입력(스캔 3장, 상태, 기억)을 두 actor에 넣어 출력(plan 평균 20개, 표준편차, 속도 짐작, 새 기억 상태)을 비교.
인자: checkpoint [경주 수 8] [결정 수 60] [장치 cuda] [smooth | clear]
smooth (11-25, plan_rl --pred-smooth): 부모가 옛 예측 머리를 가진 checkpoint일 때, 매끈한 머리 fork(encoder·'있음' 갈래 그대로, 위치 출력
  새로, trunk가 예측 점을 읽는 20칸 0)를 같은 길로 만들어 비교. 부모의 plan이 자기 예측 점을 읽었다면(0이 아닌 가중치) fork는 그만큼
  다르게 몲 -> 그 차이를 plan 10점 거리(cm)로 재고, '점 20칸만 0으로 한 부모'와 fork가 같은지(부동소수 반올림까지)도 따로 잼.
clear (11-25, plan_rl --pred-clear): 부모(예측 머리 있는 checkpoint)에 plan-예측 간격 입력 10칸(trunk 끝, 0에서 시작)을 붙인 fork를
  plan_rl.init_actor로 만들어 같은 입력에 비교 -- 간격 값이 실제로 움직이는 동안(상대 차가 예측되는 결정) 출력이 부동소수 반올림까지
  같은지, 그리고 간격 입력의 분포(1 s 안 가장 좁은 값, '멀리' 3 m 아닌 결정 몫)를 찍음.
"""
import os, sys, json
os.environ["F1SIM_NO_STAGGER"] = "1"
import torch
sys.path.insert(0, os.environ.get("F1SIM_SRC", "/home/shchon11/F1tenth_E2E/f1sim"))
torch.set_grad_enabled(False)

ck_path = sys.argv[1]
G = int(sys.argv[2]) if len(sys.argv) > 2 else 8
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
dev = torch.device(sys.argv[4] if len(sys.argv) > 4 else "cuda")
SMOOTH = len(sys.argv) > 5 and sys.argv[5] == "smooth"
CLEAR = len(sys.argv) > 5 and sys.argv[5] == "clear"
ck = torch.load(ck_path, map_location=dev, weights_only=False)
from f1sim.learn.plan_policy import lidar_mount_of, load_actor2, Actor2, PlanObs2, plan_world, PT_IDX, PRED_N, PC_SCALE, PC_FAR
os.environ["F1SIM_LIDAR_MOUNT_X"] = str(lidar_mount_of(ck))
os.environ["F1SIM_PLAN_LAT_MARGIN"] = str((ck.get("args") or {}).get("plan_lat_margin", 0.95))
from f1sim.learn import plan_rl as PR
from f1sim import plan_steps as PS
from f1sim import map_controller as M

W = os.path.join(os.path.dirname(PR.WORLD), "plan_world_korea26scan_race.json")
env, lim, a = PR.build(W, 2 * G, 3, dev, overrides={"opp_present": 1.0, "align_episodes": True})
base = load_actor2(ck, dev)
ca = ck["args"]
kw = {}
if SMOOTH:
    assert getattr(base, "pred_opp", False) and not getattr(base, "pred_smooth", False), "smooth: a checkpoint with the OLD head"
    _lk = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev, margin=1.0)
    kw = dict(pred_smooth=True, pred_kin=(_lk.v_grid, _lk.k_tyre))     # as plan_rl --pred-smooth builds it
if CLEAR:
    assert getattr(base, "pred_opp", False) and not getattr(base, "pred_clear", False), "clear: a checkpoint with the head, no clearances"
    kw = dict(pred_clear=True, pc_geom=(float(env.cfg.vehicle.length), float(env.cfg.vehicle.width), float(ca.get("plan_car_margin", 0.05))))
    if getattr(base, "pred_smooth", False):
        kw.update(pred_smooth=True, pred_kin=(base.pred.kin_v, base.pred.kin_k))
fork = Actor2(int(ck["beams"]), std0=float(ca.get("std0", 0.6)), memory=int(ca.get("memory", 0)), mem_front=bool(ca.get("mem_front")),
              mem_value="v_mem.weight" in ck["actor"], pred_opp=True, **kw).to(dev)
added = PR.init_actor(fork, ck["actor"])
fork.eval()
assert added == ("smooth" if SMOOTH else "" if CLEAR else "new") and fork.pred is not None, added
ref = None
if SMOOTH:
    ref = load_actor2(ck, dev)
    ref.trunk[0].weight[:, -2 * PRED_N:] = 0.0                  # the parent with its point columns zeroed
lid = env.learner_ids
env.ep_limit[:] = 10 ** 6
obs = PlanObs2(env, lim, None)
so = PR.SelfOpponents(env, lim, 1.0, seed=4, max_pool=1, p_newest=1.0, pace=(0.6, 0.95, 0.2))
so.add(base, int(ck.get("update") or 0))
act0 = torch.zeros(env.B, env.act_dim, device=dev)
hb = torch.zeros(lid.numel(), base.memory, device=dev); hf = hb.clone(); hr = hb.clone()
dm = dls = dv = dh = 0.0
rm = rv = rp = 0.0
pv_rng, disp, there = [], [], []
pcs = []                                                        # clear: the fork's clearance inputs [m], (Bl, 10) a decision
ii = PT_IDX.to(dev)
for t in range(N):
    sc, s = obs.scans.clone(), obs.state()
    mb, lsb, vb = base(sc, s, hb); hb1 = base.h_out; qb = base.pred_out
    mf, lsf, vf = (fork(sc, s, hf, prev=obs.prev_poses()) if CLEAR else fork(sc, s, hf)); hf1 = fork.h_out
    if CLEAR:
        pcs.append(fork.pc_out.float() * PC_SCALE)
    dm = max(dm, float((mb - mf).abs().max())); dls = max(dls, float((lsb - lsf).abs().max()))
    dv = max(dv, float((vb - vf).abs().max())); dh = max(dh, float((hb1 - hf1).abs().max()))
    pv_rng.append(torch.sigmoid(fork.pred_out[:, -1]))
    st = env.sim.state[lid].clone()
    p = PS.roll(torch.tanh(mb), vb.clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim)
    if SMOOTH:
        mr, _, vr = ref(sc, s, hr); hr = ref.h_out
        rm = max(rm, float((mr - mf).abs().max())); rv = max(rv, float((vr - vf).abs().max()))
        rp = max(rp, float((ref.pred_out[:, -1] - fork.pred_out[:, -1]).abs().max()))
        pf = PS.roll(torch.tanh(mf), vf.clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim)
        disp.append(torch.stack([p["x"][:, ii] - pf["x"][:, ii], p["y"][:, ii] - pf["y"][:, ii]], -1).norm(dim=-1))   # (Bl, 10)
        there.append(torch.sigmoid(qb[:, -1]) > 0.5)
    hb, hf = hb1, hf1
    x, y, _ = plan_world(p, st)
    so.plan()
    env.sim.ideal_vcap[lid] = 1e3
    rs = torch.zeros(lid.numel(), dtype=torch.bool, device=dev)
    for j in range(PS.SUB):
        env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        so.command(j)
        _, _, te, trc, _ = env.step(act0)
        so.stepped(te | trc); rs |= (te | trc)[lid]
    so.after()
    obs.push_scan(); obs.set_plan(torch.stack([x[:, ii], y[:, ii]], -1), p["pv"]); obs.set_plan2(p)
    if bool(rs.any()):
        r_ = torch.nonzero(rs)[:, 0]; obs.reset_rows(r_); hb[r_] = 0.0; hf[r_] = 0.0; hr[r_] = 0.0
pv = torch.stack(pv_rng)
kind = "매끈한 예측 머리로 바꿈" if SMOOTH else "plan-예측 간격 입력 10칸 붙임" if CLEAR else "예측 머리 붙임"
print(f"[예측 머리 0 초기화 확인] {os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)} -> fork ({kind}), "
      f"경주 {G}개 x {N} 결정, 상대 차 = 복사본(배율 0.6-0.95 / 제 페이스)")
print(f"  부모와 fork의 최대 차이: plan 평균 {dm:.3g} | 표준편차 {dls:.3g} | 속도 짐작 {dv:.3g} m/s | 새 기억 상태 {dh:.3g}")
print(f"  그동안 fork 예측 머리의 '보임' 확률: 최소 {float(pv.min()):.3f} 최대 {float(pv.max()):.3f}")
out = {"mean": dm, "log_std": dls, "v_hat": dv, "mem": dh, "pvis_min": float(pv.min()), "pvis_max": float(pv.max())}
if CLEAR:
    PCs = torch.stack(pcs)                                      # (N, Bl, 10) [m]
    mn = PCs.amin(-1)
    near = mn < PC_FAR - 1e-4
    q_ = lambda x, f: float(torch.quantile(x.float(), f)) if x.numel() else float("nan")
    print(f"  간격 입력: '멀리'(3 m) 아닌 결정 {float(near.float().mean()) * 100:.1f} % | 그때 1 s 안 가장 좁은 값 중앙 {q_(mn[near], 0.5):.2f} m · "
          f"p10 {q_(mn[near], 0.1):.2f} m · 최소 {q_(mn[near], 0.0):.2f} m | 겹침(< 0) 결정 {float((mn < 0).float().mean()) * 100:.1f} %")
    out.update(pc_near_share=float(near.float().mean()), pc_min_p50=q_(mn[near], 0.5), pc_min_p10=q_(mn[near], 0.1),
               pc_overlap_share=float((mn < 0).float().mean()))
if SMOOTH:
    D = torch.stack(disp)                                       # (N, Bl, 10) [m]
    TH = torch.stack(there)
    q = lambda x, f: float(torch.quantile(x.float(), f)) * 100 if x.numel() else float("nan")
    d10, dmx = D[..., -1], D.max(-1).values
    print(f"  '점 20칸만 0으로 한 부모'와 fork: plan 평균 최대 {rm:.3g} | 속도 짐작 {rv:.3g} | '있음' logit {rp:.3g} (부동소수 반올림)")
    for lab, m in (("결정 전부", torch.ones_like(TH)), ("부모 머리 P(있음) > 0.5", TH), ("P(있음) <= 0.5", ~TH)):
        print(f"  부모 plan과 fork plan 10점 거리 [{lab}, {int(m.sum())}개]: 1.0 s 점 중앙 {q(d10[m], 0.5):.2f} · p90 {q(d10[m], 0.9):.2f} · "
              f"최대 {q(d10[m], 1.0):.2f} cm | 10점 중 가장 먼 점 중앙 {q(dmx[m], 0.5):.2f} · p90 {q(dmx[m], 0.9):.2f} cm")
    out.update(ref_mean=rm, ref_v_hat=rv, ref_there_logit=rp, disp10_cm_p50=q(d10, 0.5), disp10_cm_p90=q(d10, 0.9),
               disp10_cm_max=q(d10, 1.0), disp10_there_cm_p50=q(d10[TH], 0.5), disp10_there_cm_p90=q(d10[TH], 0.9),
               disp10_nothere_cm_p90=q(d10[~TH], 0.9), there_share=float(TH.float().mean()))
print(json.dumps(out))
