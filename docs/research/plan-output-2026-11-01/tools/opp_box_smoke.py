import os, sys, json, numpy as np, torch
os.environ.setdefault("F1SIM_RACELINE_CACHE_ONLY", "1")
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn.plan_rl import build
W = "/home/shchon11/F1tenth_E2E/f1sim/f1sim/learn/configs/plan_world_korea_2026_competition.json"
dev = torch.device("cuda")
slot = json.loads(json.load(open(W))["opp_slot"]); slot["speed_scale"] = [0.9, 1.1]
for tag, ov in (("기본", {}), ("새 설정", {"opp_box_corridor": 1, "opp_box_keep": 3, "opp_slot": json.dumps(slot)})):
    torch.manual_seed(0)
    env, lim, a = build(W, 128, 7, dev, overrides=ov)
    po = env.procedural; lid = env.learner_ids; oth = env.sim.other_idx[lid][:, 0]
    rl = env.teacher.xy[0].detach().cpu().numpy()
    n_opp, n_solo, d_opp, d_solo, scales = [], [], [], [], []
    for s in range(8):
        env.reset(seed=100 + s)
        eid = env.sim.eid[lid]
        sid = po.p_sid[eid].cpu().numpy(); pose = po.p_poses[eid].cpu().numpy()
        opp = (~env.sim.absent[oth]).cpu().numpy() if getattr(env.sim, "absent_any", False) else np.ones(len(lid), bool)
        sc = env.slot_scale[oth].cpu().numpy()
        for b in range(len(lid)):
            live = sid[b] >= 0
            dd = [np.sqrt(((rl - pose[b, j, :2]) ** 2).sum(1)).min() for j in np.nonzero(live)[0]]
            if opp[b]:
                n_opp.append(live.sum()); d_opp += dd; scales.append(sc[b])
            else:
                n_solo.append(live.sum()); d_solo += dd
    n_opp, n_solo = np.array(n_opp), np.array(n_solo)
    d_opp, d_solo = np.array(d_opp), np.array(d_solo)
    print(f"[{tag}] 상대 차 경주 {len(n_opp)}: 상자 있는 경주 {100 * (n_opp > 0).mean():.0f} %, 상자 수 평균 {n_opp[n_opp > 0].mean() if (n_opp > 0).any() else 0:.1f} 최대 {n_opp.max()} | "
          f"레이싱 라인까지 상자 중심 거리 < 0.30 m 비율 {100 * (d_opp < 0.30).mean() if len(d_opp) else 0:.0f} % (최소 {d_opp.min() if len(d_opp) else float('nan'):.2f} m)")
    print(f"[{tag}] 혼자 경주 {len(n_solo)}: 상자 있는 경주 {100 * (n_solo > 0).mean():.0f} %, 상자 수 평균 {n_solo[n_solo > 0].mean():.1f} 최대 {n_solo.max()} | "
          f"라인까지 < 0.30 m 비율 {100 * (d_solo < 0.30).mean():.0f} %")
    print(f"[{tag}] 상대 차 속도 배율 {np.min(scales):.2f}-{np.max(scales):.2f}")
    del env; torch.cuda.empty_cache()
