"""정직한 궤적 + MAP의 속도 (11-01): 최소시간 레이싱 라인(7 / 6.5 / 4)을 차 앞 L(= 0.3 s x 속도)만큼 wp4 계획으로 맞춰
(mpc.fit_wp4, 속도는 라인의 웨이포인트 속도) 정책과 똑같은 학습 환경·MAP(PLAN_MAP, 20 Hz 결정)으로 달린다.
랩 시간, 벽 충돌 시작, 라인에서 벗어난 거리, 라인 자체의 이론 랩. 인자: 맵 [초]"""
import sys, types, time
import numpy as np, torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim import mpc
from f1sim.learn import state_rl, common

mp = sys.argv[1]; secs = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
hz = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3                  # the plan's length [s] (state_rl --plan-horizon)
dev = torch.device("cuda")
a = types.SimpleNamespace(action="plan", plan_lookahead=0.235, plan_horizon=hz, repeat=2, tracks=mp, warps=0, envs=4,
                          beams=36, episode_s=secs + 5, stage="solo", seed=1, v_min=0.0, reverse_ext=0.0)
env = state_rl.build_env(a, dev)
_, rls = common.load_tracks([mp], racelines=True, **dict(common.teacher_limits(7.0, 6.5, 4.0), objective="min_time"))
rl = rls[0]
xy = torch.tensor(np.asarray(rl.xy), dtype=torch.float32, device=dev); vr = torch.tensor(np.asarray(rl.v), dtype=torch.float32, device=dev)
seg = torch.linalg.norm(torch.roll(xy, -1, 0) - xy, dim=1); s_rl = torch.cat([torch.zeros(1, device=dev), torch.cumsum(seg, 0)[:-1]]); Ltot = float(seg.sum())
print(f"{mp}: 레이싱 라인 {Ltot:.1f} m, 이론 랩 {float((seg / vr.clamp_min(0.1)).sum()):.2f} s, 최고 {float(vr.max()):.1f} m/s")
inner = env.tracker._inner; spec = inner.spec; v_max = float(inner.v_max)
obs, _ = env.reset(seed=1)
lid = env.learner_ids; B = int(lid.numel())
act_full = torch.zeros(env.B, env.act_dim, device=dev)
M = mpc.WP_FIT_POINTS
k_prev = torch.zeros(B, mpc.WP_K, device=dev)


def line_at(s_q):                                                 # world xy and v on the line at arc lengths s_q (B, m)
    s_q = torch.remainder(s_q, Ltot)
    i = torch.searchsorted(s_rl, s_q.contiguous(), right=True) - 1
    i1 = (i + 1) % len(s_rl)
    w = ((s_q - s_rl[i]) / seg[i]).clamp(0, 1)
    return xy[i] * (1 - w[..., None]) + xy[i1] * w[..., None], vr[i] * (1 - w) + vr[i1] * w


laps, hits, prev_col, lat = [], 0, torch.zeros(B, dtype=torch.bool, device=dev), []
t0 = time.time()
for k in range(int(secs / 0.025)):
    if k % 2 == 0:
        st = env.sim.state[lid]
        p, yaw, v = st[:, :2], st[:, 2], st[:, 3]
        d = torch.cdist(p, xy); j = d.argmin(1); lat.append(d.min(1).values)
        vm = env.last_result.odom[lid, 3] if env.last_result is not None else v
        L = mpc.plan_length(vm, spec)
        fr = torch.arange(1, M + 1, device=dev)[None] / M * L[:, None]
        q, _ = line_at(s_rl[j][:, None] + fr)
        dx, dy = q[..., 0] - p[:, None, 0], q[..., 1] - p[:, None, 1]
        c, s_ = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        tx, ty = c * dx + s_ * dy, -s_ * dx + c * dy
        kap = mpc.fit_wp4(k_prev, tx, ty, L, spec); k_prev = kap
        _, vk = line_at(s_rl[j][:, None] + torch.arange(1, mpc.WP_K + 1, device=dev)[None] / mpc.WP_K * L[:, None])
        act = mpc.encode_wp4(kap, vk, v_max, spec, v_meas=vm)
        act_full[lid] = act
    _, r_, te, trc, info = env.step(act_full)
    col = info["reward_components"]["collision"][lid] < 0
    hits += int((col & ~prev_col).sum()); prev_col = col
    if info["lap_times"].numel():
        laps += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
lat = torch.cat(lat)
print(f"  계획 {hz} s: {B}대 x {secs:.0f} s ({time.time() - t0:.0f} s 걸림): 랩 {len(laps)}개 {np.round(sorted(laps), 2).tolist()[:12]} | "
      f"벽 충돌 시작 {hits}회 | 라인에서 벗어남 평균 {float(lat.mean()):.3f} m, p95 {float(lat.quantile(0.95)):.3f} m")
