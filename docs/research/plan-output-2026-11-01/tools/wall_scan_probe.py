"""벽에 코를 댄 차의 스캔이 무엇을 읽나 (11-12: 벽에 서 버린 차들의 앞 스캔이 0.02 m -> 5 m로 바뀌고 그대로임 -- 벽이 스캔에서 사라짐).
학습 세계(상자·상대 없음)의 학습 차를 정해 둔 명령(처음 1.2 s 조향 한쪽 끝, 속도 3 m/s 유지)으로 벽에 몰아넣고, 제어 스텝(25 ms)마다
속도·차체와 벽 사이 간격·차체 기울기(roll, pitch)·LiDAR 원점 높이·앞 ±10도 스캔(최소/중앙)·그 빔들이 닿은 것의 종류를 적음.
인자: [환경 수, 기본 32] [초, 기본 6] [속도 명령 m/s, 기본 3.0]"""
import os, sys
os.environ["F1SIM_NO_STAGGER"] = "1"
import math
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim import dynamics as dyn

n_env = int(sys.argv[1]) if len(sys.argv) > 1 else 32
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
v_cmd = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
dev = torch.device("cuda")
env, lim, world = PR.build(PR.WORLD, n_env, 4242, dev, fmt="v2", overrides={"episode_s": secs + 5.0, "opp_present": 0.0, "prop_fraction": 0.0})
lid = env.learner_ids; Bl = int(lid.numel())
sim = env.sim
act0 = torch.zeros(env.B, env.act_dim, device=dev)
nb = sim.lidar.n
kf = int(round(math.radians(10.0) / sim.lidar.angle_increment))
steer = torch.where(torch.arange(Bl, device=dev) % 2 == 0, 0.35, -0.35)
T = int(secs / 0.025)
rec = {k: [] for k in ("v", "clr", "roll", "pitch", "oz", "fmin", "fmed", "ftyp", "x", "y", "yaw")}
for t in range(T):
    env.sim.ideal_vcap[lid] = 1e3
    env.ideal_cmd[lid, 0] = steer if t * 0.025 < 1.2 else torch.zeros_like(steer)
    env.ideal_cmd[lid, 1] = v_cmd
    env.step(act0)
    r = env.last_result
    st = sim.state[lid]
    P = sim.P
    roll, pitch = sim.att[lid, 0], sim.att[lid, 2]
    phi = roll + P["mount_roll"][lid]; th = pitch + P["mount_pitch"][lid]
    oz = -P["mount_x"][lid] * torch.sin(th) + (P["mount_y"][lid] * torch.sin(phi) + P["mount_z"][lid] * torch.cos(phi)) * torch.cos(th)
    sc = r.scan[lid].float(); ty = r.scan_type[lid]
    f = sc[:, nb // 2 - kf: nb // 2 + kf + 1]; ft = ty[:, nb // 2 - kf: nb // 2 + kf + 1]
    rec["v"].append(st[:, dyn.IVX].cpu()); rec["clr"].append(sim._footprint_clearance(sim.state)[lid].cpu())
    rec["roll"].append(roll.cpu()); rec["pitch"].append(pitch.cpu()); rec["oz"].append(oz.cpu())
    rec["fmin"].append(f.min(1).values.cpu()); rec["fmed"].append(f.median(1).values.cpu())
    rec["ftyp"].append(torch.stack([(ft == k).float().mean(1) for k in range(5)], 1).cpu())
    rec["x"].append(st[:, 0].cpu()); rec["y"].append(st[:, 1].cpu()); rec["yaw"].append(st[:, 2].cpu())
D = {k: torch.stack(v).numpy() for k, v in rec.items()}
dh = (sim.track.t_duct_h[sim.tid[lid]] * (sim.P["duct_scale"][lid] if "duct_scale" in sim.P else 1.0)).cpu().numpy()
print(f"학습 차 {Bl}대, {secs:.0f} s, 속도 명령 {v_cmd} m/s | 덕트 높이 {dh.min():.2f}-{dh.max():.2f} m, LiDAR 장착 높이 {float(sim.P['mount_z'][lid].mean()):.3f} m, "
      f"앞으로 {float(sim.P['mount_x'][lid].mean()):.3f} m")
print("빔 종류 번호: 0 = 닿은 것 없음, 1 = 덕트, 2 = 높은 벽, 3 = 바닥, 4 = 차/상자")
stuck = (np.abs(D["v"][-40:]) < 0.5).all(0) & (D["clr"][-40:] < 0.03).all(0)
print(f"끝 1 s 동안 벽에 붙어 선 차: {int(stuck.sum())}대 / {Bl}")
for b in np.nonzero(stuck)[0][:6]:
    t0 = int(np.argmax(D["clr"][:, b] < 0.0))
    print(f"-- 차 {b}: 처음 벽에 닿은 때 {t0 * 0.025:.2f} s (덕트 높이 {dh[b]:.2f} m)")
    print("   t-t0[s]   속도  간격[m]  roll[도] pitch[도] LiDAR높이[m] 앞최소[m] 앞중앙[m]  빔종류(없음/덕트/높은벽/바닥)")
    for t in list(range(max(t0 - 8, 0), min(t0 + 12, T), 2)) + list(range(t0 + 20, T, 20)):
        ft = D["ftyp"][t, b]
        print(f"   {(t - t0) * 0.025:+6.2f}  {D['v'][t, b]:5.2f}  {D['clr'][t, b]:+6.3f}  {math.degrees(D['roll'][t, b]):+6.1f}  {math.degrees(D['pitch'][t, b]):+6.1f}   "
              f"{D['oz'][t, b]:+6.3f}     {D['fmin'][t, b]:6.2f}   {D['fmed'][t, b]:6.2f}    {ft[0]:.2f}/{ft[1]:.2f}/{ft[2]:.2f}/{ft[3]:.2f}")
if stuck.any():
    s = stuck
    print(f"붙어 선 차들의 끝 1 s 평균: pitch {math.degrees(D['pitch'][-40:, s].mean()):+.1f}도, LiDAR 원점 높이 {D['oz'][-40:, s].mean():+.3f} m, "
          f"앞 스캔 최소 중앙값 {np.median(D['fmin'][-40:, s]):.2f} m, 2 m 넘게 읽는 비율 {100 * (D['fmin'][-40:, s] > 2).mean():.0f} %")
