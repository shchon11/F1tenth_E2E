"""상자 선생이 상자를 지나치게 의식하는 게 통로 지도 해상도 때문인가 (11-03, 사용자: "korea 2025 참 선생 혼자+상자로 장애물 있는
맵에서 혼자 돌려봤을 때 너무 지나치게 장애물을 의식해 / occupancy resolution이 낮아서 생기는 문제는 아니야?").
1) 깨끗한 맵에서 같은 선생(혼자 정책이 운전)으로 20 s -> 빠른 길(트랙 s별 가로 위치). 2) 학습 장애물 배치마다, 선생의 결정마다:
 - 통로 지도(선생이 보는 13 x 7)가 앞 8 m 안에서 빠른 길이 지나는 칸을 막힘으로 표시했나,
 - 실제로 상자 가장자리가 그 빠른 길(앞 8 m)에서 몇 m 떨어져 있나(차 반폭 0.16 m를 뺀 여유),
 - 두 정책의 실제 실행 속도 명령: 경로점 속도에 뒤에서부터의 제동 한계를 적용한 첫 경로점(mpc.decode_wp4 -- MAP은 이 값을 씀),
   상자 정책(운전) / 혼자 정책(같은 상태였다면).
여유별로 나눠 "지도만 막힘(실제 여유 >= 0.3 m)"에서 감속이 몰리면 해상도 탓, "둘 다 비었음"에서도 크면 해상도 아닌 다른 원인.
인자: 체크포인트 맵id 배치당초 배치수 시드"""
import os, sys, math
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.learn import state_rl as SR
from f1sim import mpc


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, secs, n_lay, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
REC = {}
_orig = SR.SoloMix.__call__


def _rec(self, a_pol, last):
    out = _orig(self, a_pol, last)
    o = SR.blank_opponents(self.sobs(last.clamp(-1, 1))); o[:, self.prop_cols] = 0.0
    REC.update(a_pol=a_pol[:1].detach().clone(), a_b=self.net.pi(self.nrm(o)).clamp(-1, 1)[:1].detach().clone(),
               gate=bool(self.gate[0]), new=True)
    return out


SR.SoloMix.__call__ = _rec


def session(procedural):
    w = sim_worker.SimWorker(_N(), _N())
    w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
    s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=1, device="cuda", compile=False,
                                        collision_mode="soft", procedural=procedural, episode_s=3600.0, randomize=False), 0)
    return w, s


# ---- 1) the fast line: the solo policy on the clean map
w, s = session(False)
env = s["env"]; tr = env.sim.track; tid0 = env.sim.tid[:1]
S_, LAT = [], []
for k in range(int(20.0 / 0.025)):
    w._step_once(s)
    if k > 120:
        S_.append(float(env.last_result.s[0])); LAT.append(float(env.last_result.lateral[0]))
Ltr = float(tr.length[tid0[0]])
S_ = np.mod(np.array(S_), Ltr); LAT = np.array(LAT)
grid = np.arange(0.0, Ltr, 0.1)
# per 0.1 m of track: the mean lateral of the fast line (several laps)
idx = np.clip((S_ / 0.1).astype(int), 0, len(grid) - 1)
acc = np.zeros(len(grid)); cnt = np.zeros(len(grid))
np.add.at(acc, idx, LAT); np.add.at(cnt, idx, 1)
ok = cnt > 0
fast = np.interp(grid, grid[ok], acc[ok] / cnt[ok], period=Ltr)
print(f"빠른 길: 혼자 정책 20 s, 트랙 {Ltr:.1f} m, 덮은 비율 {ok.mean() * 100:.0f} %", flush=True)
del w, s, env
torch.cuda.empty_cache()

# ---- 2) the props layouts
w, s = session(True)
env = s["env"]; tr = env.sim.track; po = env.procedural; lid = env.learner_ids
inner = env.tracker._inner if hasattr(env.tracker, "_inner") else env.tracker
spec = inner.spec
U = torch.tensor(SR.CORRIDOR_U, device="cuda")
AH = SR.AHEAD
fast_t = torch.tensor(fast, device="cuda", dtype=torch.float32)
D_AHEAD = torch.arange(0.1, 8.0 + 1e-6, 0.2, device="cuda")                       # fast-line samples 0..8 m ahead


def eff_v1(a, v_meas):
    _, _, v = mpc.decode_wp4(a.float(), v_meas, 10.0, env.speed_cap[lid][:1].float(), spec)
    return float(v[0, 0])


def check():
    """(grid marks a fast-line cell blocked within 8 m, true clearance of the fast line to the props within 8 m [m])"""
    s0 = env.last_result.s[lid][:1]
    sk = torch.remainder(s0 + D_AHEAD, Ltr)
    lat = fast_t[torch.clamp((sk / 0.1).long(), 0, len(fast) - 1)]
    xy_c, yaw = tr.pose_at_s(sk, tid0.expand(sk.numel()))
    w_ = tr.sample_edt(xy_c[None], tid0[:, None].expand(1, sk.numel()))[0]
    u = lat / w_.clamp_min(1e-3)
    b = (U[None, :] - u[:, None]).abs().argmin(1) // 2                               # the 7-bin column the line is in
    row = torch.bucketize(D_AHEAD, torch.tensor(AH, device="cuda"))                   # its 13-row segment
    cb = SR.corridor_block(env, lid, env.last_result.s[lid], env.sim.tid[lid]).reshape(-1, len(AH), SR.CORRIDOR_BINS)[0]
    g_block = bool(cb[row, b].any())
    # true geometry: the fast-line points against every live prop's polygon (as geo_risk)
    pxy = torch.stack([xy_c[:, 0] - lat * torch.sin(yaw), xy_c[:, 1] + lat * torch.cos(yaw)], 1)   # (n, 2)
    live = po.p_sid[lid[0]] >= 0
    if not bool(live.any()):
        return g_block, 99.0
    pp = po.p_poses[lid[0]][live]; nrm = po.p_n[lid[0]][live]; off = po.p_d[lid[0]][live]
    dx = pxy[None, :, 0] - pp[:, None, 0]; dy = pxy[None, :, 1] - pp[:, None, 1]
    c, s_ = torch.cos(pp[:, 2])[:, None], torch.sin(pp[:, 2])[:, None]
    lx, ly = dx * c + dy * s_, -dx * s_ + dy * c
    fin = torch.isfinite(off)
    sd = nrm[:, None, :, 0] * lx[..., None] + nrm[:, None, :, 1] * ly[..., None] - torch.where(fin, off, torch.zeros_like(off))[:, None, :]
    sd = torch.where(fin[:, None, :], sd, torch.full_like(sd, -1e9)).amax(-1)         # (C, n) > 0 outside (box: = distance)
    return g_block, float(sd.min()) - 0.16


R = []
for L in range(n_lay):
    if L > 0:
        s["obs"], _ = env.reset()
    for k in range(int(secs / 0.025)):
        REC["new"] = False
        w._step_once(s)
        if not REC["new"] or k < 40:
            continue
        v = env.sim.state[lid[:1], 3].float()
        g_block, clear = check()
        R.append((L, REC["gate"], g_block, clear, eff_v1(REC["a_pol"], v), eff_v1(REC["a_b"], v), float(v[0]),
                  float(env.sim.cmd[0, 1])))
    print(f"배치 {L}: 결정 {len(R)}", flush=True)
A = np.array(R, dtype=float)
gate, gb, cl, vp, vb, vv, cmd = A[:, 1] > 0, A[:, 2] > 0, A[:, 3], A[:, 4], A[:, 5], A[:, 6], A[:, 7]
g = gate
print(f"\n== {run} @ {mp}: {n_lay}배치 x {secs:.0f} s, 상자 정책 운전 결정 {g.sum()} / {len(A)}")
print(f"실행 명령 확인: 운전 정책의 decode 첫 경로점 vs 실제 MAP 속도 명령 차이 중앙 {np.median(np.abs(np.where(g, vp, vb) - cmd)):.2f} m/s")
print(f"상자 정책 운전 중: 실행 속도 명령 {vp[g].mean():.2f} vs 혼자 정책이었다면 {vb[g].mean():.2f} m/s, 실제 속도 {vv[g].mean():.2f}")
cats = [("진짜 막힘 (빠른 길 여유 < 0.1 m)", g & (cl < 0.1)),
        ("빠듯함 (여유 0.1~0.3 m)", g & (cl >= 0.1) & (cl < 0.3)),
        ("지도만 막힘 (여유 >= 0.3 m인데 지도는 빠른 길 칸 막힘)", g & (cl >= 0.3) & gb),
        ("둘 다 비었음 (여유 >= 0.3 m, 지도도 빔)", g & (cl >= 0.3) & ~gb)]
for name, m in cats:
    if m.sum() == 0:
        print(f"  {name}: 0"); continue
    d = vb[m] - vp[m]
    print(f"  {name}: 결정 {m.sum() / g.sum() * 100:4.1f} % | 상자 정책 {vp[m].mean():.2f} vs 혼자였다면 {vb[m].mean():.2f} m/s "
          f"(평균 {d.mean():+.2f} 낮게, 1 m/s 넘게 낮은 결정 {np.mean(d > 1.0) * 100:.0f} %) | 실제 속도 {vv[m].mean():.2f}")
print(f"  (지도가 빠른 길 칸을 막았을 때 실제 여유: 중앙 {np.median(cl[g & gb]):.2f} m, 0.3 m 이상 {np.mean(cl[g & gb] >= 0.3) * 100:.0f} %)")
