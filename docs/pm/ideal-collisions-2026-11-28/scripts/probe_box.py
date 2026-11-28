"""입력 조작 시험 (학습 없음, 주행은 그대로): 결정마다 정책을 한 번 더 돌려, LiDAR 입력에서 상자를 지운 경우(상자에 먼저 맞은 빔을
그 뒤 벽까지 거리로 바꿈, 3장 쌓기 모두)의 계획을 실제 계획과 비교. 기억(GRU) 상태는 같은 것을 씀(실제 주행의 것), 지운 쪽의 기억
갱신은 버림. 차는 실제 계획대로(이상 실행) 달림 -> exam_record.py 기록은 보통 시험과 같음.
결정마다 남김(PB_OUT npz, (T, B_l)): nb = 가장 새 스캔에서 상자에 먼저 맞은 빔 수(기하), da0 / da10 = 첫 단계 조향·속도 행동(tanh)
차이(실제 - 지움), dsteer = 첫 단계 끝 조향각 차이 [rad], dv1 = 첫 단계 끝 속도 차이 [m/s], dlat5 / dlat10 = 0.5 s / 1.0 s 점의 옆 차이
[m](차 틀, + 왼쪽), dmax = 10점 최대 위치 차이 [m].
exam_record.py를 수정 없이 runpy로 부름. 인자는 exam_record.py 그대로. 환경 변수 PB_OUT."""
import os, sys, runpy, atexit
TOOLS = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools"
sys.path.insert(0, TOOLS)
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
import numpy as np
import torch
import f1sim.learn.plan_policy as PP
from f1sim import plan_steps as PS
from f1sim import props as _props

OUT = os.environ["PB_OUT"]
REC = {k: [] for k in ("nb", "nb_real", "da0", "da10", "dsteer", "dv1", "dlat5", "dlat10", "dmax")}
_FPV = {}


def _fpv(po, dev):
    if "f" not in _FPV:
        fp = [torch.as_tensor(np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint), dtype=torch.float32)
              for sh in po.shapes]
        V = max(int(f.shape[0]) for f in fp)
        F = torch.zeros(len(fp), V, 2, device=dev)
        for k, f in enumerate(fp):
            F[k, :f.shape[0]] = f.to(dev); F[k, f.shape[0]:] = f[-1].to(dev)
        _FPV["f"] = F
    return _FPV["f"]


def erased(obs):
    """(B_l, N) the newest scan with every box removed (normalized), (B_l,) beams that hit a box first"""
    env, lid = obs.env, obs.lid
    dev = env.device
    raw = env.last_result.scan[lid].float()
    po = env.procedural
    if po is None:
        return (raw / obs.range).clamp(0, 1), torch.zeros(raw.shape[0], device=dev)
    mx = float(os.environ.get("F1SIM_LIDAR_MOUNT_X") or 0.297)
    st = env.sim.state[lid].float()
    yaw = st[:, 2] + (env.sim.P["mount_yaw"][lid].float() if "mount_yaw" in env.sim.P else 0.0)
    sx = st[:, 0] + mx * torch.cos(st[:, 2]); sy = st[:, 1] + mx * torch.sin(st[:, 2])
    ang = yaw[:, None] + env.sim.lidar.angles.float()[None]
    dx, dy = torch.cos(ang), torch.sin(ang)
    sid = po.p_sid[lid]; ok = sid >= 0
    pose = po.p_poses[lid].float()
    F = _fpv(po, dev)[sid.clamp_min(0)]
    c_, s_ = torch.cos(pose[..., 2])[..., None], torch.sin(pose[..., 2])[..., None]
    wx = pose[..., 0:1] + c_ * F[..., 0] - s_ * F[..., 1]; wy = pose[..., 1:2] + s_ * F[..., 0] + c_ * F[..., 1]
    ax, ay = wx, wy; bx, by = torch.roll(wx, -1, -1), torch.roll(wy, -1, -1)
    ex, ey = bx - ax, by - ay
    asx = (ax - sx[:, None, None])[:, None]; asy = (ay - sy[:, None, None])[:, None]
    dxx = dx[:, :, None, None]; dyy = dy[:, :, None, None]
    den = dxx * ey[:, None] - dyy * ex[:, None]
    t_ = (asx * ey[:, None] - asy * ex[:, None]) / den
    u_ = (asx * dyy - asy * dxx) / den
    hit = (den.abs() > 1e-9) & (t_ > 0) & (u_ >= 0) & (u_ <= 1) & ok[:, None, :, None]
    rb = torch.where(hit, t_, torch.full_like(t_, float("inf"))).amin((2, 3))            # (B_l, N)
    # walls: sphere tracing on the distance field from the scanner
    tid = env.sim.tid[lid][:, None].expand_as(dx)
    s = torch.zeros_like(dx); done = torch.zeros_like(dx, dtype=torch.bool)
    for _ in range(96):
        p = torch.stack([sx[:, None] + s * dx, sy[:, None] + s * dy], -1)
        e = env.sim.track.sample_edt(p, tid)
        done = done | (e <= 0.006) | (s >= obs.range)
        s = torch.where(done, s, s + torch.clamp(e - 0.003, min=0.006))
    rw = s.clamp(max=obs.range)
    first = (rb < rw - 0.02) & (rb <= obs.range)
    new = torch.where(first, rw, raw)
    obs._nb_real = (first & ((raw - rb).abs() < 0.08)).sum(1).float()     # the simulator's scan did return the box there
    return (new / obs.range).clamp(0.0, 1.0), first.sum(1).float()


_orig_push, _orig_reset, _orig_plan = PP.PlanObs.push_scan, PP.PlanObs.reset_rows, PP.plan_v2


def push_scan(self, rows=None):
    _orig_push(self, rows)
    cf, nb = erased(self)
    new = torch.cat([cf[:, None], self.scans_cf[:, :-1]], 1)
    self.scans_cf = new if rows is None else torch.where(rows[:, None, None], new, self.scans_cf)
    self._nb = nb


def reset_rows(self, rows):
    _orig_reset(self, rows)
    if rows.numel() == 0:
        return
    cf, nb = erased(self)
    if not hasattr(self, "scans_cf"):
        self.scans_cf = self.scans.clone(); self._nb = nb
    self.scans_cf[rows] = cf[rows][:, None, :].expand(-1, self.scans.shape[1], -1)


def plan_v2(actor, obs, lim, sample=False, eps=None, cap=None, intent_fix=None):
    pc = None
    if hasattr(obs, "scans_cf"):
        if getattr(actor, "memory", 0) and getattr(obs, "mem_h", None) is None:
            obs.mem_h = torch.zeros(obs.scans.shape[0], actor.memory, device=obs.scans.device)
        h = getattr(obs, "mem_h", None) if getattr(actor, "memory", 0) else None
        with torch.no_grad():
            m_cf, _, vh_cf = actor(obs.scans_cf, obs.state(), h)
        pc = PS.roll(torch.tanh(m_cf), vh_cf.clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim, cap)
        a_cf = torch.tanh(m_cf)
    out = _orig_plan(actor, obs, lim, sample, eps, cap, intent_fix)
    if pc is not None:
        u, mean, log_std, v_hat, p = out
        a = torch.tanh(mean)
        st = obs.env.sim.state[obs.lid].float()
        c, s_ = torch.cos(st[:, 2])[:, None], torch.sin(st[:, 2])[:, None]
        d = p["pts"] - pc["pts"]                                          # (B_l, 10, 2) world
        dlat = -d[..., 0] * s_ + d[..., 1] * c
        REC["nb"].append(obs._nb.cpu()); REC["nb_real"].append(obs._nb_real.cpu()); REC["da0"].append((a[:, 0] - a_cf[:, 0]).cpu())
        REC["da10"].append((a[:, PS.N_STEP] - a_cf[:, PS.N_STEP]).cpu())
        REC["dsteer"].append((p["cmd_d"][:, -1] - pc["cmd_d"][:, -1]).cpu() if p["cmd_d"].dim() == 2 else (p["cmd_d"] - pc["cmd_d"]).cpu())
        REC["dv1"].append((p["cmd_v"][:, -1] - pc["cmd_v"][:, -1]).cpu() if p["cmd_v"].dim() == 2 else (p["cmd_v"] - pc["cmd_v"]).cpu())
        REC["dlat5"].append(dlat[:, 4].cpu()); REC["dlat10"].append(dlat[:, 9].cpu())
        REC["dmax"].append(d.norm(dim=-1).amax(1).cpu())
    return out


PP.PlanObs.push_scan, PP.PlanObs.reset_rows, PP.plan_v2 = push_scan, reset_rows, plan_v2


@atexit.register
def _save():
    if REC["nb"]:
        np.savez_compressed(OUT, **{k: torch.stack(v).numpy().astype(np.float32) for k, v in REC.items()})
        print(f"[probe] saved {OUT} ({len(REC['nb'])} decisions)", flush=True)


sys.argv = [os.path.join(TOOLS, "exam_record.py")] + sys.argv[1:]
runpy.run_path(os.path.join(TOOLS, "exam_record.py"), run_name="__main__")
