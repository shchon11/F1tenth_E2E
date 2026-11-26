"""(11-26, 사용자: "그 정보들이 네트워크에 있는지 디버깅용 디코더 ... 출력은 맵, 장애물 배치, ego state와 pose ... 시각화도 가능하게")
시험 기록 하나(exam_record.py + F1SIM_EXAM_SAVE_MEM=1: npz + _mem.npz)로 policy 안쪽 값을 읽는 디코더를 학습하고 지표를 씀.
policy는 얼린 채(기록된 값만 읽음), 디코더만 배움 -- 디버깅 도구, policy로 가는 gradient 없음. 11-19 feature_decode.py를 넓힌 것.
입력 셋: feat(지금 스캔 인코더 출력 + 정규화한 상태 = 기억이 읽는 것) / mem(그 결정 뒤 기억 상태) / feat+mem.
출력(f1sim/learn/state_decoder.py): pose(x, y, 방향), ego(앞 속도, 옆 속도, 회전 속도, 조향), 차 기준 벽 칸·상자 칸(앞 -3..9 m,
옆 ±6 m, 0.25 m 칸 48x48, 지금 안 보이는 것 포함), 상대 차(10 m 안에 있나, 위치·방향·속도, 0.5 s·1.0 s 뒤 위치).
정답(기록에서; policy 입력 아님): 차 상태(x, y, yaw, vx), 옆 속도 vy, 회전 속도·조향 = 기록의 vx, vy에서 ideal 실행의 운동학 자전거로
(sim.py: vy = v tan b, tan b = lr / L tan d, r = v cos b tan d / L; |vx| < 0.5 m/s는 뺌; race33b u975 exam4에서 참 r(_scan.npz의 ego)과
비교해 p99 오차 0, 접촉하는 순간만 어긋남), 벽 = 기록의 벽 거리장 <= 0, 상자 = 칸 중심이 상자 외곽선을 BOX_GROW(0.3 m) 넓힌 안
(0.22 x 0.18 m 상자가 0.25 m 칸 하나에 못 미쳐서), 상대 차 = op / on.
디코더(f1sim/learn/state_decoder.py): 입력 셋마다 출력 묶음(pose+ego / 벽 / 상자 / 상대 차)별 2층 MLP 탑 + 선형 머리, 학습 결정으로
정규화한 입력. 손실 = pose MSE x20 + ego MSE + 벽 BCE + 상자 BCE x10 + 상대 차 BCE + Huber. Adam 1e-3, 코사인, 검증 손실이 가장 낮은 epoch.
정직함: 한 지도라 디코더가 pose에서 지도를 외워 안 보이는 벽을 '되살릴' 수 있음 -> 상자 배치로 나눔. 상자 배치(기록 처음의 상자 자리)
마다 묶어 무작위(고정 시드)로 시험 1/4 · 검증 1/8 · 학습 나머지, 빈 트랙 세션도 같은 비율. 시험 배치는 디코더가 한 번도 못 봄
(같은 은행을 쓰는 exam3·exam4에서 같은 배치가 시험). 검증 = 학습 epoch 고르기만. 지표는 시험 세션, 에피소드 처음 1 s(기억 빔)는 뺌.
  - pose: 위치 오차 중앙 / 90 %, 방향 오차 중앙
  - 벽 IoU: 지금 스캔에 보이는 칸 / 안 보이는 칸, 옆에 기준 '이 자리의 평균 지도'(학습 결정 중 참 pose가 가장 가까운 16개의 평균)
  - 상자 찾음(칸 확률 > 0.5가 그 상자 칸에 하나라도): (a) 지금 보임 (b) 지금 안 보이나 이 세션에서 전에 봄(기억이 들고 있나)
    (c) 안 보이고 아직 본 적 없음(비어 있어야 함: 찾으면 배치를 외웠거나 샘) (d) 치운 상자 자리(전에 봤고 지금은 없음: 비어 있어야 함)
    정밀도 = 0.5 넘는 덩어리 중 참 상자(한 칸 둘레 포함)에 걸친 것, 빈 트랙 세션의 덩어리 수. 기준 = 같은 평균 지도.
  - ego: 평균 절대 오차(옆에 정답의 표준편차)
  - 상대 차(있는 기록만): 10 m 안에 있을 때 찾음 / 없을 때 잘못 찾음, 위치·방향·속도·0.5 s·1.0 s 뒤 위치 오차 -- 지금 보임 / 가려짐,
    3 m 안 / 3-10 m로 나눔. 보임 = 스캐너에서 중심(0.35 m 앞까지) 또는 모서리 하나가 시야각 ±135°·10 m 안, 벽에 안 가림.
출력: <out>_dec.npz (디코더 셋; 콘솔은 torch로, 사이트는 numpy로 돌림), <out>_decin.npz (시험 세션의 feat·mem, 사이트 재생용),
<out>_decode.txt (지표).
--ckpt-copy 경로: 같은 디코더 파일을 checkpoint 옆(plan_uN_dec.npz, 콘솔이 찾는 자리)에도 씀.
대조(기록에 F1SIM_EXAM_SAVE_SCAN의 <기록>_scan.npz가 있거나 --scan 경로): policy가 읽는 가장 새 스캔 하나에 policy와 같은 모양의 스캔
인코더(ScanCtl)를 새로 붙여 같은 정답으로 학습해 같은 표에 한 줄 더 -- 그 크기의 인코더가 지금 스캔에서 꺼낼 수 있는 것 (feat가 그보다
못하면 policy 인코더가 버린 것). 저장은 안 함.
인자: 기록npz [--out 앞머리] [--epochs 60] [--device auto|cpu|cuda] [--ckpt-copy 경로] [--scan 경로]"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(HERE, "state_decoder.py")):          # the lab: a copy beside this file (its f1sim may predate it)
    sys.path.insert(0, HERE)
    import state_decoder as SD
else:
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../../../../f1sim")))
    from f1sim.learn import state_decoder as SD

# the car the exam drives (f1sim.params.VehicleParams; ideal execution = a kinematic bicycle about the CoG)
LF, LR = 0.15875, 0.17145
WB = LF + LR
FOV, RMAX = math.radians(135.0), 10.0
SCAN = "(대조) 원시 스캔"


class ScanCtl(nn.Module):
    """the control's own scan encoder: the policy's ScanEncoder shape (plan_policy.py, one scan as the channel), trained from
    scratch with the towers for the decoding targets -- what an encoder of the policy's size can get out of the newest scan
    (11-26: a plain MLP on the 1081 raw beams generalised so badly -- pose 2.2 m -- that it said nothing about the scan)"""

    def __init__(self, beams, hidden):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 32, 7, stride=3, padding=3), nn.ReLU(), nn.Conv1d(32, 64, 5, stride=3, padding=2), nn.ReLU(),
                                  nn.Conv1d(64, 64, 5, stride=3, padding=2), nn.ReLU(), nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.ReLU())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 1, beams)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n, 256), nn.ReLU())
        self.dec = SD.DecoderNet(256, hidden)

    def forward(self, x):
        return self.dec(self.fc(self.conv(x[:, None, :])))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--mem", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ckpt-copy", default="")
    ap.add_argument("--hidden", type=int, default=512)
    # a control: the same decoder on the raw newest scan (F1SIM_EXAM_SAVE_SCAN's <record>_scan.npz, or this path) -- what a
    # decoder of this size gets out of the scan itself; never saved with the policy's decoders
    ap.add_argument("--scan", default=None)
    return ap.parse_args()


class Track:
    """the record's wall distance field(s), looked up on the device"""

    def __init__(self, z, dev):
        self.edt = torch.as_tensor(z["edt"].astype(np.float32), device=dev)
        self.off = torch.as_tensor(z["t_off"].astype(np.int64), device=dev)
        self.H = torch.as_tensor(z["t_H"].astype(np.int64), device=dev)
        self.W = torch.as_tensor(z["t_W"].astype(np.int64), device=dev)
        self.res = torch.as_tensor(np.asarray(z["t_res"], np.float32).reshape(-1), device=dev)
        self.org = torch.as_tensor(np.asarray(z["t_origin"], np.float32).reshape(-1, 2), device=dev)
        k = 0
        self.ext = [float(self.org[k, 0]), float(self.org[k, 0] + self.W[k] * self.res[k]),
                    float(self.org[k, 1]), float(self.org[k, 1] + self.H[k] * self.res[k])]

    def dist(self, px, py, tid):
        """[m] to the nearest wall at (px, py) (any shape), tid broadcast to it; 0 in a wall or off the grid"""
        tid = tid.expand(px.shape)
        res, ox, oy = self.res[tid], self.org[tid, 0], self.org[tid, 1]
        c = torch.floor((px - ox) / res).long(); r = torch.floor((py - oy) / res).long()
        W, H = self.W[tid], self.H[tid]
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        idx = self.off[tid] + r.clamp(min=0) * W + c.clamp(min=0)
        idx = torch.where(ok, idx, torch.zeros_like(idx))
        return torch.where(ok, self.edt[idx], torch.zeros_like(px))


def los(trk, sx, sy, syaw, px, py, tid, short, ok=None, step=0.1):
    """bool, shape of px: the point (px, py) seen from the scanner (sx, sy, heading syaw; broadcast to px): within RMAX and
    the field of view, no wall on the line up to `short` m before it (plan_rl.boxes_in_view's test)"""
    dx, dy = px - sx, py - sy
    d = torch.sqrt(dx * dx + dy * dy)
    rel = torch.remainder(torch.atan2(dy, dx) - syaw + math.pi, 2.0 * math.pi) - math.pi
    vis = (d <= RMAX) & (rel.abs() <= FOV)
    if ok is not None:
        vis = vis & ok
    n = int(math.ceil(RMAX / step))
    t = torch.arange(1, n + 1, device=px.device, dtype=px.dtype) * step
    ux, uy = (dx / d.clamp_min(1e-6))[..., None], (dy / d.clamp_min(1e-6))[..., None]
    qx = (sx + 0 * px)[..., None] + ux * t; qy = (sy + 0 * py)[..., None] + uy * t
    wall = trk.dist(qx, qy, tid[..., None]) <= 0.0
    wall = wall & (t < (d - short)[..., None])
    return vis & ~wall.any(-1)


def main():
    a = parse()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0); np.random.seed(0)
    if a.device == "auto":
        dev = torch.device("cpu")
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            dev = torch.device("cuda") if free > 2.5e9 else dev
    else:
        dev = torch.device(a.device)
    rec = a.record
    out = a.out or rec[:-4]
    t_start = time.time()
    z = np.load(rec, allow_pickle=False)
    M = np.load(a.mem or rec[:-4] + "_mem.npz")
    feat_all, mem_all = M["feat"], M["mem"]                          # (T, B, F), (T, B, H) float16
    T, B = feat_all.shape[:2]
    st = z["st"][:T].astype(np.float64)
    vy = z["vy"][:T].astype(np.float64)
    rs = z["rs"][:T].astype(bool)
    sid = z["sid"][:T].astype(np.int64)
    pose = z["pose"][:T].astype(np.float64)
    tid = z["tid"][:T].astype(np.int64)
    has_opp = "op" in z.files and bool(z["on"].any())
    op = z["op"].astype(np.float64) if "op" in z.files else np.zeros((T + 1, B, 4))
    on = z["on"].astype(bool) if "on" in z.files else np.zeros((T + 1, B), bool)
    ors = z["ors"][:T].astype(bool) if "ors" in z.files else np.zeros((T, B), bool)
    mount = float(z["lidar_mount_x"]) if "lidar_mount_x" in z.files and float(z["lidar_mount_x"]) > 0 else 0.297
    car = np.asarray(z["car"], float) if "car" in z.files else np.array([0.58, 0.31])
    dt = float(z["dt"])
    name = str(z["name"]); exam = str(z["exam"]) if "exam" in z.files else "?"
    fps = {int(k[3:]): z[k].astype(np.float64) for k in z.files if k.startswith("fp_")}
    trk = Track(z, dev)
    # the decision each row's episode began at
    k0 = np.zeros((T, B), int); cur = np.zeros(B, int)
    for t in range(T):
        k0[t] = cur
        cur = np.where(rs[t], t + 1, cur)

    # ---- held-out box LAYOUTS: the layout = the boxes' places at the first decision; sorted keys, a fixed permutation
    box_s = (sid[0] >= 0).any(-1)
    key = {}
    for b in np.nonzero(box_s)[0]:
        m = sid[0, b] >= 0
        key[b] = tuple(sorted(map(tuple, np.round(pose[0, b, m, :2], 2).tolist())))
    keys = sorted(set(key.values()))
    role = np.zeros(B, int)                                          # 0 train, 1 validation (picks the epoch), 2 test
    def split(n, seed):
        p = np.random.RandomState(seed).permutation(n); r = np.zeros(n, int)
        n_te, n_va = int(round(n / 4)), int(round(n / 8))
        r[p[:n_te]] = 2; r[p[n_te:n_te + n_va]] = 1
        return r
    kr = dict(zip(keys, split(len(keys), 1)))
    for b, k in key.items():
        role[b] = kr[k]
    emp = np.nonzero(~box_s)[0]
    role[emp] = split(len(emp), 2)
    print(f"[{name} · {exam}] 결정 {T} x 세션 {B} | 상자 배치 {len(keys)}개 (시험 {sum(v == 2 for v in kr.values())} · 검증 "
          f"{sum(v == 1 for v in kr.values())}) · 빈 트랙 세션 {len(emp)}개 (시험 {int((role[emp] == 2).sum())}) | 상대 차 "
          f"{'있음' if has_opp else '없음'} | {dev}", flush=True)

    # ---- samples: every decision; flat index i = t * B + b
    tt, bb = np.meshgrid(np.arange(T), np.arange(B), indexing="ij")
    tt, bb = tt.ravel(), bb.ravel()
    rl = role[bb]
    X_, Y_, YAW = st[tt, bb, 0], st[tt, bb, 1], st[tt, bb, 2]
    x0, x1, y0, y1 = trk.ext
    pose_t = np.stack([(X_ - x0) / (x1 - x0), (Y_ - y0) / (y1 - y0), np.cos(YAW), np.sin(YAW)], 1).astype(np.float32)
    # ego: vx, vy recorded; r and steering from ideal execution's kinematics (unknown below 0.5 m/s)
    vx_, vy_ = st[tt, bb, 3], vy[tt, bb]
    okv = np.abs(vx_) > 0.5
    beta = np.where(okv, np.arctan(vy_ / np.where(okv, vx_, 1.0)), 0.0)
    tan_d = np.where(okv, WB / LR * vy_ / np.where(okv, vx_, 1.0), 0.0)
    r_ = vx_ * np.cos(beta) * tan_d / WB
    ego = np.stack([vx_, vy_, r_, np.arctan(tan_d)], 1)
    ego_m = np.stack([np.ones_like(okv), np.ones_like(okv), okv, okv], 1).astype(np.float32)
    ego_scale = np.array([max(ego[:, k][ego_m[:, k] > 0].std(), 1e-3) for k in range(4)], np.float32)
    ego_t = (ego / ego_scale).astype(np.float32)

    # ---- opponent targets (car frame at the decision)
    c_, s_ = np.cos(YAW), np.sin(YAW)
    def to_car(px, py):
        dx, dy = px - X_, py - Y_
        return dx * c_ + dy * s_, -dx * s_ + dy * c_
    ox, oy = to_car(op[tt, bb, 0], op[tt, bb, 1])
    odist = np.hypot(ox, oy)
    there = on[tt, bb] & (odist <= SD.OPP_RANGE)
    ohead = SD.wrap(op[tt, bb, 2] - YAW)
    opp_t = np.zeros((len(tt), 10), np.float32); opp_m = np.zeros((len(tt), 10), np.float32)
    opp_t[:, 0] = there
    opp_t[:, 1], opp_t[:, 2] = ox / SD.OPP_SCALE, oy / SD.OPP_SCALE
    opp_t[:, 3], opp_t[:, 4] = np.cos(ohead), np.sin(ohead)
    opp_t[:, 5] = op[tt, bb, 3] / SD.OPP_SCALE
    opp_m[:, 1:6] = there[:, None]
    # restarts in the window break a future: ours (rs) or the other car's (ors); cumulative counts per row
    brk = np.concatenate([np.zeros((1, B), int), np.cumsum(rs | ors, 0)], 0)          # brk[t] = breaks in [0, t)
    onc = np.concatenate([np.zeros((1, B), int), np.cumsum(~on, 0)], 0)               # absences in [0, t)
    fut_ok = np.zeros((len(tt), 2), bool)
    for j, k in enumerate(SD.OPP_FUT):
        te = np.minimum(tt + k, T)
        ok = (tt + k <= T) & (brk[te, bb] - brk[tt, bb] == 0) & (onc[np.minimum(te + 1, T + 1), bb] - onc[tt, bb] == 0) & there
        fx, fy = to_car(op[te, bb, 0], op[te, bb, 1])
        opp_t[:, 6 + 2 * j], opp_t[:, 7 + 2 * j] = fx / SD.OPP_SCALE, fy / SD.OPP_SCALE
        opp_m[:, 6 + 2 * j: 8 + 2 * j] = ok[:, None]
        fut_ok[:, j] = ok
    opp_t[~np.isfinite(opp_t)] = 0.0

    # ---- grid targets: walls and boxes in the car's frame
    cells = torch.as_tensor(SD.cell_centres(), device=dev)                           # (C, 2)
    NCL = cells.shape[0]
    Vmax = max([len(v) for v in fps.values()] + [4])
    fp_t = torch.zeros(max(fps.keys(), default=0) + 1, Vmax, 2, dtype=torch.float32, device=dev)
    for k, v in fps.items():
        vv = np.concatenate([v, np.repeat(v[-1:], Vmax - len(v), 0)], 0)
        if (v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]).sum() < 0:   # clockwise -> counter-clockwise
            vv = np.concatenate([v[::-1], np.repeat(v[:1], Vmax - len(v), 0)], 0)
        fp_t[k] = torch.as_tensor(vv, dtype=torch.float32, device=dev)
    SL = sid.shape[2]
    sid_f = torch.as_tensor(sid.reshape(T * B, SL), device=dev)
    pose_f = torch.as_tensor(pose.reshape(T * B, SL, 3), dtype=torch.float32, device=dev)
    st_f = torch.as_tensor(np.stack([X_, Y_, YAW], 1), dtype=torch.float32, device=dev)
    tid_f = torch.as_tensor(tid[tt, bb], device=dev)
    grow = SD.BOX_GROW

    def box_cells(i, sids, poses):
        """(n, K, C) bool: the cells of each box's zone (centre within the outline grown by BOX_GROW), K slots. Only the live
        slots are computed (a record keeps 18 slots, ~3 live: all of them made a (n, 18, C, V) tensor of ~1.4 GB a chunk)"""
        n, K = sids.shape
        out = torch.zeros(n, K, cells.shape[0], dtype=torch.bool, device=dev)
        r, k = torch.nonzero(sids >= 0, as_tuple=True)
        if len(r) == 0:
            return out
        q = fp_t[sids[r, k]]                                                         # (M, V, 2) box frame
        pz = poses[r, k]                                                             # (M, 3)
        cb, sb = torch.cos(pz[:, 2])[:, None], torch.sin(pz[:, 2])[:, None]
        wx = pz[:, 0:1] + q[..., 0] * cb - q[..., 1] * sb; wy = pz[:, 1:2] + q[..., 0] * sb + q[..., 1] * cb
        s0 = st_f[i][r]
        c0, s0_ = torch.cos(s0[:, 2])[:, None], torch.sin(s0[:, 2])[:, None]
        dx, dy = wx - s0[:, 0:1], wy - s0[:, 1:2]
        cx, cy = dx * c0 + dy * s0_, -dx * s0_ + dy * c0                              # (M, V) car frame
        ex, ey = torch.roll(cx, -1, -1) - cx, torch.roll(cy, -1, -1) - cy
        ln = torch.sqrt(ex * ex + ey * ey).clamp_min(1e-9)
        nx, ny = ey / ln, -ex / ln                                                   # outward (counter-clockwise outline)
        deg = ln < 1e-6
        dd = (cells[None, :, None, 0] - cx[:, None, :]) * nx[:, None, :] + (cells[None, :, None, 1] - cy[:, None, :]) * ny[:, None, :]
        dd = torch.where(deg[:, None, :], torch.full_like(dd, -1e9), dd)             # (M, C, V)
        out[r, k] = dd.max(-1).values < grow
        return out

    N = len(tt)
    wall_t = torch.zeros(N, NCL, dtype=torch.uint8, device=dev)
    box_t = torch.zeros(N, NCL, dtype=torch.uint8, device=dev)
    CH = 2048
    with torch.no_grad():
        for i0 in range(0, N, CH):
            i = torch.arange(i0, min(i0 + CH, N), device=dev)
            s0 = st_f[i]
            c0, s0_ = torch.cos(s0[:, 2])[:, None], torch.sin(s0[:, 2])[:, None]
            wx = s0[:, 0:1] + cells[None, :, 0] * c0 - cells[None, :, 1] * s0_
            wy = s0[:, 1:2] + cells[None, :, 0] * s0_ + cells[None, :, 1] * c0
            wall_t[i] = (trk.dist(wx, wy, tid_f[i][:, None]) <= 0.0).to(torch.uint8)
            sids = sid_f[i]
            if bool((sids >= 0).any()):
                box_t[i] = box_cells(i, sids, pose_f[i]).any(1).to(torch.uint8)
    print(f"정답 칸: 벽 {wall_t.float().mean() * 100:.0f} % · 상자 {box_t.float().mean() * 100:.2f} % | 상대 차 10 m 안 "
          f"{there.mean() * 100:.1f} % | {time.time() - t_start:.0f} s", flush=True)

    # ---- inputs
    feat_np = feat_all.reshape(T * B, -1); mem_np = mem_all.reshape(T * B, -1)
    scan_p = a.scan or rec[:-4] + "_scan.npz"
    SETS = list(SD.SETS)
    if os.path.exists(scan_p):
        scan_np = np.load(scan_p)["scan"][:T].reshape(T * B, -1)                     # range / range_max, float16
        SETS.append(SCAN)
    F_, H_ = feat_np.shape[1], mem_np.shape[1]
    tr_i = np.nonzero(rl == 0)[0]; va_i = np.nonzero(rl == 1)[0][::2]
    te_i = np.nonzero((rl == 2) & (tt - k0[tt, bb] >= 10))[0]
    pose_g = torch.as_tensor(pose_t, device=dev); ego_g = torch.as_tensor(ego_t, device=dev)
    egom_g = torch.as_tensor(ego_m, device=dev)
    opp_g = torch.as_tensor(opp_t, device=dev); oppm_g = torch.as_tensor(opp_m, device=dev)
    bce = nn.functional.binary_cross_entropy_with_logits

    def loss_of(o, i):
        lp = 20.0 * ((o["pose"] - pose_g[i]) ** 2).mean()
        em = egom_g[i]
        le = (((o["ego"] - ego_g[i]) ** 2) * em).sum() / em.sum().clamp_min(1.0)
        lw = bce(o["wall"], wall_t[i].float())
        lb = 10.0 * bce(o["box"], box_t[i].float())
        om = oppm_g[i]
        hub = nn.functional.smooth_l1_loss(o["opp"][:, 1:], opp_g[i, 1:], reduction="none", beta=0.2)
        lo = bce(o["opp"][:, 0], opp_g[i, 0]) + (hub * om[:, 1:]).sum() / om[:, 1].sum().clamp_min(1.0)
        return lp + le + lw + lb + lo

    nets, norms, preds = {}, {}, {}
    for which in SETS:
        t0 = time.time()
        Xn = (feat_np if which == "feat" else mem_np if which == "mem" else scan_np if which == SCAN
              else np.concatenate([feat_np, mem_np], 1))
        mu = Xn[tr_i].astype(np.float32).mean(0); sd = Xn[tr_i].astype(np.float32).std(0) + 1e-4
        Xg = torch.as_tensor(Xn, device=dev)                                         # float16 on the device
        mu_g, sd_g = torch.as_tensor(mu, device=dev), torch.as_tensor(sd, device=dev)
        net = (ScanCtl(Xn.shape[1], a.hidden) if which == SCAN else SD.DecoderNet(Xn.shape[1], a.hidden)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        tr_g = torch.as_tensor(tr_i, device=dev); va_g = torch.as_tensor(va_i, device=dev)
        best, best_sd, best_ep = float("inf"), None, -1
        for ep in range(a.epochs):
            net.train()
            for idx in tr_g[torch.randperm(len(tr_g), device=dev)].split(1024):
                o = net((Xg[idx].float() - mu_g) / sd_g)
                loss = loss_of(o, idx)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            sch.step()
            if ep % 2 == 1 or ep == a.epochs - 1:
                net.eval(); vl = 0.0
                with torch.no_grad():
                    for idx in va_g.split(4096):
                        vl += float(loss_of(net((Xg[idx].float() - mu_g) / sd_g), idx)) * len(idx)
                vl /= max(len(va_g), 1)
                if vl < best:
                    best, best_ep = vl, ep
                    best_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_sd); net.eval()
        nets[which], norms[which] = net, (mu, sd)
        with torch.no_grad():
            o = []
            for idx in torch.as_tensor(te_i, device=dev).split(4096):
                r_ = net((Xg[idx].float() - mu_g) / sd_g)
                o.append({k: (v.half() if k in ("wall", "box") else v).cpu() for k, v in r_.items()})
            preds[which] = {k: torch.cat([d[k] for d in o], 0) for k in o[0]}
        del Xg
        print(f"[{which}] 학습 {time.time() - t0:.0f} s, 고른 epoch {best_ep + 1}/{a.epochs} (검증 손실 {best:.4f})", flush=True)

    # ---- what is in view at the test decisions: wall cells (ray cast from the scanner) and the boxes / the other car
    te_g = torch.as_tensor(te_i, device=dev)
    n_te = len(te_i)
    vis_cell = torch.zeros(n_te, NCL, dtype=torch.bool)
    ang = torch.arange(-135.0, 135.01, 1.0, device=dev) * math.pi / 180.0
    dr = torch.arange(0.05, RMAX + 1e-6, 0.05, device=dev)
    cr = cells - torch.tensor([mount, 0.0], device=dev)
    c_r = torch.sqrt((cr ** 2).sum(1)); c_a = torch.atan2(cr[:, 1], cr[:, 0])
    ia = torch.clamp(torch.round(c_a * 180.0 / math.pi + 135.0).long(), 0, len(ang) - 1)
    infov = c_a.abs() <= FOV
    with torch.no_grad():
        for j0 in range(0, n_te, 512):
            i = te_g[j0:j0 + 512]
            s0 = st_f[i]
            sx = s0[:, 0] + mount * torch.cos(s0[:, 2]); sy = s0[:, 1] + mount * torch.sin(s0[:, 2])
            a_w = s0[:, 2:3] + ang[None, :]
            px = sx[:, None, None] + dr[None, None, :] * torch.cos(a_w)[..., None]
            py = sy[:, None, None] + dr[None, None, :] * torch.sin(a_w)[..., None]
            hit = trk.dist(px, py, tid_f[i][:, None, None]) <= 0.0
            rng = torch.where(hit.any(-1), dr[hit.float().argmax(-1)], torch.full(hit.shape[:2], RMAX, device=dev))
            vis_cell[j0:j0 + 512] = (infov[None, :] & (c_r[None, :] <= rng[:, ia] + 0.15)).cpu()

    # boxes: visible now (centre in view, plan_rl.boxes_in_view), seen earlier this episode, gone (removed after being seen)
    test_rows = np.unique(bb[te_i])
    box_vis = np.zeros((T, B, SL), bool)
    with torch.no_grad():
        for b in test_rows:
            if not (sid[:, b] >= 0).any():
                continue
            s0 = torch.as_tensor(st[:, b, :3], dtype=torch.float32, device=dev)
            sx = (s0[:, 0] + mount * torch.cos(s0[:, 2]))[:, None]; sy = (s0[:, 1] + mount * torch.sin(s0[:, 2]))[:, None]
            pz = torch.as_tensor(pose[:, b], dtype=torch.float32, device=dev)
            lv = torch.as_tensor(sid[:, b] >= 0, device=dev)
            box_vis[:, b] = los(trk, sx, sy, s0[:, 2:3], pz[..., 0], pz[..., 1], torch.as_tensor(tid[:, b], device=dev)[:, None],
                                0.35, lv).cpu().numpy()
    seen_before = np.zeros((T, B, SL), bool); gone = np.zeros((T, B, SL), bool); gone_pose = np.zeros((T, B, SL, 3))
    gone_sid = np.full((T, B, SL), -1, np.int64)
    for b in test_rows:
        acc = np.zeros(SL, bool); last = np.zeros((SL, 3)); last_sid = np.full(SL, -1); seen_any = np.zeros(SL, bool)
        for t in range(T):
            if t == k0[t, b] or (t > 0 and rs[t - 1, b]):
                acc[:] = False; seen_any[:] = False; last_sid[:] = -1
            live = sid[t, b] >= 0
            moved = live & (last_sid >= 0) & ((np.abs(pose[t, b, :, :2] - last[:, :2]).sum(-1) > 0.05) | (sid[t, b] != last_sid))
            acc[moved] = False; seen_any[moved] = False
            seen_before[t, b] = acc & live
            gone[t, b] = ~live & seen_any & (last_sid >= 0)
            gone_pose[t, b] = last; gone_sid[t, b] = last_sid
            acc = acc | (box_vis[t, b] & live)
            seen_any = seen_any | (box_vis[t, b] & live)
            last = np.where(live[:, None], pose[t, b], last); last_sid = np.where(live, sid[t, b], last_sid)
    # the other car in view: its centre (to 0.35 m short) or a corner
    opp_vis = np.zeros(len(tt), bool)
    if has_opp:
        with torch.no_grad():
            i = te_g
            s0 = st_f[i]
            sx = (s0[:, 0] + mount * torch.cos(s0[:, 2]))[:, None]; sy = (s0[:, 1] + mount * torch.sin(s0[:, 2]))[:, None]
            o_ = torch.as_tensor(op[tt[te_i], bb[te_i]], dtype=torch.float32, device=dev)
            hl, hw = 0.5 * float(car[0]), 0.5 * float(car[1])
            q = torch.tensor([[0.0, 0.0], [hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], device=dev)
            co, so = torch.cos(o_[:, 2])[:, None], torch.sin(o_[:, 2])[:, None]
            px = o_[:, 0:1] + q[None, :, 0] * co - q[None, :, 1] * so; py = o_[:, 1:2] + q[None, :, 0] * so + q[None, :, 1] * co
            v = torch.zeros(len(te_i), 5, dtype=torch.bool, device=dev)
            for k in range(5):
                v[:, k] = los(trk, sx[:, 0], sy[:, 0], s0[:, 2], px[:, k], py[:, k], tid_f[i], 0.35 if k == 0 else 0.05)
            opp_vis[te_i] = v.any(1).cpu().numpy()
    scan_chk = ""
    if has_opp and os.path.exists(scan_p):                       # F1SIM_EXAM_SAVE_SCAN: what each beam hit -> the test above checked
        typ = np.load(scan_p)["typ"]
        beam_car = (typ[tt[te_i], bb[te_i]] == 4).any(-1)                                      # lidar.HIT_CAR
        m_ = there[te_i]
        if m_.any():
            agree = (opp_vis[te_i][m_] == beam_car[m_]).mean()
            scan_chk = (f" 기록의 빔(F1SIM_EXAM_SAVE_SCAN)으로 확인: 10 m 안 결정 중 '보임'이 빔이 차에 맞음과 같은 결정 {100 * agree:.1f} % "
                        f"(보임인데 빔 없음 {100 * (opp_vis[te_i][m_] & ~beam_car[m_]).mean():.1f} %, 가려짐인데 빔 맞음 "
                        f"{100 * (~opp_vis[te_i][m_] & beam_car[m_]).mean():.1f} %).")

    # ---- the baseline: the average map at this pose (the 16 training decisions nearest in true pose)
    from scipy.spatial import cKDTree
    kp = lambda ii: np.stack([X_[ii], Y_[ii], np.cos(YAW[ii]), np.sin(YAW[ii])], 1)
    nn_idx = cKDTree(kp(tr_i)).query(kp(te_i), k=16)[1]
    nn_idx = tr_i[nn_idx]                                                             # (n_te, 16) flat sample indices
    base = {"wall": torch.zeros(n_te, NCL, dtype=torch.float16), "box": torch.zeros(n_te, NCL, dtype=torch.float16)}
    with torch.no_grad():
        for j0 in range(0, n_te, 2048):
            ii = torch.as_tensor(nn_idx[j0:j0 + 2048], device=dev)
            base["wall"][j0:j0 + 2048] = wall_t[ii].float().mean(1).half().cpu()
            base["box"][j0:j0 + 2048] = box_t[ii].float().mean(1).half().cpu()

    # ---- metrics on the test decisions
    wall_te = wall_t[te_g].bool().cpu(); box_te = box_t[te_g].bool().cpu()
    from scipy import ndimage
    tt_te, bb_te = tt[te_i], bb[te_i]
    box_sess = box_s[bb_te]
    # per box instance in the window: its cells, the category
    inst = []                                                                         # (test row, slot, cat) cat 0 a 1 b 2 c 3 gone
    inst_cells, inst_d = [], []
    with torch.no_grad():
        for j0 in range(0, n_te, 1024):
            jj = np.arange(j0, min(j0 + 1024, n_te))
            ii = torch.as_tensor(te_i[jj], device=dev)
            t_, b_ = tt_te[jj], bb_te[jj]
            live = sid[t_, b_] >= 0                                                   # (n, SL)
            g = gone[t_, b_]
            use = live | g
            if not use.any():
                continue
            sids = np.where(live, sid[t_, b_], np.where(g, gone_sid[t_, b_], -1))
            poses = np.where(live[..., None], pose[t_, b_], gone_pose[t_, b_])
            cellm = box_cells(ii, torch.as_tensor(sids, device=dev), torch.as_tensor(poses, dtype=torch.float32, device=dev)).cpu()
            for r, s in zip(*np.nonzero(use)):
                cm = cellm[r, s]
                if not bool(cm.any()):
                    continue                                                          # outside the window
                t1, b1 = t_[r], b_[r]
                cat = 3 if g[r, s] else (0 if box_vis[t1, b1, s] else (1 if seen_before[t1, b1, s] else 2))
                inst.append((j0 + r, s, cat)); inst_cells.append(cm)
                inst_d.append(float(np.hypot(poses[r, s, 0] - st[t1, b1, 0], poses[r, s, 1] - st[t1, b1, 1])))
    inst = np.array(inst, int).reshape(-1, 3)
    inst_d = np.array(inst_d, float)
    inst_cells = torch.stack(inst_cells) if inst_cells else torch.zeros(0, NCL, dtype=torch.bool)
    CAT = ("(a) 지금 보임", "(b) 안 보임·전에 봄", "(c) 안 보임·본 적 없음", "(d) 치운 자리(전에 봄)")

    def box_eval(prob):
        """recall per category, precision (blobs over 0.5 touching a true box, a cell round), blobs a decision on the
        empty-track sessions"""
        hit = np.zeros(len(inst), bool)
        for k0_ in range(0, len(inst), 8192):
            sl = slice(k0_, k0_ + 8192)
            pp = prob[torch.as_tensor(inst[sl, 0])].float()
            hit[sl] = ((pp > 0.5) & inst_cells[sl]).any(1).numpy()
        rec_ = {c: (int(hit[inst[:, 2] == c].sum()), int((inst[:, 2] == c).sum())) for c in range(4)}
        near = inst_d < 4.0
        rec_d = {(c, nr): (int(hit[(inst[:, 2] == c) & (near == nr)].sum()), int(((inst[:, 2] == c) & (near == nr)).sum()))
                 for c in (0, 1) for nr in (True, False)}
        tp = fp = fp_vis = fp_hid = emp_blobs = 0
        n_emp = int((~box_sess).sum())
        st8 = np.ones((3, 3), bool)
        for j in np.nonzero((prob > 0.5).any(1).numpy())[0]:
            pm = (prob[j] > 0.5).numpy().reshape(SD.NX, SD.NY)
            lab, nb = ndimage.label(pm)
            tru = ndimage.binary_dilation(box_te[j].numpy().reshape(SD.NX, SD.NY), st8)
            vz = vis_cell[j].numpy().reshape(SD.NX, SD.NY)
            for k in range(1, nb + 1):
                blob = lab == k
                if (blob & tru).any():
                    tp += 1
                else:
                    fp += 1
                    if vz[blob].mean() >= 0.5:
                        fp_vis += 1
                    else:
                        fp_hid += 1
                if not box_sess[j]:
                    emp_blobs += 1
        return rec_, tp, fp, fp_vis, fp_hid, emp_blobs / max(n_emp, 1), rec_d

    def iou(pred, tru, m):
        a_, b_ = pred & m, tru & m
        return float((a_ & b_).sum()) / max(float((a_ | b_).sum()), 1.0)

    lines = []
    W = lines.append
    ck = name
    nb_te = int(len(set(key[b] for b in test_rows if b in key)))
    W(f"{ck} · {exam} {str(z['mode']) if 'mode' in z.files else ''} · 디코더 시험 = 학습에 안 쓴 상자 배치 {nb_te}개 + 빈 트랙 세션 "
      f"{int(((role == 2) & ~box_s).sum())}개 (결정 {n_te}개, 에피소드 처음 1 s 뺌) | 학습 {len(tr_i)}개 결정")
    W(f"디코더: 2층 MLP {a.hidden} -> pose 4 · ego 4 · 벽 칸 {SD.NX}x{SD.NY} · 상자 칸 {SD.NX}x{SD.NY} (차 기준 앞 {SD.GRID_X0:g}..{SD.GRID_X1:g} m, "
      f"옆 {SD.GRID_Y0:g}..{SD.GRID_Y1:g} m, {SD.GRID_RES:g} m 칸) · 상대 차 10 | 입력 feat {F_} · mem {H_} | {a.epochs} epoch, 검증 배치로 고름")
    W("")
    # the band within 0.5 m of a wall edge: the far inside of a wall mass is wall for anything that knows roughly where it is
    band = torch.zeros_like(wall_te)
    for j0 in range(0, n_te, 4096):
        g = wall_te[j0:j0 + 4096].float().view(-1, 1, SD.NX, SD.NY)
        dil = nn.functional.max_pool2d(g, 5, 1, 2); ero = -nn.functional.max_pool2d(-g, 5, 1, 2)
        band[j0:j0 + 4096] = ((dil > 0.5) & (ero < 0.5)).view(-1, NCL)
    W("| 입력 | 위치 오차 중앙 / 90 % | 방향 오차 중앙 | 벽 IoU 보이는 칸 | 벽 IoU 안 보이는 칸 | 벽 가장자리 0.5 m 띠: 보이는 칸 | 안 보이는 칸 |")
    W("|---|---|---|---|---|---|---|")
    X_te, Y_te, YAW_te = X_[te_i], Y_[te_i], YAW[te_i]
    vc = vis_cell
    pct = lambda k, n: f"{100.0 * k / n:.1f} % ({k}/{n})" if n else "-"
    BASE = "기준: 이 자리의 평균 지도"
    box_rows, box_near = [], []
    for which in SETS + [BASE]:
        if which in preds:
            P = preds[which]
            ph = SD.physical({"pose": P["pose"]}, [x0, x1, y0, y1], ego_scale)
            e_pos = np.hypot(ph["pose"][:, 0] - X_te, ph["pose"][:, 1] - Y_te)
            e_yaw = np.degrees(np.abs(SD.wrap(ph["pose"][:, 2] - YAW_te)))
            wall_p = torch.sigmoid(P["wall"].float()) > 0.5
            box_prob = torch.sigmoid(P["box"].float())
            pos_s = f"{np.median(e_pos):.2f} / {np.percentile(e_pos, 90):.2f} m"; yaw_s = f"{np.median(e_yaw):.1f}°"
        else:
            wall_p = base["wall"].float() > 0.5
            box_prob = base["box"].float()
            pos_s = yaw_s = "(참 pose에서)"
        W(f"| {which} | {pos_s} | {yaw_s} | {iou(wall_p, wall_te, vc):.2f} | {iou(wall_p, wall_te, ~vc):.2f} | "
          f"{iou(wall_p, wall_te, vc & band):.2f} | {iou(wall_p, wall_te, ~vc & band):.2f} |")
        rec_, tp, fp, fpv, fph, emp, rec_d = box_eval(box_prob)
        box_rows.append(f"| {which} | " + " | ".join(pct(*rec_[c]) for c in range(4))
                        + f" | {100.0 * tp / max(tp + fp, 1):.0f} % ({tp}/{tp + fp}) | {fpv} · {fph} | {emp:.3f} |")
        sh = lambda k: f"{100.0 * k[0] / k[1]:.1f} %" if k[1] else "-"
        box_near.append(f"| {which} | {sh(rec_d[(0, True)])} | {sh(rec_d[(0, False)])} | {sh(rec_d[(1, True)])} | {sh(rec_d[(1, False)])} |")
    W("기준 = 학습 결정 중 참 pose가 가장 가까운 16개의 평균 지도 (0.5 넘는 칸). 안 보이는 칸 대부분은 벽 덩어리 안쪽이라 가장자리 띠를 따로 봄.")
    W("")
    W("| 입력 | 상자 찾음 (a) 지금 보임 | (b) 안 보임·전에 봄 | (c) 안 보임·본 적 없음 | (d) 치운 자리 | 정밀도 | 틀린 덩어리: 보이는 곳 · 안 보이는 곳 | 빈 트랙 세션 덩어리/결정 |")
    W("|---|---|---|---|---|---|---|---|")
    for r in box_rows:
        W(r)
    W("")
    W("| 입력 | (a) 지금 보임: 4 m 안 | 4 m 밖 | (b) 안 보임·전에 봄: 4 m 안 | 4 m 밖 |")
    W("|---|---|---|---|---|")
    for r in box_near:
        W(r)
    if SCAN in SETS:
        W(f"{SCAN} = policy가 읽는 가장 새 스캔 하나에 policy와 같은 모양의 스캔 인코더를 새로 붙여 디코더와 함께 이 정답들로 학습한 대조 "
          "(policy 안쪽 값이 아님; 저장 안 함): 그 크기의 인코더가 지금 스캔에서 꺼낼 수 있는 것. feat가 이보다 못하면 policy 인코더가 버린 것.")
    W("상자 찾음 = 그 상자가 걸친 칸 중 하나라도 디코더 확률 > 0.5. (b)는 기억이 지나온 상자를 들고 있는지, (c)·(d)는 0이어야 함 "
      "(아니면 배치를 외웠거나 샘). 정밀도 = 0.5 넘는 덩어리 중 참 상자(한 칸 둘레)에 걸친 것. 보임 = 상자 중심이 시야각·10 m 안, 벽에 안 가림.")
    W("")
    # ego
    eg_te = ego[te_i]; egm_te = ego_m[te_i] > 0
    W("| 입력 | 앞 속도 vx 오차 | 옆 속도 vy 오차 | 회전 속도 r 오차 | 조향 오차 |")
    W("|---|---|---|---|---|")
    for which in SETS:
        e = preds[which]["ego"].float().numpy() * ego_scale
        cells_ = []
        for k, (u, f) in enumerate((("m/s", 1.0), ("m/s", 1.0), ("°/s", 180 / math.pi), ("°", 180 / math.pi))):
            m = egm_te[:, k]
            cells_.append(f"{np.abs(e[m, k] - eg_te[m, k]).mean() * f:.3f} {u}")
        W(f"| {which} | " + " | ".join(cells_) + " |")
    sdv = [eg_te[egm_te[:, k], k].std() * f for k, f in enumerate((1.0, 1.0, 180 / math.pi, 180 / math.pi))]
    W(f"| (정답의 표준편차) | {sdv[0]:.3f} m/s | {sdv[1]:.3f} m/s | {sdv[2]:.1f} °/s | {sdv[3]:.2f} ° |")
    W("평균 절대 오차. 회전 속도·조향 정답은 기록의 vx, vy에서 ideal 실행의 운동학으로 (|vx| < 0.5 m/s 뺌); 정책 입력에 자이로와 마지막 조향 명령이 있음.")
    # opponent
    if has_opp:
        W("")
        th_te = there[te_i]; ov = opp_vis[te_i]; od = odist[te_i]
        W("| 입력 | 상대 차 경우 | 결정 수 | 찾음 (P > 0.5) | 위치 오차 중앙 / 90 % | 방향 오차 중앙 | 속도 오차 | 0.5 s 뒤 위치 오차 중앙 | 1.0 s 뒤 |")
        W("|---|---|---|---|---|---|---|---|---|")
        for which in SETS:
            q = preds[which]["opp"].float().numpy()
            p_ = 1.0 / (1.0 + np.exp(-q[:, 0]))
            px_, py_ = q[:, 1] * SD.OPP_SCALE, q[:, 2] * SD.OPP_SCALE
            ph_ = np.arctan2(q[:, 4], q[:, 3]); pv_ = q[:, 5] * SD.OPP_SCALE
            tx, ty = opp_t[te_i, 1] * SD.OPP_SCALE, opp_t[te_i, 2] * SD.OPP_SCALE
            th = np.arctan2(opp_t[te_i, 4], opp_t[te_i, 3]); tv = opp_t[te_i, 5] * SD.OPP_SCALE
            fo = fut_ok[te_i]
            far = ~(on[tt[te_i], bb[te_i]] & (od <= 12.0))
            sc_ = np.concatenate([q[th_te, 0], q[far, 0]]); y_ = np.concatenate([np.ones(int(th_te.sum())), np.zeros(int(far.sum()))])
            rk = np.argsort(np.argsort(sc_)) + 1.0
            auc_ = ((rk[y_ > 0].sum() - y_.sum() * (y_.sum() + 1) / 2) / max(y_.sum() * (1 - y_).sum(), 1.0)) if 0 < y_.sum() < len(y_) else float("nan")
            for lab, m in (("10 m 안 전체", th_te), ("지금 보임", th_te & ov), ("가려짐", th_te & ~ov),
                           ("3 m 안", th_te & (od < 3.0)), ("3-10 m", th_te & (od >= 3.0)),
                           ("12 m 안에 없음", ~(on[tt[te_i], bb[te_i]] & (od <= 12.0)))):
                n = int(m.sum())
                if n == 0:
                    W(f"| {which} | {lab} | 0 | - | - | - | - | - | - |"); continue
                if lab == "12 m 안에 없음":
                    W(f"| {which} | {lab} | {n} | 잘못 찾음 {100.0 * (p_[m] > 0.5).mean():.1f} % (10 m 안과 가르는 AUC {auc_:.2f}) | - | - | - | - | - |"); continue
                e = np.hypot(px_[m] - tx[m], py_[m] - ty[m])
                eh = np.degrees(np.abs(SD.wrap(ph_[m] - th[m])))
                fs = []
                for j in range(2):
                    mm = m & fo[:, j]
                    ef = np.hypot(q[mm, 6 + 2 * j] * SD.OPP_SCALE - opp_t[te_i][mm, 6 + 2 * j] * SD.OPP_SCALE,
                                  q[mm, 7 + 2 * j] * SD.OPP_SCALE - opp_t[te_i][mm, 7 + 2 * j] * SD.OPP_SCALE)
                    fs.append(f"{np.median(ef):.2f} m" if len(ef) else "-")
                W(f"| {which} | {lab} | {n} | {100.0 * (p_[m] > 0.5).mean():.1f} % | {np.median(e):.2f} / {np.percentile(e, 90):.2f} m | "
                  f"{np.median(eh):.0f}° | {np.abs(pv_[m] - tv[m]).mean():.2f} m/s | {fs[0]} | {fs[1]} |")
        W("상대 차: 우리 차 기준. 보임 = 스캐너에서 그 차 중심(0.35 m 앞까지)이나 모서리 하나가 시야각 ±135°·10 m 안, 벽에 안 가림 "
          "(상자에 가린 것은 안 셈). 0.5 / 1.0 s 뒤 = 지금 차 기준 그때 상대 차 자리. 없음 = 12 m 밖 (10-12 m는 경계라 뺌)." + scan_chk)
    txt = "\n".join(lines) + "\n"
    open(out + "_decode.txt", "w").write(txt)
    print(txt, flush=True)

    # ---- the decoder file (+ a copy beside the checkpoint for the console) and the replay inputs of the test sessions
    for pth in [out + "_dec.npz"] + ([a.ckpt_copy] if a.ckpt_copy else []):
        SD.save(pth, {k: v for k, v in nets.items() if k in SD.SETS}, norms, [x0, x1, y0, y1], ego_scale, F_, H_, record=os.path.basename(rec), checkpoint=name,
                hidden=a.hidden, has_opp=has_opp, extra={"test_sessions": np.nonzero(role == 2)[0], "role": role,
                                                         "epochs": a.epochs})
    ts = np.nonzero(role == 2)[0]
    np.savez_compressed(out + "_decin.npz", feat=feat_all[:, ts], mem=mem_all[:, ts], sessions=ts, role=role,
                        record=np.array(os.path.basename(rec)))
    print(f"저장 {out}_dec.npz · {out}_decin.npz · {out}_decode.txt{' · ' + a.ckpt_copy if a.ckpt_copy else ''} | 전체 "
          f"{time.time() - t_start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
