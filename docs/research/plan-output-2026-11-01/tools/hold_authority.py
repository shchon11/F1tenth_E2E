"""한 결정이 유지되는 동안의 실행 권한 (2026-11-01, plan-output 설계 (a)).

10-23 `map_authority.py`는 한 스텝만 봤다: MAP이 매 스텝 새 계획을 받으면 속도는 서 있는 점(계획의 4 %), 조향은 L1 점(66 %)만 읽어
wp4 8개 중 v2~v4·kappa4가 죽음. 실제 스택처럼 플래너가 K 스텝마다 결정하고 그 사이 MAP이 그 계획을 트랙 위에 붙잡아 두면
(`EnvConfig.decision_every`) 차가 계획을 따라 나아가며 뒤쪽 점도 읽힐 것 -- 그게 얼마인지.

방법: 체크포인트(wp4 계획 정책)를 한국 맵 혼자 주행(32대 x 30 s)으로 몰며 MAP 호출마다 내 차 행의 입력(계획, 측정 속도, 속도 상한, 요레이트,
지연, MAP 내부 상태, 그립, 지금 자세)을 기록. 결정 스텝(유지 아님)마다 그 뒤 K-1 유지 스텝을 묶어 한 창으로. 각 창, 숫자 i마다
계획 a +- h e_i(h 0.05)로 경로점을 다시 만들고(결정 때의 속도·상한으로), 창의 각 스텝에서 기록된 자세(계획을 낸 자세 기준)와 MAP 상태로
MAP을 다시 불러 명령 차이 |cmd+ - cmd-| / 2h를 창 평균 -> 숫자별 조향 [도/단위], 속도 명령 [m/s/단위]. 1차 근사(바뀐 명령이 그 뒤 자세를
바꾸는 효과는 뺌), 10-23과 같은 정의를 창으로 넓힌 것.
환경 변수: K (결정 주기, 1/2/4/8), LA (MAP speed_lookahead [s]), HZ (wp_horizon_s), EVERY (몇 번째 결정마다, 기본 5), TRACK.
사용: python hold_authority.py CKPT OUT.json
"""
import dataclasses
import json
import os
import sys

import torch

from f1sim import map_controller as mc, mpc
from f1sim.learn import common
from f1sim.learn.evaluate import evaluate
from f1sim.mpc import contract_of
from f1sim.params import Config

CK, OUT = sys.argv[1], sys.argv[2]
K = int(os.environ.get("K", "1"))
LA = float(os.environ.get("LA", "0"))
HZ = float(os.environ.get("HZ", "0.6"))
EVERY = int(os.environ.get("EVERY", "5"))
TRACK = os.environ.get("TRACK", "real:korea_2026_competition")
H = 0.05
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extra = torch.load(CK, map_location="cpu", weights_only=False)["extra"]
MAPJ = dict(contract_of(extra).get("plan_follower_map") or {}, speed_lookahead=LA)

# this design's env: decide every K steps (the plan held on the track between), the plan HZ s of the measured speed
_mk = common.make_env


def _make(trs, n, device, ecfg, *a, **kw):
    ecfg.decision_every = K
    env = _mk(trs, n, device, ecfg, *a, **kw)
    if env.tracker is not None:
        env.tracker._inner.spec = dataclasses.replace(env.tracker._inner.spec, wp_horizon_s=HZ)
    return env


common.make_env = _make
import f1sim.learn.evaluate as _ev  # noqa: E402
_ev.common.make_env = _make

rec = []                       # per MAP call: dict of the learner rows' inputs (cpu)
live = {}
orig = mc.MAPPlanTracker.__call__


def spy(self, action, v_meas, speed_cap, yaw_rate=None, delay=None):
    if self.P is not None and bool(self.map_rows.any()):
        if not live:
            live.update(lut=self.map.lut, p=self.map.p, nominal=self.nominal, spec=self._inner.spec,
                        v_max=float(self._inner.v_max), wb=float(self._inner.wb), s_max=float(self._inner.s_max))
        r = self.map_rows
        B = action.shape[0]
        hold = (self.hold_rows if (self.hold_rows is not None and self._held is not None) else torch.zeros(B, dtype=torch.bool, device=action.device))
        pose = self.pose_now if self.pose_now is not None else torch.zeros(B, 3, device=action.device)
        dly = delay if torch.is_tensor(delay) else torch.full((B,), float(delay or 0.0), device=action.device)
        rec.append({k: v[r].float().cpu() for k, v in dict(
            a=action, v=v_meas, cap=speed_cap, yaw=(torch.zeros(B, device=action.device) if yaw_rate is None else yaw_rate),
            delay=dly, acc=self.map.acc_now, steer0=self.map.curr_steer, curv=self.map.curv_wp,
            mu=self.P["mu"].reshape(-1), muf=self.P["mu_f_scale"].reshape(-1), hold=hold.float(), pose=pose).items()})
    return orig(self, action, v_meas, speed_cap, yaw_rate, delay)


mc.MAPPlanTracker.__call__ = spy
cfg = Config()
cfg.rand.ranges["vehicle.mu"] = (1.0, 1.0)
cfg.rand.ranges["vehicle.mu_f_scale"] = (0.92, 0.92)
ev = evaluate(CK, [TRACK], 32, 1200, 9.0, str(dev), seed=123, cfg=cfg, protocol="trials", budget_laps=3.0,
              plan_follower="map", plan_follower_map=MAPJ, dial_value=1.0489)
mc.MAPPlanTracker.__call__ = orig
spec = dataclasses.replace(live["spec"], wp_horizon_s=HZ)
params = dataclasses.replace(live["p"], speed_lookahead=LA)
T, R = len(rec), rec[0]["a"].shape[0]
# windows: a decision (not hold) at call t for row r, then its hold calls t+1.. while hold
wins = []
for r in range(R):
    t, n_dec = 0, 0
    while t < T:
        if rec[t]["hold"][r] < 0.5:
            j = t + 1
            while j < T and rec[j]["hold"][r] > 0.5 and j - t < K:
                j += 1
            if n_dec % EVERY == 0 and j - t == K:
                wins.append((r, t, j))
            n_dec += 1
            t = j
        else:
            t += 1
names = [f"kappa{i + 1}" for i in range(4)] + [f"v{i + 1}" for i in range(4)]
inner = mpc.PlanTracker(1, dev, live["wb"], live["s_max"], live["v_max"], spec=spec, compile_solver=False)
lut, nominal = live["lut"], live["nominal"]


def stack(key, idx):
    return torch.stack([rec[t][key][r] for (r, t) in idx]).to(dev)


# every (window, step-in-window) as one batch row: its decision's plan inputs, the step's own pose and MAP state
rows_dec, rows_step, wstep = [], [], []
for w, (r, t0, t1) in enumerate(wins):
    for j in range(t0, t1):
        rows_dec.append((r, t0)); rows_step.append((r, j)); wstep.append(j - t0)
N = len(rows_dec)
A = stack("a", rows_dec); Vd = stack("v", rows_dec); Cd = stack("cap", rows_dec); P0 = stack("pose", rows_dec)
Vs = stack("v", rows_step); Ys = stack("yaw", rows_step); Ds = stack("delay", rows_step); Ps = stack("pose", rows_step)
acc, st0, curv = stack("acc", rows_step), stack("steer0", rows_step), stack("curv", rows_step)
mu, muf = stack("mu", rows_step), stack("muf", rows_step)
Cs = stack("cap", rows_step)
wstep = torch.tensor(wstep, device=dev)
d = Ps[:, :2] - P0[:, :2]
c, s = torch.cos(P0[:, 2]), torch.sin(P0[:, 2])
pos = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)
yaw = torch.atan2(torch.sin(Ps[:, 2] - P0[:, 2]), torch.cos(Ps[:, 2] - P0[:, 2]))
inner_n = mpc.PlanTracker(N, dev, live["wb"], live["s_max"], live["v_max"], spec=spec, compile_solver=False)


def cmd(a):
    tr = mc.MAPPlanTracker(inner_n, lut, params, nominal)
    path, pv, pk = tr.waypoints(a, Vd, Cd)
    tr.map.acc_now = acc.clone(); tr.map.curr_steer = st0.clone(); tr.map.curv_wp = curv.clone()
    z = torch.zeros(N, device=dev)
    steer, speed = tr.map(pos, yaw, Vs.abs(), path, pv, pk, torch.full((N,), mc.PLAN_WP_SPACING, device=dev), z,
                          mc.grip_index({"mu": mu, "mu_f_scale": muf}, nominal), yaw_rate=Ys,
                          delay=Ds if tr.map.p.delay_comp < 0 else None)
    return torch.stack([steer, torch.minimum(speed, Cs)], 1)


res = {"ckpt": CK, "K": K, "speed_lookahead": LA, "wp_horizon_s": HZ, "windows": len(wins), "rows": N,
       "lap": ev.get("lap_time_s") if isinstance(ev, dict) else None, "dims": {}}
for i, nm in enumerate(names):
    e = torch.zeros_like(A); e[:, i] = H
    dd = (cmd((A + e).clamp(-1, 1)) - cmd((A - e).clamp(-1, 1))).abs() / (2 * H)
    res["dims"][nm] = {"steer_deg": round(float(dd[:, 0].mean()) * 57.2958, 3), "v_cmd": round(float(dd[:, 1].mean()), 3),
                       "by_step": [[round(float(dd[wstep == k, 0].mean()) * 57.2958, 2), round(float(dd[wstep == k, 1].mean()), 3)]
                                   for k in range(K)]}
print(f"[권한] K {K} 속도 미리보기 {LA} s 계획 {HZ} s | 창 {len(wins)}개 | " +
      " ".join(f"{nm} {res['dims'][nm]['steer_deg']:.1f}도/{res['dims'][nm]['v_cmd']:.2f}" for nm in names), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
