"""(11-24) 경주 학습 롤아웃을 빠르게 한 코드(race-fast)가 예전 코드와 똑같이 달리나 -- plan_rl.main 그대로(race32a 인자, GPU 하나).

같은 인자, 같은 seed로 N 결정을 돌리며 결정마다 기록: 모든 차의 시뮬레이터 상태, 학습 차가 뽑은 행동, 보상, 재시작, 학습 차와
복사본의 기억, 복사본의 plan(명령), live tap의 평균 plan, 모든 차의 env 명령(last_cmd). 상대 차 복사본 풀은 race_step_prof.py와 같게
(시작 policy + 인자로 준 checkpoint + 가중치를 조금 흔든 복사본, 풀 밖 복사본도 몲) 채움. 결과를 npz로 저장; 두 npz를 비교하려면
인자 하나 'cmp A.npz B.npz'.
인자: args.json world.json init.pt 복사본_tag수 결정수 out.npz [추가 복사본 checkpoint ...]
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

if sys.argv[1] == "cmp":
    A, B = np.load(sys.argv[2]), np.load(sys.argv[3])
    print(f"[check] {os.path.basename(sys.argv[2])} 대 {os.path.basename(sys.argv[3])}")
    for k in sorted(A.files):
        if k not in B.files:
            print(f"  {k:14s} 한쪽에만 있음")
            continue
        a, b = A[k], B[k]
        if a.shape != b.shape:
            print(f"  {k:14s} 모양 다름 {a.shape} {b.shape}")
            continue
        if a.dtype.kind in "fc":
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            d = np.where(np.isnan(a) & np.isnan(b), 0.0, d)
        else:
            d = (a != b).astype(np.float64)
        bad = np.nonzero(d.reshape(d.shape[0], -1).max(1) > 0)[0] if d.ndim > 0 and d.size else []
        first = int(bad[0]) if len(bad) else -1
        print(f"  {k:14s} 최대 차 {float(np.nanmax(d)) if d.size else 0.0:.3g}  처음 다른 결정 {first}  다른 결정 수 {len(bad)}/{d.shape[0]}  "
              f"{'(완전히 같음)' if not len(bad) else ''}")
    if "last_cmd" in A.files and "last_cmd" in B.files and A["last_cmd"].shape == B["last_cmd"].shape:
        on = (A["absent"] < 0.5) & (B["absent"] < 0.5)
        d = np.abs(A["last_cmd"].astype(np.float64) - B["last_cmd"].astype(np.float64)).max(-1)
        print(f"  last_cmd: 자리에 있는 차만 최대 차 {float(np.where(on, d, 0.0).max()):.3g}, 빈 자리(주차된 차)만 "
              f"{float(np.where(~on, d, 0.0).max()):.3g}")
    raise SystemExit(0)

ARGS_JSON, WORLD, INIT = sys.argv[1], sys.argv[2], sys.argv[3]
N_TAGS, N = int(sys.argv[4]), int(sys.argv[5])
OUT = sys.argv[6]
EXTRA = sys.argv[7:]

from f1sim.learn import plan_rl as PR
from f1sim.learn import plan_policy as PP
from f1sim import plan_steps as PS

saved = json.load(open(ARGS_JSON))["args"]
name = f"racefast_check_{time.strftime('%m%d_%H%M%S')}"
over = dict(name=name, world=WORLD, init=INIT, horizon=N + 2, horizon_auto=0, dist_world=1, dist_rank=0,
            updates=int(saved["update_offset"]) + 1)
_orig_pa = argparse.ArgumentParser.parse_args


def _pa(self, argv=None, namespace=None):
    a = _orig_pa(self, ["--name", name])
    for k, v in saved.items():
        if hasattr(a, k):
            setattr(a, k, v)
    for k, v in over.items():
        setattr(a, k, v)
    return a


argparse.ArgumentParser.parse_args = _pa
R = {k: [] for k in ("state", "u", "reward", "restarted", "mem", "opp_mem", "opp_cmd_d", "opp_cmd_v", "opp_live", "tap_pm",
                     "last_cmd", "absent", "ideal")}
ST = type("ST", (), {})()
ST.lobs = ST.env = ST.drv = None
ST.k = -1
ST.ftg_rows = []


def snap(t):
    return t.detach().float().cpu().numpy().copy()


def boundary():
    ST.k += 1
    if ST.k == 0 or ST.drv is None:
        return
    env = ST.env
    R["state"].append(snap(env.sim.state)); R["last_cmd"].append(snap(env.last_cmd))
    R["absent"].append(snap(env.sim.absent) if getattr(env.sim, "absent", None) is not None else np.zeros(env.B, np.float32))
    R["ideal"].append(snap(env.sim.ideal_rows))
    R["mem"].append(snap(ST.lobs.mem_h))
    so = ST.drv.opp
    R["opp_mem"].append(snap(so.obs.mem_h))
    if so.p is not None:
        R["opp_cmd_d"].append(snap(so.p["cmd_d"])); R["opp_cmd_v"].append(snap(so.cmd_v_driven)); R["opp_live"].append(snap(so.live))
    if ST.k > 1:
        R["tap_pm"].append(np.stack([snap(tp.pm) for tp in ST.taps]) if ST.taps else np.zeros((1, 10, 2), np.float32))
    if len(R["state"]) >= N:
        report()
        os._exit(0)


def report():
    L = min(len(v) for v in R.values() if len(v))
    out = {k: np.stack(v[:L]) for k, v in R.items() if len(v)}
    np.savez_compressed(OUT, **out)
    f = np.array(ST.ftg_rows, dtype=np.float64)
    print(f"[check] {L}결정 기록 -> {OUT}; 외부 운전자(FTG)에 물은 줄 수 (env step당): 평균 {f.mean() if f.size else 0:.1f}, "
          f"최대 {f.max() if f.size else 0:.0f}, 0인 step {int((f == 0).sum())}/{f.size}", flush=True)


_state = PP.PlanObs2.state


def _state_w(self):
    if ST.lobs is not None and self is ST.lobs:
        boundary()
    return _state(self)


PP.PlanObs2.state = _state_w
_drv_init = PR.Driver.__init__


def _drv_init_w(self, env, lim, pr, obs):
    _drv_init(self, env, lim, pr, obs)
    ST.lobs, ST.env, ST.drv = obs, env, self
    for i, d in env.external_drivers.items():
        if hasattr(d, "command"):
            _c = d.command

            def cnt(r, v=None, _c=_c):
                ST.ftg_rows.append(int(np.asarray(r).shape[0]))
                return _c(r, v)
            d.command = cnt
    _ext = env._external_commands

    def ext_w(cmd):
        n0 = len(ST.ftg_rows)
        out = _ext(cmd)
        if len(ST.ftg_rows) == n0:
            ST.ftg_rows.append(0)
        return out
    env._external_commands = ext_w


PR.Driver.__init__ = _drv_init_w
_dec = PR.Driver.decision


def _dec_w(self, a, p=None, **kw):
    R["u"].append(snap(a))
    out = _dec(self, a, p, **kw)
    R["reward"].append(snap(out[0]["reward"])); R["restarted"].append(snap(out[0]["restarted"]))
    return out


PR.Driver.decision = _dec_w
try:
    from f1sim.learn import live_tap as LT
    _lt_init = LT.LiveTap.__init__

    def _lt_init_w(self, *a, **kw):
        _lt_init(self, *a, **kw)
        ST.taps = getattr(ST, "taps", []) + [self]
    LT.LiveTap.__init__ = _lt_init_w
except ImportError:
    pass
ST.taps = []

_add = PR.SelfOpponents.add
_filled = [False]


def _add_w(self, actor, tag):
    _add(self, actor, tag)
    if _filled[0]:
        return
    _filled[0] = True
    dev = self.on.device
    cps = []
    for i, pth in enumerate(EXTRA):
        a = copy.deepcopy(actor)
        a.load_state_dict(torch.load(pth, map_location=dev, weights_only=False)["actor"])
        cps.append((a, int(tag) + 25 * (i + 1)))
    g = torch.Generator(device="cpu"); g.manual_seed(5)
    while len(cps) + 1 < N_TAGS:
        a = copy.deepcopy(actor)
        with torch.no_grad():
            for q in a.parameters():
                q.add_(0.01 * q.std().nan_to_num(0.0) * torch.randn(q.shape, generator=g).to(dev))
        cps.append((a, 9000 + len(cps)))
    every = torch.arange(self.on.numel(), device=dev)
    for a, t_ in cps[:max(0, self.max_pool - 1)]:
        _add(self, a, t_)
    self.draw(every)
    rest = cps[max(0, self.max_pool - 1):]
    for a, t_ in rest:
        _add(self, a, t_)
    if rest:
        self.draw(every[::2])
    used = sorted(set(int(t) for t in self.row_tag[self.on].tolist()))
    print(f"[check] 복사본 tag {len(used)}개가 모는 중: {used} (풀 {self.tags}, archive {sorted(self.archive)})", flush=True)


PR.SelfOpponents.add = _add_w
PR.main()
report()
