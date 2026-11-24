"""(11-24) 경주 학습(race32a)의 롤아웃 한 결정이 어디에 시간을 쓰나 -- plan_rl.main 그대로(같은 인자, GPU 하나), 몇 결정만.

race32a의 args.json을 그대로 읽어 plan_rl.main을 돌리되(--dist-world 1, 롤아웃 길이만 짧게) 롤아웃이 끝나면 update 전에 나감.
상대 차 복사본 풀은 시작 policy 하나 + 인자로 준 checkpoint들(+ 모자라면 가중치를 조금 흔든 복사본)로 채워 tag가 여럿 섞이게 함
(풀 4개가 차면 다음 것은 archive로 -- 학습 후반처럼 풀 밖 복사본도 계속 몲).
결정 구간:  [0, W) 준비 | [W, W+N1) 동기화 없이 결정당 시간(학습과 같은 조건) | [.., +N2) 부분마다 cuda 동기화해 잰 시간 |
            [.., +N3) cProfile(파이썬 함수별 CPU 시간)
인자: args.json world.json init.pt 복사본_tag수 W N1 N2 N3 [추가 복사본 checkpoint ...]
"""
import argparse
import collections
import copy
import cProfile
import io
import json
import os
import pstats
import sys
import time

import torch

ARGS_JSON, WORLD, INIT = sys.argv[1], sys.argv[2], sys.argv[3]
N_TAGS = int(sys.argv[4])
W, N1, N2, N3 = (int(x) for x in sys.argv[5:9])
N4 = int(os.environ.get("PROF_TORCH", "0"))     # then N4 decisions under torch.profiler (GPU kernel time per part)
EXTRA = sys.argv[9:]
T = W + N1 + N2 + N3 + N4

from f1sim.learn import plan_rl as PR
from f1sim.learn import plan_policy as PP
from f1sim.learn import plan_reward as PRW
from f1sim.learn import scene_replay as SRP
from f1sim.learn import state_rl as SR
from f1sim.learn import live_tap as LT
from f1sim import plan_steps as PS

saved = json.load(open(ARGS_JSON))["args"]
name = f"racefast_prof_{time.strftime('%m%d_%H%M%S')}"
over = dict(name=name, world=WORLD, init=INIT, horizon=T, horizon_auto=0, dist_world=1, dist_rank=0, updates=int(saved["update_offset"]) + 1)

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

# ------------------------------------------------------------------ timers
ST = type("ST", (), {})()
ST.sync = False          # phase 2: every wrapped part between cuda syncs
ST.stack = []
ST.incl = collections.defaultdict(float); ST.excl = collections.defaultdict(float); ST.n = collections.defaultdict(int)
ST.top = 0.0
ST.in_opp = 0
ST.k = -1                # decisions begun
ST.t_last = None
ST.dec = {"free": [], "sync": [], "prof": []}
ST.phase = "warm"
ST.prof = None
ST.lobs = None
ST.tp = None
ST.dec["torch"] = []


def timed(label, fn):
    lab = label if callable(label) else (lambda *a: label)

    def w(*a, **kw):
        if ST.tp is not None:
            with torch.profiler.record_function(lab(*a).strip()):
                return fn(*a, **kw)
        if not ST.sync:
            return fn(*a, **kw)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ST.stack.append(0.0)
        try:
            return fn(*a, **kw)
        finally:
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            ch = ST.stack.pop()
            L = lab(*a)
            ST.incl[L] += dt; ST.excl[L] += dt - ch; ST.n[L] += 1
            if ST.stack:
                ST.stack[-1] += dt
            else:
                ST.top += dt
    return w


def wrap_cls(cls, meth, label):
    setattr(cls, meth, timed(label, getattr(cls, meth)))


def wrap_obj(obj, meth, label):
    f = getattr(obj, meth, None)
    if f is not None:
        setattr(obj, meth, timed(label, f))


# the copies' forward vs the learner's
_opp_plan = PR.SelfOpponents.plan


def _opp_plan_w(self):
    ST.in_opp += 1
    try:
        return _opp_plan(self)
    finally:
        ST.in_opp -= 1


PR.SelfOpponents.plan = _opp_plan_w
wrap_cls(PR.SelfOpponents, "plan", "복사본 plan() 전체")
wrap_cls(PP.Actor2, "forward", lambda *a: "  복사본 actor forward" if ST.in_opp else "학습 actor forward")
wrap_cls(PP.Critic, "forward", "critic forward")
for m_ in ("after", "snapshot", "restore", "draw", "prune", "command", "stepped"):
    wrap_cls(PR.SelfOpponents, m_, f"복사본 {m_}()")
wrap_cls(PR.Driver, "decision", "drv.decision 전체")
PS.roll = timed(lambda *a: "  PS.roll (복사본)" if ST.in_opp else "PS.roll", PS.roll)
wrap_cls(PRW.PlanReward, "control_step", "보상 control_step")
wrap_cls(PRW.PlanReward, "decision", "보상 decision")
wrap_cls(PRW.PlanReward, "plan_cost", "보상 plan_cost")
for m_ in ("capture", "take_loaded", "add", "report", "load", "after_fill"):
    wrap_cls(SRP.SceneReplay, m_, f"장면 {m_}()")
wrap_cls(SR.PropRemoval, "step", "상자 치우기 step")
wrap_cls(SR.OppRestart, "step", "상대 재시작 step")
wrap_cls(LT.LiveTap, "record", "live tap 기록")
PR.track_boxes = timed("track_boxes", PR.track_boxes)
wrap_cls(PP.PlanObs, "priv", "critic 입력 priv()")
wrap_cls(PP.PlanObs2, "set_plan2", "obs set_plan2")
wrap_cls(PP.PlanObs, "push_scan", "obs push_scan")
wrap_cls(PP.PlanObs2, "reset_rows", "obs reset_rows")
_state_t = timed(lambda self: "학습 obs.state()" if self is ST.lobs else "  복사본 obs.state()", PP.PlanObs2.state)


def _state_w(self):
    if ST.lobs is not None and self is ST.lobs and not ST.stack:
        boundary()
    return _state_t(self)


PP.PlanObs2.state = _state_w
_drv_init = PR.Driver.__init__


def _drv_init_w(self, env, lim, pr, obs):
    _drv_init(self, env, lim, pr, obs)
    ST.lobs = obs
    ST.env = env
    wrap_obj(env, "step", "env.step 전체")
    wrap_obj(env, "_opponent_actions", "  env 스크립트 상대 명령")
    wrap_obj(env, "_math", "  env 보상/종료 계산")
    wrap_obj(env, "_obs", "  env 관측")
    wrap_obj(env, "_reset_envs", "  env 재시작")
    wrap_obj(env, "car_gap_closing", "차 간격(near-miss)")
    wrap_obj(env.sim, "step", "  sim.step 전체")
    wrap_obj(env.sim, "_roll", "    sim 물리 roll")
    wrap_obj(env.sim, "_prop_contact", "    sim 상자 접촉")
    wrap_obj(env.sim.lidar, "scan", "    LiDAR scan")


PR.Driver.__init__ = _drv_init_w

# ------------------------------------------------------------------ the copy pool
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
    print(f"[prof] 복사본 tag {len(used)}개가 모는 중: {used} (풀 {self.tags}, archive {sorted(self.archive)})", flush=True)


PR.SelfOpponents.add = _add_w


# ------------------------------------------------------------------ decision boundaries
def boundary():
    now_sync = ST.sync
    if now_sync:
        torch.cuda.synchronize()
    t = time.perf_counter()
    if ST.k >= 0 and ST.t_last is not None and ST.phase in ("free", "sync", "prof", "torch"):
        ST.dec[ST.phase].append(t - ST.t_last)
    ST.k += 1
    k = ST.k
    if ST.phase == "prof" and ST.prof is not None and k >= W + N1 + N2 + N3:
        ST.prof.disable()
    if ST.phase == "torch" and k >= T and ST.tp is not None:
        torch.cuda.synchronize()
        ST.tp.stop()
    ph = ("warm" if k < W else "free" if k < W + N1 else "sync" if k < W + N1 + N2 else "prof" if k < W + N1 + N2 + N3
          else "torch" if k < T else "end")
    if ph == "torch" and ST.phase != "torch":
        torch.cuda.synchronize()
        ST.tp = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        ST.tp.start()
    if ph == "end":
        report()
        sys.stdout.flush()
        os._exit(0)
    if ph == "sync" and ST.phase != "sync":
        torch.cuda.synchronize()
        ST.sync = True
    if ph == "prof" and ST.phase != "prof":
        ST.sync = False
        torch.cuda.synchronize()
        ST.prof = cProfile.Profile()
        ST.prof.enable()
    ST.phase = ph
    if ST.sync:
        torch.cuda.synchronize()
    ST.t_last = time.perf_counter()


def report():
    f = lambda v: f"{sum(v) / max(1, len(v)):.3f} s (n {len(v)}, 최소 {min(v) if v else 0:.3f}, 최대 {max(v) if v else 0:.3f})"
    print(f"\n[prof] 한 결정 (동기화 없음, 학습과 같은 조건): {f(ST.dec['free'])}")
    print(f"[prof] 한 결정 (부분마다 동기화): {f(ST.dec['sync'])}")
    tot = sum(ST.dec["sync"])
    print(f"[prof] 부분별 (동기화 구간 {len(ST.dec['sync'])}결정, 결정당 ms; 포함 = 안쪽 부분 포함, 자기 = 안쪽 뺀 것)")
    for L in sorted(ST.incl, key=lambda L: -ST.incl[L]):
        n_ = max(1, len(ST.dec["sync"]))
        print(f"  {L:28s} 포함 {1e3 * ST.incl[L] / n_:8.1f}  자기 {1e3 * ST.excl[L] / n_:8.1f}  호출/결정 {ST.n[L] / n_:5.1f}")
    print(f"  {'(감싸지 않은 나머지)':28s} {1e3 * (tot - ST.top) / max(1, len(ST.dec['sync'])):8.1f}")
    if ST.prof is not None:
        s = io.StringIO()
        ps = pstats.Stats(ST.prof, stream=s).sort_stats("tottime")
        ps.print_stats(40)
        print(f"\n[prof] cProfile {N3}결정, 자기 시간 순 (상위 40)")
        print("\n".join(s.getvalue().splitlines()[:70]))
        s = io.StringIO()
        pstats.Stats(ST.prof, stream=s).sort_stats("tottime").print_callers(r"method 'to'|method 'tolist'|method 'cpu'|method 'item'|__bool__|nonzero")
        print(f"\n[prof] cProfile {N3}결정, host 동기화가 되는 호출의 호출자")
        print("\n".join(l_ for l_ in s.getvalue().splitlines() if l_.strip())[:12000])
        s = io.StringIO()
        pstats.Stats(ST.prof, stream=s).sort_stats("cumulative").print_stats(45)
        print(f"\n[prof] cProfile {N3}결정, 누적 시간 순 (상위 45)")
        print("\n".join(s.getvalue().splitlines()[:75]))
    if ST.tp is not None:
        ka = ST.tp.key_averages()
        n_ = max(1, N4)
        dev_t = lambda e: getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0.0)
        dev_i = lambda e: getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0.0)
        kern = sum(dev_t(e) for e in ka) / 1e3 / n_
        print(f"\n[prof] torch.profiler {N4}결정: 결정당 벽시계 {sum(ST.dec['torch']) / n_ * 1e3:.1f} ms, GPU 커널 시간 합 {kern:.1f} ms")
        rows = [(e.key, dev_i(e) / 1e3 / n_, e.count / n_) for e in ka if e.key in set(ST.incl) | {L.strip() for L in ST.incl}]
        for k_, t_, c_ in sorted(rows, key=lambda r: -r[1]):
            print(f"  {k_:28s} GPU {t_:7.1f} ms/결정  호출 {c_:5.1f}")
        print(ka.table(sort_by="self_cuda_time_total" if not hasattr(ka[0], "self_device_time_total") else "self_device_time_total",
                       row_limit=25, max_name_column_width=60))
    print(f"[prof] GPU 최대 할당 {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB")


PR.main()
print("[prof] 롤아웃이 끝나기 전에 main이 끝남", flush=True)
