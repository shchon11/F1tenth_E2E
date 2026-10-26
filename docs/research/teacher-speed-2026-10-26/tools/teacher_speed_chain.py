"""Research (2026-10-26): WHERE does the raceline teacher lose its speed? The user: the policy never uses the car's limits on
straights; measured solo on Korea, no props: policy max 7.05 m/s, distill 6.58, teacher 8.04 (mean 3.90) against its own
raceline profile's 9.67 (where the profile is 6-7 m/s the teacher commands 4.4).
Per learner step this records, binned by the raceline profile speed at the car:
  guard   -- the speed multiplier the env's guard (InteractiveTeacher over the raceline teacher, speeds 1.0/0.8/0.55/0.3) chose
  plan    -- the chosen plan's decoded speeds: wp4 knots 0 and 3 (knot 0 is what MAP reads at the car); linear v0 and v1
  command -- the speed command the follower sent; speed -- the car's speed
usage: teacher_speed_chain.py <evaluate args incl. --teacher>   (env OUT=json path for the per-bin table)"""
import json
import os
import sys

import numpy as np
import torch

from f1sim import gym_env, mpc
from f1sim.learn import common
from f1sim.learn import evaluate as ev

trs, rls = common.load_tracks(["real:korea_2026_competition"], racelines=True, a_lat=7.0, a_acc=6.5, a_brake=4.0,
                              objective="min_time")
XY = torch.tensor(np.asarray(rls[0].xy), dtype=torch.float32)
VP = torch.tensor(np.asarray(rls[0].v), dtype=torch.float32)
SPEEDS = (1.0, 0.8, 0.55, 0.3)
REC = []
_orig = gym_env.F1VecEnv.step


def step(self, a):
    lr = self.learner
    st = self.sim.state[lr]
    d = torch.cdist(st[:, :2].float().cpu(), XY)
    prof = VP[d.argmin(1)]
    v_meas = self._tracker_plan_speed()
    spec = getattr(self.tracker, "_inner", self.tracker).spec
    vmax = float(self.ecfg.v_max_policy)
    cap = self.speed_cap if torch.is_tensor(self.speed_cap) else torch.full_like(v_meas, float(self.speed_cap))
    if spec.speed_mode == "wp4":
        _, _, vk = mpc.decode_wp4(a.float(), v_meas.float(), vmax, cap.float().expand_as(v_meas), spec)
    else:                                              # linear: (v0, v1) placed as knots 0 and 3
        _, _, v0, v1 = mpc.decode(a.float(), v_meas.float(), vmax, cap.float().expand_as(v_meas), spec)
        vk = torch.stack([v0, v0 + (v1 - v0) / 3, v0 + 2 * (v1 - v0) / 3, v1], 1)
    guards = getattr(self, "_raceline_teacher_guards", None) or {}
    g = next(iter(guards.values()), None)
    choice = g.last_choice[lr].cpu() if g is not None and torch.is_tensor(getattr(g, "last_choice", None)) else torch.full((int(lr.sum()),), -1)
    hc = getattr(g, "last_hard_clear", None) if g is not None else None
    full_clear = hc[0][lr].float().cpu() if torch.is_tensor(hc) else torch.full((int(lr.sum()),), -1.0)
    blk = getattr(g, "last_blocked", None) if g is not None else None
    blocked = blk[lr].float().cpu() if torch.is_tensor(blk) else torch.full((int(lr.sum()),), -1.0)
    if g is not None and not getattr(step, "_said", False):
        print("guard preview model:", getattr(g, "last_preview_model", None), flush=True); step._said = True
    out = _orig(self, a)
    REC.append(torch.cat([prof[:, None], choice.float()[:, None], vk[lr].float().cpu(),
                          self.last_cmd_raw[lr, 1:2].float().cpu(), self.sim.state[lr, 3:4].float().cpu(),
                          full_clear[:, None], blocked[:, None]], 1))
    return out


gym_env.F1VecEnv.step = step

if os.environ.get("TEACHER_ALAT") or os.environ.get("TEACHER_ABRAKE"):
    # the teacher's speed profile with other limits on the SAME line (the raceline geometry stays the cached 7/6.5/4 one)
    _orig_make = common.make_teacher

    def make_teacher(rls, env, grip="true", recover_time=0.0, **limits):
        if os.environ.get("TEACHER_ALAT"):
            limits["a_lat"] = float(os.environ["TEACHER_ALAT"])
        if os.environ.get("TEACHER_ABRAKE"):
            limits["a_brake"] = float(os.environ["TEACHER_ABRAKE"])
        print("teacher limits:", limits, flush=True)
        return _orig_make(rls, env, grip=grip, recover_time=recover_time, **limits)
    common.make_teacher = make_teacher


class _NoGuard:
    """NOGUARD=1: the raceline teacher's own plan, without the env's InteractiveTeacher guard (speeds 1.0/0.8/0.55/0.3)"""
    def __init__(self, base):
        self.base, self.last_label_valid = base, None

    def plan_action(self, state, P, tid, v_max, spec, offset=None, plan_speed=None):
        a = self.base.plan_action(state, P, tid, v_max, spec, offset=offset, plan_speed=plan_speed)
        self.last_label_valid = getattr(self.base, "last_label_valid", None)
        return a

    def plan_rows(self, rows, state, P, tid, v_max, spec, offset=None, plan_speed=None):
        return self.plan_action(state, P, tid, v_max, spec, offset=offset, plan_speed=plan_speed)[rows]


if os.environ.get("NOGUARD") == "1":
    _orig_label = gym_env.F1VecEnv.teacher_label

    def teacher_label(self, teacher, *a, **k):
        from f1sim.teacher import RacelineTeacher
        if isinstance(teacher, RacelineTeacher):
            g = getattr(self, "_raceline_teacher_guards", None)
            if g is None:
                g = self._raceline_teacher_guards = {}
            g.setdefault(teacher, _NoGuard(teacher))
        return _orig_label(self, teacher, *a, **k)
    gym_env.F1VecEnv.teacher_label = teacher_label

if os.environ.get("GUARD_PLAN_ONLY") in ("0", "1"):
    # the env's guard as built in teacher_label, walls checked on the plan only (1, the default since 10-26) or on the
    # whole preview (0, before)
    _orig_label3 = gym_env.F1VecEnv.teacher_label

    def teacher_label_window(self, teacher, *a, **k):
        from f1sim.interactive_teacher import InteractiveTeacher
        from f1sim.teacher import RacelineTeacher
        if isinstance(teacher, RacelineTeacher):
            g = getattr(self, "_raceline_teacher_guards", None)
            if g is None:
                g = self._raceline_teacher_guards = {}
            if teacher not in g:
                g[teacher] = InteractiveTeacher(teacher, self, offsets=(0.0,), cand_iters=6, lane_clamp=True,
                                                walls_within_plan=os.environ["GUARD_PLAN_ONLY"] == "1")
        return _orig_label3(self, teacher, *a, **k)
    gym_env.F1VecEnv.teacher_label = teacher_label_window

if os.environ.get("GUARD_CLEAR"):
    # the env's guard as built in teacher_label, with the wall-margin weight set (TeacherCost.clear, default 1.0)
    _orig_label2 = gym_env.F1VecEnv.teacher_label

    def teacher_label_clear(self, teacher, *a, **k):
        from f1sim.interactive_teacher import InteractiveTeacher, TeacherCost
        from f1sim.teacher import RacelineTeacher
        if isinstance(teacher, RacelineTeacher):
            g = getattr(self, "_raceline_teacher_guards", None)
            if g is None:
                g = self._raceline_teacher_guards = {}
            if teacher not in g:
                g[teacher] = InteractiveTeacher(teacher, self, offsets=(0.0,), cand_iters=6, lane_clamp=True,
                                                cost=TeacherCost(clear=float(os.environ["GUARD_CLEAR"])))
        return _orig_label2(self, teacher, *a, **k)
    gym_env.F1VecEnv.teacher_label = teacher_label_clear
sys.argv = ["evaluate"] + sys.argv[1:]
ev.main()
R = torch.cat(REC[100:]).numpy()
if os.environ.get("OUT"):
    np.save(os.environ["OUT"].replace(".json", ".npy"), R)
print(f"choice index range {int(R[:, 1].min())}..{int(R[:, 1].max())}")
prof, ch, vk, cmd, v = R[:, 0], R[:, 1], R[:, 2:6], R[:, 6], R[:, 7]
rows = []
print(f"steps {len(R)} | guard choice shares: " + ", ".join(f"x{s} {100 * np.mean(ch == i):.1f} %" for i, s in enumerate(SPEEDS)))
for lo, hi in ((0, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 11)):
    m = (prof >= lo) & (prof < hi)
    if m.sum() < 20:
        continue
    mult = np.array([SPEEDS[int(c) % len(SPEEDS)] if c >= 0 else np.nan for c in ch[m]])   # offset-major: c = k * len(SPEEDS) + m
    r = dict(bin=f"{lo}-{hi}", share=float(m.mean()), profile=float(prof[m].mean()), guard_mult=float(np.nanmean(mult)),
             full_pace=float(np.mean((ch[m] >= 0) & (ch[m] % len(SPEEDS) == 0))), plan_k0=float(vk[m, 0].mean()), plan_k3=float(vk[m, 3].mean()),
             command=float(cmd[m].mean()), speed=float(v[m].mean()))
    r.update(full_pace_flagged=float(np.mean(R[m, 8] == 0)), blocked=float(np.mean(R[m, 9] == 1)))
    rows.append(r)
    print(f"profile {r['bin']:5s} ({100 * r['share']:4.1f} %): profile {r['profile']:.2f} | guard x{r['guard_mult']:.2f} "
          f"(full pace {100 * r['full_pace']:.0f} %) | plan knot0 {r['plan_k0']:.2f} knot3 {r['plan_k3']:.2f} | "
          f"command {r['command']:.2f} | speed {r['speed']:.2f} | full pace previewed into a wall/prop/car "
          f"{100 * r['full_pace_flagged']:.0f} %, all blocked {100 * r['blocked']:.0f} %")
if os.environ.get("OUT"):
    json.dump(rows, open(os.environ["OUT"], "w"), indent=1)
