"""Measurement harness around `python -m f1sim.learn.dagger --arch v2` (copied from v2smoke/run_smoke.py
for the copycat study on feat/v2-copycat: the labeler now takes `rows` and the guard's choice is per
labelled row; nothing else changed).

Nothing in the trainer is changed; this only wraps it to measure:
  * the labelling time inside collection (Wp4Labels.__call__, CUDA-synchronised before/after),
  * env.step time inside collection (synchronised),
  * per-row label validity and the guard's choice, split by traffic (an opponent present in the
    race / within 6 m / within 2 m of the learner),
  * GPU memory (device-wide free memory sampled every 0.5 s; torch peak reserved per phase),
  * every training log row the trainer emits (grad norms, aux-balance multipliers) into a JSONL
    file, since the run goes with --wandb disabled.
usage: SMOKE_OUT=<dir> python run_smoke.py <dagger args...>
"""
import json
import math
import os
import sys
import threading
import time

import torch

OUT = os.environ["SMOKE_OUT"]
os.makedirs(OUT, exist_ok=True)

from f1sim.learn import common, dagger, dagger_v2  # noqa: E402
from f1sim.gym_env import F1VecEnv  # noqa: E402

PHASE = ["init"]
ITER = [0]
ACC = {}
EVENTS = open(os.path.join(OUT, "harness.jsonl"), "a", buffering=1)
LOGROWS = open(os.path.join(OUT, "trainlog.jsonl"), "a", buffering=1)


def emit(kind, **d):
    EVENTS.write(json.dumps({"kind": kind, "t": time.time(), "phase": PHASE[0], "iter": ITER[0], **d}) + "\n")


def acc(key, v):
    ACC[key] = ACC.get(key, 0.0) + float(v)


# ------------------------------------------------------------------ GPU memory sampler
MEM = {"min_free": None, "total": None}
_stop = threading.Event()


def _sampler():
    dev = 0
    while not _stop.is_set():
        try:
            free, total = torch.cuda.mem_get_info(dev)
            MEM["total"] = total
            if MEM["min_free"] is None or free < MEM["min_free"]:
                MEM["min_free"] = free
            MEM.setdefault("phase_min_free", {})
            k = PHASE[0]
            pm = MEM["phase_min_free"]
            if k not in pm or free < pm[k]:
                pm[k] = free
        except Exception:
            pass
        _stop.wait(0.5)


if torch.cuda.is_available():
    threading.Thread(target=_sampler, daemon=True).start()


def sync(dev):
    if isinstance(dev, torch.device) and dev.type == "cuda":
        torch.cuda.synchronize(dev)


# ------------------------------------------------------------------ labels
_orig_call = dagger_v2.Wp4Labels.__call__


def timed_call(self, mu=None, rows=None):
    env = self.env
    dev = env.device
    sync(dev)
    t0 = time.perf_counter()
    lab, valid = _orig_call(self, mu=mu, rows=rows)
    sync(dev)
    dt = time.perf_counter() - t0
    if PHASE[0] != "collect":
        acc(f"label_s_{PHASE[0]}", dt)
        return lab, valid
    acc("label_s", dt)
    acc("label_calls", 1)
    with torch.no_grad():
        M = env.M
        ids = torch.arange(0, env.B, M, device=env.device)
        v = valid[ids].bool()
        # the guard's choice for the learner rows (0 = the plain line at full speed)
        guard = None
        g = getattr(env, "_raceline_teacher_guards", None)
        if g:
            guard = next(iter(g.values()))
        ch = None
        if guard is not None and getattr(guard, "last_choice", None) is not None:
            lc = guard.last_choice
            if rows is not None and lc.shape[0] == rows.numel():
                # a subset call: its choices are the labelled rows', in `rows` order
                full = torch.full((env.B,), -1, dtype=lc.dtype, device=lc.device)
                full[rows.to(lc.device)] = lc
                lc = full
            ch = lc[ids]
        cats = {"all": torch.ones_like(v)}
        if M > 1:
            absent = env.sim.absent.view(-1, M)
            present = ~absent[:, 1:]
            xy = env.sim.state[:, :2].view(-1, M, 2)
            d = (xy[:, 1:] - xy[:, :1]).norm(dim=-1)
            d = torch.where(present, d, torch.full_like(d, float("inf")))
            dmin = d.min(1).values
            cats["race_has_opp"] = present.any(1)
            cats["solo_race"] = ~present.any(1)
            cats["opp_within_6m"] = dmin < 6.0
            cats["opp_within_2m"] = dmin < 2.0
            acc("n_opp_present_sum", float(present.sum()))
        for name, m in cats.items():
            n = int(m.sum())
            acc(f"rows_{name}", n)
            acc(f"valid_{name}", int((v & m).sum()))
            if ch is not None:
                acc(f"line_{name}", int(((ch == 0) & m).sum()))           # plain line, full speed
                acc(f"slow_{name}", int((((ch >= 1) & (ch <= 3)) & m).sum()))
                acc(f"stop_{name}", int(((ch == 4) & m).sum()))
                acc(f"evade_{name}", int(((ch >= 5) & m).sum()))
        acc("label_nonfinite", int((~torch.isfinite(lab[ids]).all(1)).sum()))
    return lab, valid


dagger_v2.Wp4Labels.__call__ = timed_call

# ------------------------------------------------------------------ env.step
_orig_step = F1VecEnv.step


def timed_step(self, action):
    if PHASE[0] != "collect":
        return _orig_step(self, action)
    sync(self.device)
    t0 = time.perf_counter()
    out = _orig_step(self, action)
    sync(self.device)
    acc("step_s", time.perf_counter() - t0)
    acc("step_calls", 1)
    n = int(ACC["step_calls"])
    if n in (1, 2, 5) or n % 20 == 0:
        emit("progress", step_calls=n, step_s=ACC.get("step_s", 0.0), label_s=ACC.get("label_s", 0.0),
             label_calls=ACC.get("label_calls", 0.0))
    return out


F1VecEnv.step = timed_step

# ------------------------------------------------------------------ optional: what env.step spends on
if os.environ.get("SUBTIMERS") == "1":
    from f1sim.mpc import PlanTracker
    from f1sim.sim import Simulator

    def _sub(cls, meth, key, dev_of):
        orig = getattr(cls, meth)

        def w(self, *a, **k):
            if PHASE[0] != "collect":
                return orig(self, *a, **k)
            dev = dev_of(self)
            sync(dev)
            t0 = time.perf_counter()
            out = orig(self, *a, **k)
            sync(dev)
            acc(key, time.perf_counter() - t0)
            return out
        setattr(cls, meth, w)

    _sub(F1VecEnv, "_opponent_actions", "sub_opponents_s", lambda s: s.device)
    _sub(F1VecEnv, "_obs", "sub_obs_s", lambda s: s.device)
    _sub(PlanTracker, "__call__", "sub_tracker_s", lambda s: s.device if isinstance(getattr(s, "device", None), torch.device) else torch.device("cuda"))
    _sub(Simulator, "step", "sub_sim_s", lambda s: s.device)

# ------------------------------------------------------------------ phases
_orig_collect = dagger_v2.collect
_orig_train = dagger_v2.train_epochs
_orig_rollout = common.rollout_metrics


def _phase_begin(name):
    PHASE[0] = name
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def _phase_end(name, t0, **extra):
    dt = time.perf_counter() - t0
    rec = {"wall_s": dt, **extra}
    if torch.cuda.is_available():
        rec["peak_alloc_GB"] = torch.cuda.max_memory_allocated() / 1e9
        rec["peak_reserved_GB"] = torch.cuda.max_memory_reserved() / 1e9
        pm = MEM.get("phase_min_free", {}).get(name)
        if pm is not None and MEM["total"]:
            rec["device_used_peak_GB"] = (MEM["total"] - pm) / 1e9
        MEM.get("phase_min_free", {}).pop(name, None)
    emit(f"{name}_end", **rec)
    PHASE[0] = "idle"
    return rec


_NCOLL = [0]


def collect_wrapped(env, model, labeler, steps, beta, device, buf, **kw):
    ITER[0] = _NCOLL[0]
    _NCOLL[0] += 1
    for k in list(ACC):
        del ACC[k]
    t0 = _phase_begin("collect")
    out = _orig_collect(env, model, labeler, steps, beta, device, buf, **kw)
    rec = _phase_end("collect", t0, beta=beta, steps=steps, B=int(env.B), M=int(env.M), acc=dict(ACC))
    return out


def train_wrapped(model, bufs, **kw):
    t0 = _phase_begin("train")
    loss, parts = _orig_train(model, bufs, **kw)
    _phase_end("train", t0, n_bufs=len(bufs), rows=sum(len(b) for b in bufs),
               valid=sum(b.valid_count for b in bufs), parts=parts)
    return loss, parts


def rollout_wrapped(env, policy_fn, steps, *a, **kw):
    name = "eval"
    t0 = _phase_begin(name)
    ACC.pop("label_s_eval", None)
    m = _orig_rollout(env, policy_fn, steps, *a, **kw)
    _phase_end(name, t0, steps=steps, label_s=ACC.pop("label_s_eval", 0.0),
               metrics={k: v for k, v in m.items() if not isinstance(v, list)})
    return m


dagger_v2.collect = collect_wrapped
dagger_v2.train_epochs = train_wrapped
common.rollout_metrics = rollout_wrapped


# ------------------------------------------------------------------ the trainer's log rows
class _Run:
    def log(self, d):
        LOGROWS.write(json.dumps({"iter_h": ITER[0], **{k: (v if not isinstance(v, float) or math.isfinite(v) else None)
                                                        for k, v in d.items()}}) + "\n")

    def finish(self):
        pass


common.wandb_init = lambda *a, **k: _Run()

if __name__ == "__main__":
    sys.argv = ["f1sim.learn.dagger"] + sys.argv[1:]
    t0 = time.time()
    emit("start", argv=sys.argv, device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    try:
        dagger.main()
    finally:
        _stop.set()
        emit("end", total_s=time.time() - t0,
             device_used_peak_GB=((MEM["total"] - MEM["min_free"]) / 1e9) if MEM["min_free"] else None)
