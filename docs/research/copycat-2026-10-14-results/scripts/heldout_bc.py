"""Held-out imitation error of v2 DAgger checkpoints, with no training.

Builds the env and the teacher through `f1sim.learn.dagger.main` itself (its `dagger_v2.run` is swapped
for the function below), so the world is the training world on the tracks given by --tracks. Collects
two held-out buffers with the trainer's own `dagger_v2.collect` -- the teacher driving (beta 1) and the
LAST checkpoint driving (beta 0, no noise) -- and scores every checkpoint on the same windows (fixed
seed): the decoded-space BC (the trainer's Huber, and plain MAE of the four waypoints' lateral position
[m] and speed [m/s]) and B8, overall and on rows with a B8 candidate car.
env: HELD_CKPTS (comma list, last = the driving student), HELD_OUT (json), HELD_STEPS, HELD_NSEQ, HELD_NBATCH
"""
import json
import os
import sys
import time

import torch

from f1sim.learn import common, dagger, dagger_v2
from f1sim.learn import conditioning as cond_mod
from f1sim.learn.model import load_checkpoint
from f1sim.learn.train_v2 import b8_sequence_loss, bc_loss_wp4, steps_since_window_start, wp4_decoded

CKPTS = [p for p in os.environ["HELD_CKPTS"].split(",") if p]
OUTF = os.environ["HELD_OUT"]
STEPS = int(os.environ.get("HELD_STEPS", "400"))
NSEQ = int(os.environ.get("HELD_NSEQ", "64"))
NBATCH = int(os.environ.get("HELD_NBATCH", "8"))


@torch.no_grad()
def score(model, s, burn, plan_spec, v_max):
    actor = model.actor
    out = actor.sequence(s["scan"], s["ego"], s["c"], keep=s["keep"], burn_in=burn)
    sl = slice(burn, None)
    T, m = out["mu"].shape[:2]
    flat = lambda x: x[sl].reshape(T * m, *x.shape[2:])
    v_meas = flat(s["ego"])[:, 0] * v_max
    valid = flat(s["valid"]).float()
    mu = out["mu"].reshape(T * m, -1)
    lab = flat(s["label"])
    bc, parts = bc_loss_wp4(mu, lab, v_meas, plan_spec, v_max, weight=valid)
    ys, vs = wp4_decoded(mu, v_meas, plan_spec, v_max)
    yt, vt = wp4_decoded(lab, v_meas, plan_spec, v_max)
    e_lat = (ys - yt).abs().mean(-1)
    e_v = (vs - vt).abs().mean(-1)
    e_lat_end = (ys[:, -1] - yt[:, -1]).abs()
    e_v_end = (vs[:, -1] - vt[:, -1]).abs()
    cand_any = flat(s["b8_cand"]).any(-1) if "b8_cand" in s else torch.zeros_like(valid, dtype=torch.bool)
    res = {"bc_huber": float(bc), "bc_lat_huber": float(parts["lat"]), "bc_speed_huber": float(parts["speed"])}
    sgn_v = (vs - vt).mean(-1)
    ok = valid > 0
    masks = [("all", ok), ("car_rows", ok & cand_any), ("no_car_rows", ok & ~cand_any),
             ("vmeas_lt1", ok & (v_meas < 1.0)), ("vmeas_1to3", ok & (v_meas >= 1.0) & (v_meas < 3.0)),
             ("vmeas_ge3", ok & (v_meas >= 3.0))]
    for name, msk in masks:
        n = int(msk.sum())
        if n:
            res[f"bias_v_mps_{name}"] = float(sgn_v[msk].mean())
            res[f"label_v_mps_{name}"] = float(vt[msk].mean())
    res["n_rows_scored"] = int(valid.numel())
    res["n_invalid"] = int((valid == 0).sum())
    # trivial predictors on the same rows: a straight path (y = 0) and "keep the measured speed"
    base_lat = yt.abs().mean(-1)
    base_v = (vt - v_meas[:, None]).abs().mean(-1)
    # the student's own curvature sign vs the label's (first waypoint's lateral position), where the label turns
    turning = yt[:, -1].abs() > 0.05
    for name, msk in masks:
        n = int(msk.sum())
        if n:
            res[f"base_lat_m_{name}"] = float(base_lat[msk].mean())
            res[f"base_v_mps_{name}"] = float(base_v[msk].mean())
            mt = msk & turning
            if int(mt.sum()):
                res[f"turn_sign_agree_{name}"] = float((torch.sign(ys[mt, -1]) == torch.sign(yt[mt, -1])).float().mean())
                res[f"turn_ratio_{name}"] = float((ys[mt, -1] / yt[mt, -1]).clamp(-3, 3).median())
    for name, msk in masks:
        n = int(msk.sum())
        res[f"n_{name}"] = n
        if n:
            res[f"mae_lat_m_{name}"] = float(e_lat[msk].mean())
            res[f"mae_v_mps_{name}"] = float(e_v[msk].mean())
            res[f"mae_lat_end_m_{name}"] = float(e_lat_end[msk].mean())
            res[f"mae_v_end_mps_{name}"] = float(e_v_end[msk].mean())
    if "b8" in out and "b8_lab" in s:
        since = steps_since_window_start(s["keep"])[sl]
        b8, st = b8_sequence_loss(out["b8"], s["b8_lab"][sl], s["b8_cand"][sl], s["b8_age"][sl], since)
        res["b8"] = float(b8)
        res["b8_knowable"] = float(st["knowable"])
        res["b8_cand_per_row"] = float(st["candidates_per_row"])
    return res


def heldout_run(a, *, env, teacher, teacher_desc, names, device, applied):
    spec = common.obs_spec(env)
    models = []
    for p in CKPTS:
        m, extra = load_checkpoint(p, device)
        m.eval()
        models.append((p, m, extra))
    final = models[-1][1]
    seq = models[-1][2].get("sequence") or {"burn_in": 40, "chunk": 80}
    burn, chunk = int(seq["burn_in"]), int(seq["chunk"])
    cspec = cond_mod.CondSpec.from_meta(final.meta["cond"])
    dial = cond_mod.DialDraw(env, a.dial_margin, a.dial_exact)
    labeler = dagger_v2.Wp4Labels(env, teacher, "env")
    plan_spec = env.tracker.spec
    v_max = float(env.ecfg.v_max_policy)
    env.sim.warmup()
    report = {"tracks": names, "ckpts": CKPTS, "steps": STEPS, "envs": int(env.B), "race_size": int(env.M),
              "burn": burn, "chunk": chunk, "nseq": NSEQ, "nbatch": NBATCH, "buffers": {}, "scores": []}
    bufs = {}
    for tag, beta in (("teacher_drives", 1.0), ("last_student_drives", 0.0)):
        torch.manual_seed(77)
        t0 = time.time()
        buf = dagger_v2.collect(env, final, labeler, STEPS, beta, device,
                                dagger_v2.SequenceBuffer(spec.n_frames, spec.frame_stride), noise=0.0, dial=dial,
                                cspec=cspec, b8=True, near=False, beam=False).finalize()
        bufs[tag] = buf
        if os.environ.get("HELD_SAVE"):
            torch.save(buf.data, os.environ["HELD_SAVE"] + f"_{tag}.pt")
        report["buffers"][tag] = {"rows": len(buf), "valid": buf.valid_count, "collect_s": time.time() - t0}
        print(f"held-out buffer {tag}: {len(buf)} rows, {buf.valid_count} valid, {time.time() - t0:.0f} s", flush=True)
    for tag, buf in bufs.items():
        for p, m, extra in models:
            torch.manual_seed(1234)          # the same windows for every checkpoint
            acc = {}
            for _ in range(NBATCH):
                s = buf.sample(NSEQ, burn + chunk, device)
                r = score(m, s, burn, plan_spec, v_max)
                for k, v in r.items():
                    acc.setdefault(k, []).append(v)
            row = {"buffer": tag, "ckpt": os.path.basename(p), "iter": extra.get("iter")}
            for k, v in acc.items():
                if k.startswith("n_"):
                    row[k] = int(sum(v))
                else:
                    row[k] = float(sum(v) / len(v))
            report["scores"].append(row)
            print(json.dumps(row), flush=True)
    json.dump(report, open(OUTF, "w"), indent=1)


dagger_v2.run = heldout_run
common.wandb_init = lambda *a, **k: None

if __name__ == "__main__":
    sys.argv = ["f1sim.learn.dagger"] + sys.argv[1:]
    dagger.main()
