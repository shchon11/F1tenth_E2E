#!/usr/bin/env python3
"""Replay `TractionGuard` over the real-car bags and score it against the labelled slip events.

The simulator cannot produce wheel lock or spin at all (no wheel rotation state in
`f1sim/f1sim/dynamics.py`, ground-speed odometry in `odom.py`), so this replay *is* the validation
for `f1sim_ros/f1sim_ros/traction.py`. It reads the bags with the same reader
`evidence/wheelslip_bags.py` uses (`f1sim.calib.bagread.read`) and re-derives that script's labels
here rather than trusting a copied JSON, so labels and detections always come off the same arrays.

    # the whole set, the table that goes into REPORT.md
    python3 scripts/replay_traction.py --root /home/shchon11/F1tenth/real_data --markdown

    # one bag, every event printed, and a parameter sweep entry
    python3 scripts/replay_traction.py --bag 20260826-173704 --events
    python3 scripts/replay_traction.py --set lock_rate=22 --set spin_rate=14 --quiet

Labels (from `evidence/wheelslip_bags.py`, unchanged): with `aw = np.gradient(v_wheel, t)` raw and
`ax` the zero-phase 5 Hz low-pass of the IMU x acceleration,

    lock = (aw < -12) & (v_wheel > 0.3)        spin = (aw > 12) & ((aw - ax) > 8)

and a run is >= 2 consecutive samples. Those are non-causal by construction (`np.gradient` is a
centred difference, the filter runs forwards and backwards); the guard is causal, so a detection is
credited to a labelled run when the two overlap within `--tol` seconds.

Exit status is 0 only when nothing in the must-catch set is missed and nothing fires while the car
is stationary or cruising, so this script can be run as a check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "f1sim"))
sys.path.insert(0, os.path.join(_HERE, "f1sim_ros"))

from f1sim.calib.bagread import bags_under, read                             # noqa: E402
from f1sim_ros.traction import LOCK, OK, SPIN, TractionGuard, TractionParams  # noqa: E402

DEFAULT_ROOT = "/home/shchon11/F1tenth/real_data"
GROUPS = ("01_competition_0826-0827_map08x23", "02_pre-competition")
TOPICS = ["/odom", "/sensors/imu/raw", "/sensors/core", "/drive", "/ackermann_cmd"]
#: The must-catch subset the contract names: every labelled run whose peak wheel acceleration
#: reaches this magnitude.
MUST_CATCH_AW = 30.0


# ------------------------------------------------------------------ labels (evidence definition)

def lp(x, t, fc=5.0):
    """Zero-phase first-order low-pass, verbatim from `evidence/wheelslip_bags.py`."""
    dt = np.median(np.diff(t)); a = np.exp(-2 * np.pi * fc * dt); y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * y[i - 1] + (1 - a) * x[i]
    z = np.empty_like(y); z[-1] = y[-1]
    for i in range(len(y) - 2, -1, -1):
        z[i] = a * z[i + 1] + (1 - a) * y[i]
    return z


def runs(mask, min_len=2):
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def zoh(tt, vv, times):
    """Zero-order hold: a command is a step, not a ramp, so it must not be interpolated."""
    idx = np.searchsorted(tt, times, side="right") - 1
    out = np.full(len(times), np.nan)
    ok = idx >= 0
    out[ok] = vv[idx[ok]]
    return out


def label_runs(t, vw, ax_lp, aw):
    lock = (aw < -12.0) & (vw > 0.3)
    spin = (aw > 12.0) & ((aw - ax_lp) > 8.0)
    out = []
    for (i, j), kind in [(r, LOCK) for r in runs(lock)] + [(r, SPIN) for r in runs(spin)]:
        k = i + int(np.argmax(np.abs(aw[i:j])))
        out.append(dict(kind=kind, t0=float(t[i]), t1=float(t[j - 1]), t_pk=float(t[k]),
                        aw=float(aw[k]), ab=float(ax_lp[k]), v=float(vw[k]),
                        must=bool(abs(aw[k]) >= MUST_CATCH_AW), hit=None))
    out.sort(key=lambda e: e["t0"])
    return out


# ------------------------------------------------------------------ replay

def replay_bag(path, params, cmd_topic="/drive", tol=0.15, quiet_v=1.0):
    """Run the guard over one bag. Returns a row dict; a bag that will not read becomes a result."""
    name = os.path.basename(path.rstrip("/"))
    try:
        b = read(path, topics=TOPICS)
    except Exception as e:
        return dict(bag=name, error=str(e)[:90])
    if not b.has("/odom", "/sensors/imu/raw"):
        return dict(bag=name, error="missing /odom or /sensors/imu/raw")

    t = b.t["/odom"].astype(np.float64)
    vw = b.v["/odom"][:, 3].astype(np.float64)
    dt_med = float(np.median(np.diff(t)))
    imu_ax = b.at("/sensors/imu/raw", t)[:, 3].astype(np.float64)     # already m/s^2 (bagread * G)
    aw_lab = np.gradient(vw, t)
    ax_lab = lp(imu_ax.copy(), t, 5.0)
    labels = label_runs(t, vw, ax_lab, aw_lab)

    has_core = b.has("/sensors/core")
    cur = (b.at("/sensors/core", t)[:, 0].astype(np.float64) if has_core
           else np.full(len(t), np.nan))
    used_cmd = cmd_topic if b.has(cmd_topic) else ("/ackermann_cmd" if b.has("/ackermann_cmd") else None)
    cmd = (zoh(b.t[used_cmd].astype(np.float64), b.v[used_cmd][:, 1].astype(np.float64), t)
           if used_cmd else np.full(len(t), np.nan))
    acc = (zoh(b.t[used_cmd].astype(np.float64), b.v[used_cmd][:, 2].astype(np.float64), t)
           if used_cmd else np.full(len(t), np.nan))

    guard = TractionGuard(params)
    state = np.zeros(len(t), dtype="<U4")
    v_body = np.zeros(len(t)); aw_c = np.zeros(len(t)); ab_c = np.zeros(len(t))
    shaped = np.full(len(t), np.nan)
    for k in range(len(t)):
        st = guard.update(t[k], vw[k], imu_ax[k], None if np.isnan(cur[k]) else cur[k])
        state[k] = st.state
        v_body[k] = st.body_speed; aw_c[k] = st.wheel_accel; ab_c[k] = st.body_accel
        if not np.isnan(cmd[k]):
            shaped[k] = guard.shape(cmd[k], None if np.isnan(acc[k]) else acc[k])

    fires = []
    for kind in (LOCK, SPIN):
        for (i, j) in runs(state == kind, min_len=1):
            w = slice(i, j)
            d = shaped[w] - cmd[w]
            k_pk = i + int(np.argmax(np.abs(aw_c[w])))
            fires.append(dict(kind=kind, t0=float(t[i]), t1=float(t[j - 1]), t_pk=float(t[k_pk]),
                              aw=float(aw_c[k_pk]), ab=float(ab_c[k_pk]),
                              aw_lab=float(aw_lab[w][np.argmax(np.abs(aw_lab[w]))]),
                              v=float(vw[k_pk]), v_body=float(v_body[k_pk]),
                              cmd_in=None if np.isnan(cmd[i]) else float(cmd[i]),
                              cmd_out=None if np.isnan(shaped[i]) else float(shaped[i]),
                              d_cmd=(float(d[np.nanargmax(np.abs(d))])
                                     if np.any(~np.isnan(d)) else None),
                              ms=int(1000 * (t[j - 1] - t[i] + dt_med)), label=None))
    fires.sort(key=lambda e: e["t0"])

    # Match by time overlap within tol, many-to-many: one firing can cover several labelled runs
    # and one labelled run can span several firings. Deliberately not a one-to-one assignment -- the
    # label rule chops a single physical event into two or three runs whenever the centred-difference
    # wheel acceleration dips back under 12 m/s^2 for one sample (20260827-115713 t=42.1 / 42.3 /
    # 42.5 is one brake lock, three runs), and a one-to-one match would score two of the three as
    # misses for an event the guard was latched through. Same kind is preferred over cross-kind: the
    # ERPM sign convention turns a lock while reversing into a positive wheel acceleration, which
    # the label rule calls "spin", so a cross-kind overlap is still a detection of that event.
    def overlaps(f, lab):
        return f["t0"] - tol <= lab["t1"] and lab["t0"] - tol <= f["t1"]

    for want_kind in (True, False):
        for lab in labels:
            if lab["hit"] is not None:
                continue
            for f in fires:
                if (want_kind and lab["kind"] != f["kind"]) or not overlaps(f, lab):
                    continue
                lab["hit"] = f["t_pk"]; lab["cross"] = lab["kind"] != f["kind"]
                break
        for f in fires:
            if f["label"] is not None:
                continue
            for li, lab in enumerate(labels):
                if (want_kind and lab["kind"] != f["kind"]) or not overlaps(f, lab):
                    continue
                f["label"] = li; f["cross"] = lab["kind"] != f["kind"]
                break

    # Context of every unmatched firing. The two the contract forbids are "stationary" and
    # "cruising". "sub-threshold" is the honest third case: the car was moving and the label rule's
    # own wheel acceleration was over its 12 m/s^2 trigger inside the firing window, but not for the
    # two consecutive samples a labelled run needs -- or the wheel collapsed straight through zero,
    # where the rule's `v_wheel > 0.3` gate excludes the sample that carries the evidence. Either
    # way there is no label to match, and the firing is not a false one.
    #
    # "Was the car moving?" is asked of the body speed over the firing window AND the 0.25 s before
    # it, not of the window alone: a lock that takes the wheel to zero also takes the body-speed
    # estimate down with it, so scoring on the window alone would label the end of every hard brake
    # "stationary".
    moving = np.abs(v_body) > quiet_v
    for f in fires:
        if f["label"] is not None:
            continue
        i = int(np.searchsorted(t, f["t0"] - 0.25))
        j = int(np.searchsorted(t, f["t1"], side="right"))
        if not moving[i:max(j, i + 1)].any():
            f["context"] = "stationary"
        elif abs(f["aw_lab"]) >= 12.0:
            f["context"] = f"sub-threshold (label a_wheel {f['aw_lab']:+.1f})"
        else:
            f["context"] = "cruising"

    miss = [lab for lab in labels if lab["hit"] is None]
    fa = [f for f in fires if f["label"] is None]
    return dict(
        bag=name, dt_ms=round(dt_med * 1000, 1), n=len(t), v_max=round(float(vw.max()), 2),
        moving_s=round(float((vw > 0.5).sum() * dt_med), 1), cmd_topic=used_cmd, core=has_core,
        n_lab=len(labels), n_lab_must=sum(1 for e in labels if e["must"]),
        n_lab_lock=sum(1 for e in labels if e["kind"] == LOCK),
        n_lab_spin=sum(1 for e in labels if e["kind"] == SPIN),
        n_fire=len(fires), n_fire_lock=sum(1 for f in fires if f["kind"] == LOCK),
        n_fire_spin=sum(1 for f in fires if f["kind"] == SPIN),
        hit=sum(1 for e in labels if e["hit"] is not None),
        hit_must=sum(1 for e in labels if e["must"] and e["hit"] is not None),
        miss=len(miss), miss_must=sum(1 for e in miss if e["must"]),
        fa=len(fa), fa_stationary=sum(1 for f in fa if f["context"] == "stationary"),
        fa_cruising=sum(1 for f in fa if f["context"] == "cruising"),
        fa_sub=sum(1 for f in fa if f["context"].startswith("sub-threshold")),
        active_s=round(float((state != OK).sum() * dt_med), 2),
        d_cmd_max=round(float(max((abs(f["d_cmd"]) for f in fires if f["d_cmd"] is not None),
                                  default=0.0)), 2),
        labels=labels, fires=fires)


# ------------------------------------------------------------------ reporting

HDR = (f"{'bag':52s} {'mov s':>6s} {'lab':>4s} {'>=30':>5s} {'fire':>5s} {'hit':>4s} "
       f"{'h>=30':>6s} {'miss':>5s} {'m>=30':>6s} {'FA':>4s} {'stat':>5s} {'cruz':>5s} "
       f"{'sub':>5s} {'act s':>6s} {'dcmd':>6s}")


def fmt_row(r):
    if "error" in r:
        return f"{r['bag'][:52]:52s} ERROR {r['error']}"
    return (f"{r['bag'][:52]:52s} {r['moving_s']:6.1f} {r['n_lab']:4d} {r['n_lab_must']:5d} "
            f"{r['n_fire']:5d} {r['hit']:4d} {r['hit_must']:6d} {r['miss']:5d} {r['miss_must']:6d} "
            f"{r['fa']:4d} {r['fa_stationary']:5d} {r['fa_cruising']:5d} {r['fa_sub']:5d} "
            f"{r['active_s']:6.2f} {r['d_cmd_max']:6.2f}")


def totals(rows):
    ok = [r for r in rows if "error" not in r]
    keys = ("n_lab", "n_lab_must", "n_lab_lock", "n_lab_spin", "n_fire", "n_fire_lock",
            "n_fire_spin", "hit", "hit_must", "miss", "miss_must", "fa", "fa_stationary",
            "fa_cruising", "fa_sub")
    t = {k: sum(r[k] for r in ok) for k in keys}
    t["bags"] = len(ok)
    t["moving_s"] = sum(r["moving_s"] for r in ok)
    t["active_s"] = sum(r["active_s"] for r in ok)
    t["d_cmd_max"] = max((r["d_cmd_max"] for r in ok), default=0.0)
    return t


def md_table(rows):
    out = ["| bag | moving s | labels (>=30) | fires | hits (>=30) | misses (>=30) "
           "| false alarms (stationary / cruising / sub-threshold) | guard active s | max dcmd m/s |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        if "error" in r:
            out.append(f"| `{r['bag']}` | - | - | - | - | - | ERROR {r['error']} | - | - |")
            continue
        out.append(f"| `{r['bag']}` | {r['moving_s']:.1f} | {r['n_lab']} ({r['n_lab_must']}) | "
                   f"{r['n_fire']} | {r['hit']} ({r['hit_must']}) | {r['miss']} ({r['miss_must']}) | "
                   f"{r['fa']} ({r['fa_stationary']} / {r['fa_cruising']} / {r['fa_sub']}) | "
                   f"{r['active_s']:.2f} | {r['d_cmd_max']:.2f} |")
    t = totals(rows)
    out.append(f"| **total, {t['bags']} bags** | **{t['moving_s']:.0f}** | **{t['n_lab']} "
               f"({t['n_lab_must']})** | **{t['n_fire']}** | **{t['hit']} ({t['hit_must']})** | "
               f"**{t['miss']} ({t['miss_must']})** | **{t['fa']} ({t['fa_stationary']} / "
               f"{t['fa_cruising']} / {t['fa_sub']})** | **{t['active_s']:.1f}** | "
               f"**{t['d_cmd_max']:.2f}** |")
    return "\n".join(out)


#: Peak |a_wheel| bucket edges, m/s^2, for the detection-rate breakdown. 12 is the label rule's own
#: trigger, 18.3 = a_body_max + 8 is where the guard's absolute gate sits, 30 is the contract's
#: must-catch line.
BUCKETS = (12.0, 18.0, 24.0, 30.0, 45.0, 70.0, float("inf"))


def bucket_table(rows):
    """Detection rate against the peak wheel acceleration of the labelled run. The guard is not
    meant to fire on every labelled run -- it acts only on slip past the friction bound -- so this
    is the shape of what it does and does not take, rather than a single number."""
    lo = 0.0
    out = ["| peak \\|a_wheel\\| of the labelled run | lock runs | spin runs | caught | rate |",
           "| --- | --- | --- | --- | --- |"]
    for hi in BUCKETS:
        sel = [e for r in rows if "error" not in r for e in r["labels"] if lo <= abs(e["aw"]) < hi]
        if sel:
            nl = sum(1 for e in sel if e["kind"] == LOCK)
            hits = sum(1 for e in sel if e["hit"] is not None)
            name = f"{lo:.0f}-{hi:.0f}" if hi != float("inf") else f"{lo:.0f}+"
            out.append(f"| {name} | {nl} | {len(sel) - nl} | {hits} | {100.0 * hits / len(sel):.0f}% |")
        lo = hi
    return "\n".join(out)


def print_events(r):
    print(f"\n  -- {r['bag']}  (command topic {r['cmd_topic']}, "
          f"/sensors/core {'yes' if r['core'] else 'NO'})")
    for i, lab in enumerate(r["labels"]):
        mark = "HIT " if lab["hit"] is not None else "MISS"
        print(f"     label {i:2d} {lab['kind']:4s} {mark} t={lab['t0']:7.2f}-{lab['t1']:7.2f} "
              f"aw={lab['aw']:+7.1f} ab={lab['ab']:+6.1f} v={lab['v']:+5.2f}"
              f"{'  [must catch]' if lab['must'] else ''}")
    for f in r["fires"]:
        tag = (f"-> label {f['label']}" + ("  (kind differs)" if f.get("cross") else "")
               if f["label"] is not None else f"UNMATCHED [{f['context']}]")
        cin = "  n/a" if f["cmd_in"] is None else f"{f['cmd_in']:5.2f}"
        cout = "  n/a" if f["cmd_out"] is None else f"{f['cmd_out']:5.2f}"
        dc = "  n/a" if f["d_cmd"] is None else f"{f['d_cmd']:+5.2f}"
        print(f"     fire  {f['kind']:4s} t={f['t0']:7.2f}-{f['t1']:7.2f} {f['ms']:4d}ms "
              f"aw={f['aw']:+7.1f} ab={f['ab']:+6.1f} v={f['v']:+5.2f} v_body={f['v_body']:+5.2f} "
              f"cmd {cin}->{cout} (max d {dc})  {tag}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="directory holding the bag groups")
    ap.add_argument("--bag", action="append", default=[], help="substring filter, repeatable")
    ap.add_argument("--cmd-topic", default="/drive",
                    help="command topic to shape (/drive is what policy_node publishes; "
                         "/ackermann_cmd is the mux output that reaches the VESC)")
    ap.add_argument("--tol", type=float, default=0.15, help="label/detection overlap tolerance, s")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a TractionParams field, repeatable")
    ap.add_argument("--events", action="store_true", help="print every label and every firing")
    ap.add_argument("--markdown", action="store_true", help="also print the REPORT.md table")
    ap.add_argument("--json", default="", help="write the full result to this path")
    ap.add_argument("--quiet", action="store_true", help="totals only")
    a = ap.parse_args(argv)

    fields = TractionParams.__dataclass_fields__
    kw = {}
    for s in a.set:
        k, _, v = s.partition("=")
        if k not in fields:
            ap.error(f"unknown TractionParams field {k!r}; have {', '.join(sorted(fields))}")
        kw[k] = float(v)
    params = TractionParams(**kw).validate()

    paths = []
    for g in GROUPS:
        d = os.path.join(a.root, g)
        if os.path.isdir(d):
            paths += bags_under(d)
    if not paths:
        paths = bags_under(a.root)
    if a.bag:
        paths = [p for p in paths if any(s in os.path.basename(p.rstrip("/")) for s in a.bag)]
    if not paths:
        ap.error(f"no bags under {a.root}")

    rows = []
    if not a.quiet:
        print("TractionParams: " + ", ".join(f"{k}={getattr(params, k)}" for k in fields))
        print(f"labels: evidence/wheelslip_bags.py rule; must-catch = |a_wheel| >= "
              f"{MUST_CATCH_AW:.0f} m/s^2; match tolerance {a.tol:.2f} s\n")
        print(HDR)
    for p in sorted(paths):
        r = replay_bag(p, params, cmd_topic=a.cmd_topic, tol=a.tol)
        rows.append(r)
        if not a.quiet:
            print(fmt_row(r))
    t = totals(rows)
    print(f"\ntotal over {t['bags']} bags, {t['moving_s']:.0f} s moving: "
          f"{t['n_lab']} labelled runs ({t['n_lab_must']} with |a_wheel| >= {MUST_CATCH_AW:.0f}), "
          f"{t['n_fire']} firings, {t['hit']} hits ({t['hit_must']} of the must-catch set), "
          f"{t['miss']} misses ({t['miss_must']} must-catch), "
          f"{t['fa']} unmatched firings ({t['fa_stationary']} stationary, {t['fa_cruising']} cruising, "
          f"{t['fa_sub']} sub-threshold), guard active {t['active_s']:.1f} s, "
          f"max |dcmd| {t['d_cmd_max']:.2f} m/s")
    if a.events:
        for r in rows:
            if "error" not in r and (r["labels"] or r["fires"]):
                print_events(r)
    if a.markdown:
        print("\n" + md_table(rows))
        print("\n" + bucket_table(rows))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(params={k: getattr(params, k) for k in fields}, tol=a.tol,
                           must_catch_aw=MUST_CATCH_AW, rows=rows), fh, indent=1)
        print(f"\nwrote {a.json}")
    return 0 if t["miss_must"] == 0 and t["fa_stationary"] == 0 and t["fa_cruising"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
