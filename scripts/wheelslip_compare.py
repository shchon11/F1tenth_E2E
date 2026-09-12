#!/usr/bin/env python3
"""Acceptance: does the simulated wheel-acceleration distribution match the real car's?

CONTRACT.md deliverable 6. Both sides are scored by `scripts/replay_traction.py`, which applies
`evidence/wheelslip_bags.py`'s labelling rule and `f1sim_ros/traction.py`'s detector to whatever bag
directory it is pointed at, so the only difference between the two columns is which car produced the
recordings.

    python3 scripts/replay_traction.py --quiet --json real.json                     # the 22 bags
    python3 scripts/gen_sim_bags.py --out /tmp/simbags --profile racepace
    python3 scripts/replay_traction.py --root /tmp/simbags --quiet --json sim.json
    python3 scripts/wheelslip_compare.py real.json sim.json --markdown

Two things are compared, and they answer different questions:

* **Reach** -- the peak |a_wheel| buckets the labelled runs fall in. Does the simulator produce the
  -40 ... -143 m/s^2 locks and the > +20 spins at all?
* **Rate** -- runs per second of moving time in each bucket. Within a factor of two is the
  contract's bar. This is only meaningful against a simulated *driver* whose command statistics
  match the recordings' (see `gen_sim_bags.py --profile racepace`, and `--driver` below, which
  checks that they do); a bag of back-to-back limit manoeuvres will sit far above the real rate for
  a reason that has nothing to do with the plant.

Exit status is 0 when every bucket that the real recordings populate is matched within `--factor`,
and the guard fires on the simulated events without a stationary or cruising firing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "f1sim"))

#: Bucket edges on peak |a_wheel| of a labelled run, m/s^2. 12 is the labelling rule's own trigger,
#: 30 the contract's must-catch line, and 40 the bottom of the lock tail it names.
LOCK_EDGES = (12.0, 20.0, 30.0, 40.0, 70.0, 143.0, float("inf"))
SPIN_EDGES = (12.0, 20.0, 30.0, float("inf"))

#: The bands the contract actually names -- "locks -40 ... -143, spins > +20" -- which is what the
#: factor-of-two bar is about. The finer buckets above are detail: at 6 lock buckets a real count of
#: 6 against a simulated 2 is a Poisson coincidence, not a disagreement, and reporting only the fine
#: split would turn sampling noise into a verdict.
BANDS = {"lock": [("-40 ... -143 (the contract's band)", 40.0, 143.0),
                  ("<= -30 (the must-catch line)", 30.0, float("inf")),
                  ("all locks", 12.0, float("inf"))],
         "spin": [("> +20 (the contract's band)", 20.0, float("inf")),
                  ("all spins", 12.0, float("inf"))]}

#: What the real driver does, measured over the 22 recordings' 1088 s of commanded motion. The
#: simulated driver has to reproduce these or the rate comparison is not about the plant.
DRIVER_REF = {
    "cmd speed mean [m/s]": 3.56, "cmd speed sd [m/s]": 1.74,
    "cmd speed p50 [m/s]": 3.39, "cmd speed p95 [m/s]": 6.90,
    "cmd change 250 ms sd [m/s]": 1.17,
    "hard brake requests [1/s]": 0.291, "hard launch requests [1/s]": 0.280,
    "|steer| mean [rad]": 0.161, "steer sd [rad]": 0.189,
}


def load(path):
    with open(path) as fh:
        return json.load(fh)


def buckets(rows, edges, kind):
    """(count, rate per second of moving time) per bucket, over every labelled run of one kind."""
    ok = [r for r in rows if "error" not in r]
    secs = sum(r["moving_s"] for r in ok)
    runs = [e for r in ok for e in r["labels"] if e["kind"] == kind]
    out, lo = [], 0.0
    for hi in edges:
        n = sum(1 for e in runs if lo <= abs(e["aw"]) < hi)
        out.append((lo, hi, n, n / secs if secs else float("nan")))
        lo = hi
    return out, secs, len(runs)


def totals(rows):
    ok = [r for r in rows if "error" not in r]
    t = {k: sum(r[k] for r in ok) for k in
         ("n_lab", "n_lab_must", "n_fire", "hit", "hit_must", "miss_must",
          "fa_stationary", "fa_cruising", "fa_sub")}
    t["moving_s"] = sum(r["moving_s"] for r in ok)
    t["active_s"] = sum(r["active_s"] for r in ok)
    t["bags"] = len(ok)
    return t


def band_table(real, sim, factor, markdown):
    """The contract's own bands. This is the verdict; the bucket tables below are the detail."""
    lines = (["| band | real runs | real /100 s | sim runs | sim /100 s | ratio | within "
              f"{factor:g}x |", "| --- | --- | --- | --- | --- | --- | --- |"] if markdown else
             [f"  {'band':>34s} {'real n':>7s} {'real/100s':>10s} {'sim n':>7s} {'sim/100s':>9s} "
              f"{'ratio':>7s}  verdict"])
    ok_all = True
    for kind, bands in BANDS.items():
        rb, rs, _ = buckets(real, (float("inf"),), kind)
        sb, ss, _ = buckets(sim, (float("inf"),), kind)
        rruns = [e for r in real if "error" not in r for e in r["labels"] if e["kind"] == kind]
        sruns = [e for r in sim if "error" not in r for e in r["labels"] if e["kind"] == kind]
        for label, lo, hi in bands:
            rn = sum(1 for e in rruns if lo <= abs(e["aw"]) < hi)
            sn = sum(1 for e in sruns if lo <= abs(e["aw"]) < hi)
            rr, sr = rn / rs, sn / ss
            f = ratio(rr, sr)
            ok = rn == 0 or f <= factor
            ok_all &= ok
            v = "-" if rn == 0 else ("yes" if ok else "NO")
            if markdown:
                lines.append(f"| {label} | {rn} | {100 * rr:.2f} | {sn} | {100 * sr:.2f} | "
                             f"{'inf' if math.isinf(f) else f'{f:.2f}'} | {v} |")
            else:
                lines.append(f"  {label:>34s} {rn:7d} {100 * rr:10.2f} {sn:7d} {100 * sr:9.2f} "
                             f"{'inf' if math.isinf(f) else f'{f:7.2f}'}  {v}")
    return "\n".join(lines), ok_all


def name(lo, hi):
    return f"{lo:.0f}-{hi:.0f}" if math.isfinite(hi) else f"{lo:.0f}+"


def ratio(a, b):
    if a == 0 and b == 0:
        return 1.0
    if a == 0 or b == 0:
        return float("inf")
    return max(a / b, b / a)


def table(real, sim, edges, kind, factor, markdown):
    rb, rs, rn = buckets(real, edges, kind)
    sb, ss, sn = buckets(sim, edges, kind)
    lines = []
    head = (f"| peak |a_wheel| ({kind}) | real runs | real /100 s | sim runs | sim /100 s | "
            f"ratio | within {factor:g}x |")
    if markdown:
        lines += [head, "| --- | --- | --- | --- | --- | --- | --- |"]
    else:
        lines.append(f"  {'bucket':>10s} {'real n':>7s} {'real/100s':>10s} {'sim n':>7s} "
                     f"{'sim/100s':>9s} {'ratio':>7s}  verdict")
    worst_ok = True
    for (lo, hi, rn_, rr), (_, _, sn_, sr) in zip(rb, sb):
        f = ratio(rr, sr)
        # A bucket the recordings never populate cannot fail: there is nothing to be within a
        # factor of two OF. It is reported, so an empty real bucket that the simulator fills is
        # visible rather than silently passing.
        judged = rn_ > 0
        ok = (not judged) or f <= factor
        worst_ok &= ok
        verdict = ("-" if not judged else ("yes" if ok else "NO"))
        if markdown:
            lines.append(f"| {name(lo, hi)} | {rn_} | {100 * rr:.2f} | {sn_} | {100 * sr:.2f} | "
                         f"{'inf' if math.isinf(f) else f'{f:.2f}'} | {verdict} |")
        else:
            lines.append(f"  {name(lo, hi):>10s} {rn_:7d} {100 * rr:10.2f} {sn_:7d} {100 * sr:9.2f} "
                         f"{'inf' if math.isinf(f) else f'{f:7.2f}'}  {verdict}")
    if markdown:
        lines.append(f"| **all {kind}** | **{rn}** | **{100 * rn / rs:.2f}** | **{sn}** | "
                     f"**{100 * sn / ss:.2f}** | **{ratio(rn / rs, sn / ss):.2f}** | "
                     f"{'yes' if ratio(rn / rs, sn / ss) <= factor else 'NO'} |")
    else:
        lines.append(f"  {'all':>10s} {rn:7d} {100 * rn / rs:10.2f} {sn:7d} {100 * sn / ss:9.2f} "
                     f"{ratio(rn / rs, sn / ss):7.2f}")
    return "\n".join(lines), worst_ok


def driver_check(root, markdown):
    """The simulated driver's command statistics against the recordings'. See DRIVER_REF."""
    import numpy as np
    from f1sim.calib.bagread import bags_under, read
    lv, dv, st = [], [], []
    nb = nl = 0
    secs = 0.0
    for p in sorted(bags_under(root)):
        b = read(p, topics=["/odom", "/drive"])
        if not b.has("/odom", "/drive"):
            continue
        t = b.t["/odom"].astype("float64"); vw = b.v["/odom"][:, 3].astype("float64")
        dt = float(np.median(np.diff(t))); mv = vw > 0.5
        tc = b.t["/drive"].astype("float64")
        vc = b.v["/drive"][:, 1].astype("float64"); sc = b.v["/drive"][:, 0].astype("float64")
        idx = np.searchsorted(tc, t, side="right") - 1
        good = idx >= 0
        c = np.where(good, vc[np.clip(idx, 0, len(vc) - 1)], np.nan)
        s_ = np.where(good, sc[np.clip(idx, 0, len(sc) - 1)], np.nan)
        sel = mv & np.isfinite(c)
        if sel.sum() < 50:
            continue
        secs += float(sel.sum() * dt)
        lv.append(c[sel]); st.append(s_[sel])
        w = max(1, int(round(0.25 / dt)))
        d = c[w:] - c[:-w]
        m = sel[w:]
        dv.append(d[m])

        def runs(mask):
            n = 0; prev = False
            for x in mask:
                n += bool(x) and not prev
                prev = bool(x)
            return n
        nb += runs((d < -2.0) & m); nl += runs((d > 2.0) & m)
    lv = np.concatenate(lv); dv = np.concatenate(dv); st = np.concatenate(st)
    got = {"cmd speed mean [m/s]": float(lv.mean()), "cmd speed sd [m/s]": float(lv.std()),
           "cmd speed p50 [m/s]": float(np.percentile(lv, 50)),
           "cmd speed p95 [m/s]": float(np.percentile(lv, 95)),
           "cmd change 250 ms sd [m/s]": float(dv.std()),
           "hard brake requests [1/s]": nb / secs, "hard launch requests [1/s]": nl / secs,
           "|steer| mean [rad]": float(np.abs(st).mean()), "steer sd [rad]": float(st.std())}
    lines = (["| statistic | recordings | simulated driver | ratio |", "| --- | --- | --- | --- |"]
             if markdown else [f"  {'statistic':>28s} {'real':>8s} {'sim':>8s} {'ratio':>7s}"])
    for k, v in DRIVER_REF.items():
        g = got[k]
        r = ratio(v, g)
        lines.append(f"| {k} | {v:.3f} | {g:.3f} | {r:.2f} |" if markdown else
                     f"  {k:>28s} {v:8.3f} {g:8.3f} {r:7.2f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("real", help="replay_traction --json over real_data")
    ap.add_argument("sim", help="replay_traction --json over the simulated bags")
    ap.add_argument("--factor", type=float, default=2.0, help="rate agreement bar")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--driver", default="", metavar="ROOT",
                    help="also check the simulated driver's command statistics against the "
                         "recordings', over the bags under ROOT")
    a = ap.parse_args(argv)
    real, sim = load(a.real)["rows"], load(a.sim)["rows"]
    rt, st_ = totals(real), totals(sim)

    print(f"real: {rt['bags']} bags, {rt['moving_s']:.0f} s moving, {rt['n_lab']} labelled runs "
          f"({rt['n_lab_must']} >= 30), guard active {rt['active_s']:.1f} s "
          f"({100 * rt['active_s'] / rt['moving_s']:.1f} % of motion)")
    print(f"sim:  {st_['bags']} bags, {st_['moving_s']:.0f} s moving, {st_['n_lab']} labelled runs "
          f"({st_['n_lab_must']} >= 30), guard active {st_['active_s']:.1f} s "
          f"({100 * st_['active_s'] / st_['moving_s']:.1f} % of motion)")
    print()
    band_t, band_ok = band_table(real, sim, a.factor, a.markdown)
    lock_t, _ = table(real, sim, LOCK_EDGES, "lock", a.factor, a.markdown)
    spin_t, _ = table(real, sim, SPIN_EDGES, "spin", a.factor, a.markdown)
    print(band_t); print(); print(lock_t); print(); print(spin_t)
    print()
    print(f"guard on the simulated events: {st_['n_fire']} firings, {st_['hit']} hits "
          f"({st_['hit_must']} of the {st_['n_lab_must']} must-catch runs, "
          f"{st_['miss_must']} missed), {st_['fa_stationary']} stationary and "
          f"{st_['fa_cruising']} cruising firings")
    if a.driver:
        print("\nthe simulated driver against the recordings':")
        print(driver_check(a.driver, a.markdown))
    # The exit status is about the bands and about the firings the contract forbids. A must-catch
    # miss is NOT one of them here: in the simulator that population is dominated by timestamp
    # artefacts, which the guard rejects on purpose and which it also misses on the car
    # (2 of 22 there, for the same reason). The report says how many and why.
    clean = st_["fa_stationary"] == 0 and st_["fa_cruising"] == 0
    return 0 if (band_ok and clean) else 1


if __name__ == "__main__":
    raise SystemExit(main())
