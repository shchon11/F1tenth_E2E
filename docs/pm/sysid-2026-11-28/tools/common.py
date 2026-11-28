"""Shared paths and the bag split for the 11-28 lateral re-identification (fixed BEFORE any fitting).

Competition car (08-26/27, one venue): train = the audit's fit half (6 bags) so the 4-value baseline is the audit's
own; the audit's held-out half is split in two by a fixed rule (alternate bags in name order): validation (model
selection) and final check (touched once, at the end).
Pre-competition car (07-14/25, other floors, other VESC setup): its own split, same rule.
The one bag where the real car spun (20260727-204907, map16x07) is kept out of every fit as a limit-behaviour check.
"""
import glob
import os
import sys

REPO = "/home/shchon11/F1tenth_E2E"
sys.path.insert(0, os.path.join(REPO, "f1sim"))
DATA = os.path.join(REPO, "real_data")
WORK = os.environ.get("SYSID_WORK", "/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad/sysid")
OUT = os.path.join(REPO, "docs/pm/sysid-2026-11-28")
FIGS = os.path.join(OUT, "figs")
os.makedirs(WORK, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

SPLIT = {
    "comp": {
        "train": ["20260826-025918", "20260826-174255", "20260826-193250", "20260826-193737", "20260826-194519",
                  "20260827-111616"],
        "val": ["20260826-173704", "20260826-193532", "20260826-194702"],
        "final": ["20260826-192819", "20260826-194050", "20260827-115713"],
    },
    "pre": {
        "train": ["20260725-014941", "20260725-020831", "20260714-223333"],
        "val": ["20260725-020031", "20260714-223707"],
        "final": ["20260725-015428", "20260725-021025"],
    },
    "spin": {"check": ["20260727-204907"]},
}
#: (car setup x place): the profile units. The pre-competition places are the default candidates (PM, 11-28); the
#: competition venue is used for structure selection and validation only.
PLACE = {
    "comp_map08x23": ["20260826-025918", "20260826-173704", "20260826-174255", "20260826-192819", "20260826-193250",
                      "20260826-193532", "20260826-193737", "20260826-194050", "20260826-194519", "20260826-194702",
                      "20260827-111616", "20260827-115713"],
    "pre_map12x16": ["20260725-014941", "20260725-015428", "20260725-020031", "20260725-020831", "20260725-021025"],
    "pre_map17x18": ["20260714-223333", "20260714-223707"],
    "pre_map16x07": ["20260727-204907"],
    "pre_nomap": ["20260725-152213"],
}
#: transfer tests: fit bags at the new place and the held-out bag(s) the refit is judged on
TRANSFER = {
    "comp->map12x16": dict(src="comp", fit=["20260725-014941", "20260725-020031", "20260725-020831"], test=["20260725-015428"]),
    "map12x16->map17x18": dict(src="map12x16", fit=["20260714-223333"], test=["20260714-223707"]),
    "comp->map17x18": dict(src="comp", fit=["20260714-223333"], test=["20260714-223707"]),
}
FALLBACK_SERVO = {"comp": (0.6300, 0.3953), "pre": (-0.5300, 0.6054)}


def bag_dir(key):
    hits = sorted(glob.glob(os.path.join(DATA, "*", key + "*")))
    assert len(hits) == 1, (key, hits)
    return hits[0]


def car_of(key):
    return "comp" if key.startswith("202608") else "pre"


def all_keys():
    return [os.path.basename(d)[:15] for d in sorted(glob.glob(os.path.join(DATA, "*", "2026*")))]

#: acceptance rule, fixed before the final check was run (22:20, 11-28): a candidate is adopted only if, on the FINAL
#: bags, its 1 s yaw-rate median and its 2 s position median are each no worse than 1.05 x the audit's 4-value model
#: (a), its 1 s yaw-rate p90 and its 1 s yaw-rate median on the near-limit windows are no worse than (a)'s, and it
#: spins (|beta| > 20 deg) in <= 1 % of the windows where the real car did not slide. Preference order among
#: candidates that pass: (c2*) (sim structure, physically plausible grip) > (c2) > none (keep the defaults).
ACCEPT = dict(yaw1_tol=1.05, pos2_tol=1.05, spin_calm_max=0.01, order=["(c2*)", "(c2)"])
