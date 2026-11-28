"""compbench: the competition-format benchmark (2026-11-28).

One fixed, versioned test for every driver -- E2E checkpoints of any generation, the published
baselines, past checkpoints as opponents -- scored the way the competition scores:

  qualifier: alone, boxes on the line removed mid-session; consecutive laps without a contact,
             best lap, did-not-finish
  final:     1 vs 1, boxes removed mid-session, first to M laps within the time limit wins

The test is defined in files (`suites/*.json`, `profiles/*.json`), each with a version and a
content hash that travels into every result. Running a driver never changes the definition.

Execution conditions:
  ideal  -- the plan driven exactly (the training condition; reference only)
  real   -- the MAP follower + the car's dynamics (the representative condition)
  stress -- real + a wider friction / delay / servo / LiDAR-noise range

The session engine is the prior developer's `exam_record.py`, run UNMODIFIED in-process
(`engine_exam`), with the suite's map / box bank / physics / driver injected around it; a
contract-v2 (October, wp4) checkpoint has its own loop (`engine_v2`) on the same sessions.
Raw records go to /data/compbench (kept across reboots).

Usage:  python -m f1sim.compbench run --driver <ckpt | baseline id> --suite dev --condition real
"""
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(PKG_DIR, "..", "..", ".."))
F1SIM_SRC = os.path.join(REPO, "f1sim")
TOOLS = os.path.join(REPO, "docs", "research", "plan-output-2026-11-01", "tools")
DATA = os.environ.get("COMPBENCH_DATA", "/data/compbench")
PY = os.environ.get("COMPBENCH_PY") or os.path.join(REPO, ".venv", "bin", "python")
#: where repo-relative checkpoint aliases (checkpoints/...) are found (another machine: a copy of just those files)
REPO_FILES = os.environ.get("COMPBENCH_REPO_FILES") or REPO
VERSION = "1.0.0"
