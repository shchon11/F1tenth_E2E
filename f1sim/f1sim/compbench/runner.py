"""Build one job per (driver, suite, condition, map | opponent, start, seed), run it in its own process on the 4070,
keep the raw record + log + job + meta under DATA/raw, score it."""
import os
import sys
import json
import time
import hashlib
import subprocess

from . import DATA, REPO, TOOLS, F1SIM_SRC, PY, VERSION
from .suite import load_suite, resolve_path, check_bank, cell_sha256
from . import drivers as D

LEDGER = os.path.join(DATA, "ledger.jsonl")


def git_rev() -> str:
    try:
        r = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        d = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip()[:12] + ("+dirty" if d.stdout.strip() else "")
    except Exception:
        return "?"


def ledger_count(suite_name: str) -> int:
    """uses of a suite on this machine plus those fetched from others (DATA/ledger_<machine>.jsonl)"""
    import glob
    n = 0
    for f in [LEDGER] + sorted(glob.glob(os.path.join(DATA, "ledger_*.jsonl"))):
        if os.path.exists(f):
            n += sum(1 for ln in open(f) if ln.strip() and json.loads(ln).get("suite") == suite_name)
    return n


def ledger_add(entry: dict) -> None:
    os.makedirs(DATA, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def gpu_env() -> dict:
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = os.environ.get("COMPBENCH_GPU", "0")
    e["PYTHONPATH"] = F1SIM_SRC
    e["WANDB_MODE"] = "disabled"
    e["PYTHONHASHSEED"] = "0"
    return e


def out_dir(s: dict, condition: str, did: str, fmt: str) -> str:
    return os.path.join(DATA, "raw", f"{s['name']}-{s['version']}", fmt, condition, did)


def run_job(job: dict, log: str, engine: str) -> int:
    jp = job["out"][:-4] + ".job.json"
    json.dump(job, open(jp, "w"), indent=1, ensure_ascii=False)
    mod = "f1sim.compbench.engine_exam" if engine == "exam" else "f1sim.compbench.engine_v2"
    t0 = time.time()
    with open(log, "w") as lf:
        rc = subprocess.call([PY, "-u", "-W", "ignore", "-m", mod, jp], stdout=lf, stderr=subprocess.STDOUT,
                             env=gpu_env(), cwd=REPO, timeout=int(job.get("timeout", 5400)))
    print(f"  {os.path.basename(job['out'])}: rc {rc}, {time.time() - t0:.0f} s", flush=True)
    return rc


def base_job(s, cond, m, drv, seed, secs, rm_t, n_env):
    c = s["conditions"][cond]
    prof = s["_profiles"][cond]
    if drv["engine"] == "exam" and "ego" in drv and c["mode"] == "ideal":
        raise ValueError("ideal is the plan driven exactly: a baseline publishes commands, there is nothing to drive ideally")
    return dict(
        compbench=VERSION, suite=s["_tag"], suite_sha256=s["_sha256"], cell_sha256=cell_sha256(s, cond, m, seed),
        condition=cond, mode=c["mode"],
        profile={k: v for k, v in prof.items() if not k.startswith("_")}, profile_sha256=prof["_sha256"],
        map_id=m["id"], track=m["track"], bank=check_bank(m, allow_unfrozen=bool(os.environ.get("COMPBENCH_UNFROZEN"))),
        world=resolve_path(s["world"]), seed=int(seed), secs=float(secs), remove_at=float(rm_t), n_env=int(n_env),
        f1sim_src=F1SIM_SRC, tools=TOOLS, git=git_rev(), driver=drv, all_boxes=True,
        host=__import__("socket").gethostname(), gpu_visible=os.environ.get("COMPBENCH_GPU", "0"),
        host_ckpt=drv.get("ckpt") if drv["engine"] == "exam" and "ckpt" in drv else resolve_path(s["host_ckpt"]))


def run_qual(driver: str, suite: str, condition: str, maps=None, seed=None, force=False, score=True, n_env_scale=1):
    s = load_suite(suite)
    drv = D.resolve(driver, solo=True)
    seed = int(seed if seed is not None else s["seed"])
    _note(s, "qual", condition, drv, seed)
    outs = []
    for m in s["maps"]:
        if maps and m["id"] not in maps:
            continue
        od = out_dir(s, condition, drv["id"], "qual")
        os.makedirs(od, exist_ok=True)
        out = os.path.join(od, f"{m['id']}_s{seed}.npz")
        if os.path.exists(out) and not force:
            print(f"  have {out}", flush=True)
            outs.append(out)
            continue
        job = base_job(s, condition, m, drv, seed, s["qual"]["secs"], s["qual"]["remove_at"],
                       2 * int(m["sessions"]) * n_env_scale)
        job.update(out=out, ego=drv.get("ego"))
        if (drv.get("ego") or {}).get("kind") in ("forzaeth", "forzaeth_pred", "raceline", "lane_switch", "aurorus"):
            job["raceline_cache_only"] = False      # a planner's min-time line on a new map: built once (1-2 min of CPU), then cached
        rc = run_job(job, out[:-4] + ".log", drv["engine"])
        if rc == 0 and os.path.exists(out):
            outs.append(out)
        else:
            print(f"  FAILED {out} (log {out[:-4]}.log)", flush=True)
    if score:
        from .score import score_files
        score_files(outs)
    return outs


def race_opponent(s, opp_id, condition):
    o = next(x for x in s["race"]["opponents"] if x["id"] == opp_id)
    if o["kind"] == "race4":
        return o, None
    slot = {"kind_mix": [], "reactive": {}, "events": [], "event_rate": 0.0, "speed_scale": [1.0, 1.0], "present": 1.0}
    slot.update(o["slot"])
    if "checkpoint" in slot:
        slot["checkpoint"] = resolve_path(slot["checkpoint"])
    if o.get("options"):
        slot["options"] = dict(o["options"])
    if o.get("settings"):
        f, kind = o["settings"].split(":")
        d = json.load(open(resolve_path(f)))[kind]
        st = d.get("race") or d.get("pick")
        if "options" in st:                     # external kinds only: a teacher slot takes no stack settings
            slot["options"] = st["options"]
    return o, slot


def run_race(driver: str, suite: str, condition: str, opponent: str, starts=None, seed=None, force=False, score=True):
    s = load_suite(suite)
    drv = D.resolve(driver, solo=False)
    if drv["engine"] != "exam" or "ego" in drv:
        raise ValueError("races: an E2E plan_steps2 checkpoint drives our car (baselines / v2 race as opponents)")
    seed = int(seed if seed is not None else s["seed"])
    _note(s, "race", condition, drv, seed, opponent)
    m = next(x for x in s["maps"] if x["id"] == s["race"]["map"])
    o, slot = race_opponent(s, opponent, condition)
    outs = []
    for start in (["race4"] if o["kind"] == "race4" else (starts or list(s["race"]["starts"]))):
        od = out_dir(s, condition, drv["id"], "race")
        os.makedirs(od, exist_ok=True)
        out = os.path.join(od, f"{opponent}_{start}_s{seed}.npz")
        if os.path.exists(out) and not force:
            print(f"  have {out}", flush=True)
            outs.append(out)
            continue
        job = base_job(s, condition, m, drv, seed, s["race"]["secs"], s["race"]["remove_at"], 2 * int(s["race"]["matches"]))
        job.update(out=out, opponent=o, start=start, M=int(s["race"]["M"]))
        if o["kind"] == "race4":
            job["race4"] = dict(ref=resolve_path(o["ref"]), paces=o["paces"])
            job["all_boxes"] = False
        else:
            sl = dict(slot, spawn="ahead" if start == "half_lap" else "alongside")
            job.update(opp_slot=sl, half_lap=(start == "half_lap"), start_rest=True)
        rc = run_job(job, out[:-4] + ".log", "exam")
        if rc == 0 and os.path.exists(out):
            outs.append(out)
        else:
            print(f"  FAILED {out} (log {out[:-4]}.log)", flush=True)
    if score:
        from .score import score_files
        score_files(outs, M=int(s["race"]["M"]))
    return outs


def _note(s, fmt, condition, drv, seed, opponent=None):
    n = ledger_count(s["name"])
    ledger_add(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), suite=s["name"], tag=s["_tag"], format=fmt,
                    condition=condition, driver=drv["id"], seed=seed, opponent=opponent, git=git_rev()))
    if s["name"].endswith("confirm"):
        print(f"[compbench] CONFIRM suite: this is use #{n + 1} (ledger {LEDGER}). Use it for final verdicts only.", flush=True)
    print(f"[compbench] {s['_tag']} · {fmt} · {condition} · {drv['id']} ({drv['label']}) · seed {seed}", flush=True)
