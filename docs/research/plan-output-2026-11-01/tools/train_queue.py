"""The training PC's side of the console's training page (11-24, the user: "콘솔에서도 내가 학습할 수 있게 ... 학습 패널로
퀄리나 지금 학습세팅들 좀 만질 수 있게", "여가서 학습"): takes the requests the page leaves (f1sim.learn.train_requests) --
from this PC's runs folder and from the lab PC's over ssh (this PC reaches the lab; the lab cannot reach this PC) --
starts runs with tools/launch_run.sh or stops them by PID, and writes what runs and how far (progress, checkpoints, the
exam rows per checkpoint, every request's outcome) to <runs>/_train_status.json here and on the lab.

Runs as the user service f1-trainq.service (KillMode=process; each run is launched in its own systemd scope, so
restarting the service never touches a training run). Never runs a command line from a request: a start is
launch_run.sh with the request's name, world, checkpoint, end update and plan_rl flags, each checked against a strict
pattern, and first `CHECK=1 launch_run.sh ...` (plan_rl --check-args: the same arguments parsed and checked, no GPU)
-- a refused start stops nothing.

    python3 train_queue.py            # the loop
    python3 train_queue.py --once     # one pass (tests; --no-lab: without the lab PC)

~/.config/systemd/user/f1-trainq.service (then systemctl --user daemon-reload && systemctl --user enable --now f1-trainq):
    [Service]
    WorkingDirectory=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
    ExecStart=/home/shchon11/F1tenth_E2E/.venv/bin/python3 train_queue.py
    Restart=always
    RestartSec=15
    KillMode=process
    [Install]
    WantedBy=default.target"""
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(RT, "out")
F1 = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "f1sim"))
sys.path.insert(0, F1)
from f1sim.learn import train_requests as TR          # noqa: E402

RUNS = TR.runs_dir()
CFG = os.path.join(F1, "f1sim", "learn", "configs")
LAUNCH = os.path.join(HERE, "launch_run.sh")
PLAN_RL = os.path.join(F1, "f1sim", "learn", "plan_rl.py")
LAB = os.environ.get("LAB_H", "shchon11@10.100.0.2")
_CM = os.path.join(os.environ.get("XDG_RUNTIME_DIR") or "/tmp", "f1trainq_%C")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ControlMaster=auto", "-o", f"ControlPath={_CM}",
       "-o", "ControlPersist=300"]
JOBS = os.path.join(RUNS, "_train_queue_jobs.json")
PRESETS = os.path.join(RUNS, "_train_presets.json")
LOG = os.path.join(OUT, "train_queue.log")
SCRATCH_ONLY = TR.SCRATCH_ONLY
BASE_NAME = "_trainq_base"          # the name the baseline check parses under (never a run folder)


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + msg + "\n")


def load(p, default):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save(p, d):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


# ------------------------------------------------------------------ requests
def local_requests():
    d = TR.req_dir(RUNS)
    os.makedirs(os.path.join(d, "taken"), exist_ok=True)
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = load(p, None)
        os.replace(p, os.path.join(d, "taken", os.path.basename(p)))
        out.append((r, "이 PC"))
    return out


LAB_PULL = r"""
import glob, json, os
d = os.path.expanduser("~/f1sim_runs/_train_requests")
os.makedirs(os.path.join(d, "taken"), exist_ok=True)
out = []
for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    try:
        out.append(json.load(open(p)))
    except Exception:
        out.append(None)
    os.replace(p, os.path.join(d, "taken", os.path.basename(p)))
print(json.dumps(out))
"""


def lab_requests():
    try:
        r = subprocess.run(SSH + [LAB, "python3", "-"], input=LAB_PULL, capture_output=True, text=True, timeout=60)
        return [(x, "연구실 PC") for x in json.loads(r.stdout or "[]")]
    except (subprocess.SubprocessError, ValueError, OSError):
        return []


def procs():
    out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
    return [(int(l.split(None, 1)[0]), l.split(None, 1)[1]) for l in out.splitlines() if len(l.split(None, 1)) == 2]


def _rank_re(name):
    return re.compile(rf"-m f1sim\.learn\.plan_rl --name {re.escape(name)} ")


def plan_runs():
    """{name: [pids]} of the plan_rl training processes on this PC (not a --check-args parse)"""
    out = {}
    for pid, a in procs():
        m = re.search(r"-m f1sim\.learn\.plan_rl --name (\S+) ", a)
        if m and not a.startswith("bash -c") and "--check-args" not in a:
            out.setdefault(m.group(1), []).append(pid)
    return out


def _signal(pids, sig):
    for p in pids:
        try:
            os.kill(p, sig)
        except OSError:
            pass


def stop_run(name):
    """the run's GPU guard first (it pauses ranks with SIGSTOP and would resume or re-pause them), then its trainer
    processes (SIGTERM + SIGCONT, so a paused rank dies too; SIGKILL after 30 s), then the other helpers (exam loop,
    lab sync, console watch) -- all by PID. Returns (trainer processes, helpers)"""
    e = re.escape(name)
    guard = [p for p, a in procs() if re.search(rf"gpu_guard3\.sh plan_\[r\]l --name {e} ", a) and a.startswith("bash ")]
    _signal(guard, signal.SIGTERM)
    rank = [p for p, a in procs() if _rank_re(name).search(a) and not a.startswith("bash -c") and "--check-args" not in a]
    _signal(rank, signal.SIGTERM)
    _signal(rank, signal.SIGCONT)
    for i in range(40):
        left = {p for p, _ in procs()} & set(rank)
        if not left:
            break
        if i == 30:
            _signal(left, signal.SIGKILL)
        time.sleep(1)
    pats = [rf"lab_exam2m_loop\.sh {e} ", rf"lab_exam_loop\.sh {e} ", rf"exam_loop\.sh {e} ", rf"core_loop\.sh {e} ",
            rf"lab_watch_sync\.sh {e}\b", rf"watch_latest_plan\.sh watch_{e} "]
    helpers = [p for p, a in procs() if a.startswith("bash ") and any(re.search(x, a) for x in pats)]
    _signal(helpers, signal.SIGTERM)
    relabel(name, "멈춤")
    return len(rank), len(helpers) + len(guard)


def relabel(name, word):
    """the console list's line for the run: '학습 중 · name …' -> 'word · name …'"""
    p = os.path.join(RUNS, "console_picks.json")
    d = load(p, None)
    if not d:
        return
    hit = False
    for e in d.get("picks", []):
        if e.get("run") == f"watch_{name}" and e.get("label", "").startswith("학습 중"):
            e["label"] = word + e["label"][len("학습 중"):]
            hit = True
    if hit:
        save(p, d)


def launch_argv(name, world_path, updates, init, flags):
    """launch_run.sh's arguments for a request (a continued run drops the flags that only mean something from scratch)"""
    cmd = ["bash", LAUNCH, name, world_path, str(int(updates))]
    if init:
        cmd += [os.path.join(RUNS, init), str(TR.init_update(init))]
        flags = [(f, v) for f, v in flags if f not in SCRATCH_ONLY]
    else:
        cmd += ["", "0"]                    # launch_run.sh's checkpoint and its update are positional: none
    for f, v in flags:
        cmd += [f] + ([v] if v != "" else [])
    return cmd


def check_args(cmd, gpu):
    """(ok, parsed or reason): plan_rl --check-args through launch_run.sh's CHECK mode, the very arguments a launch
    would give it; nothing is started and no GPU is touched"""
    p = subprocess.run(cmd, cwd=RT, env=dict(os.environ, CHECK="1", LAUNCH_GPU=gpu), capture_output=True, text=True,
                       timeout=300)
    if p.returncode == 0:
        try:
            return True, json.loads(p.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return False, "인자 확인 결과를 읽지 못함"
    lines = [l.strip() for l in (p.stderr + "\n" + p.stdout).splitlines() if l.strip()]
    why = next((l for l in reversed(lines) if "error:" in l), lines[-1] if lines else f"rc {p.returncode}")
    return False, why.replace("plan_rl.py: error: ", "")


_SCOPE = None


def scope_prefix(name):
    """the launch in its own systemd scope: the run's processes are not in this service's cgroup, so restarting or
    stopping the service never kills them (checked once; without systemd-run, KillMode=process still spares them)"""
    global _SCOPE
    if _SCOPE is None:
        try:
            _SCOPE = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "true"], capture_output=True,
                                    timeout=30).returncode == 0
        except (OSError, subprocess.SubprocessError):
            _SCOPE = False
    return (["systemd-run", "--user", "--scope", "--quiet", f"--unit=f1train-{name}-{int(time.time())}"]
            if _SCOPE else [])


def start_run(r):
    name = r["name"]
    if os.path.exists(os.path.join(RUNS, name)):
        return f"이름 {name}이 이미 있음 -- 다른 이름으로"
    world = os.path.join(CFG, r["world"])
    if not os.path.exists(world):
        return f"맵 설정 {r['world']} 없음"
    init = str(r.get("init") or "")
    if init and not os.path.exists(os.path.join(RUNS, init)):
        return f"checkpoint {init} 없음"
    gpu = str(r.get("gpu", "both"))
    cmd = launch_argv(name, world, r["updates"], init, TR.check_flags(r.get("flags") or []))
    ok, res = check_args(cmd, gpu)
    if not ok:
        return f"인자 확인 실패: {res}"
    replace = [str(n) for n in r.get("replace") or []]
    busy = [n for n in plan_runs() if n not in replace]
    if busy:
        return f"GPU 사용 중: {', '.join(busy)} -- 시작하지 않음"
    for n in replace:
        k, h = stop_run(n)
        log(f"멈춤 {n} (학습 {k}, 보조 {h}) -- {name} 시작 전")
    if plan_runs():
        return f"GPU 사용 중: {', '.join(plan_runs())} -- 멈추지 못함"
    env = dict(os.environ, EVAL=("lab3" if r.get("exam", "lab3") == "lab3" else "none"), LAUNCH_GPU=gpu)
    env.pop("CHECK", None)
    with open(os.path.join(OUT, f"train_queue_{name}.log"), "w") as lf:
        p = subprocess.run(scope_prefix(name) + cmd, cwd=RT, env=env, stdout=lf, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, start_new_session=True, timeout=300)
    if p.returncode != 0:
        return f"launch_run.sh 실패 (rc {p.returncode}, out/train_queue_{name}.log)"
    for _ in range(6):
        if name in plan_runs():
            break
        time.sleep(5)
    if name not in plan_runs():
        return f"띄웠지만 바로 죽음 (out/{name}.log)"
    jobs = load(JOBS, {})
    jobs[name] = {"started": time.time(), "by": r.get("_from", ""), "world": r["world"], "init": init}
    save(JOBS, jobs)
    return "시작함" + (f" ({', '.join(replace)} 멈춤)" if replace else "")


def act(r, src):
    try:
        TR.check_request(r)
    except (ValueError, TypeError, KeyError) as e:
        return f"잘못된 요청: {e}"
    try:
        if r["action"] == "stop":
            if r["name"] not in plan_runs():
                return "돌고 있지 않음"
            k, h = stop_run(r["name"])
            return f"멈춤 (학습 프로세스 {k}개, 보조 {h}개)"
        return start_run(dict(r, _from=src))
    except Exception as e:                                       # noqa: BLE001 -- the reason goes to the page
        return f"오류: {type(e).__name__}: {e}"


# ------------------------------------------------------------------ status
def last_progress(name):
    p = os.path.join(RUNS, name, "progress.jsonl")
    try:
        with open(p, "rb") as f:
            f.seek(max(0, os.path.getsize(p) - 20000))
            rows = [l for l in f.read().decode(errors="replace").splitlines() if l.startswith("{")]
        r = json.loads(rows[-1])
    except (OSError, ValueError, IndexError):
        return {}
    std = r.get("std") or []
    return {"update": r.get("update"), "laps_per_min": r.get("laps_per_min"), "speed": r.get("speed"),
            "walls_per_min": r.get("contacts_wall_per_min"), "boxes_per_min": r.get("contacts_prop_per_min"),
            "steer_std": std[0] if len(std) > 10 else None, "speed_std": std[10] if len(std) > 10 else None,
            "at": os.path.getmtime(p)}


NUM = r"([0-9]+(?:\.[0-9]+)?)"
#: the box table's rows (tools/box_passes.py): seen this long before passing, and which pass of the same box
BOX_ROWS = (("0.5-0.9 s 전", "seen_05"), ("0.9-1.5 s 전", "seen_09"), ("1.5-3.0 s 전", "seen_15"),
            ("첫째", "pass1"), ("둘째", "pass2"), ("셋째", "pass3"), ("넷째 이후", "pass4"))


def parse_exam(txt, boxes_txt=""):
    """one checkpoint's exam row: the empty-track lap, the box sessions' hard contacts per session, and from the box
    table [passes, hard-contact %] per seen-time row and per pass of the same box"""
    row = {}
    mm = re.search(rf"한 바퀴\(세션 중앙값\) {NUM} s", txt)
    row["lap"] = float(mm.group(1)) if mm else None
    blk = txt.split("[혼자 · 상자]", 1)
    if len(blk) == 2:
        mm = re.search(rf"세션당 세게 닿음: 벽 {NUM} \[[^]]*\] · 상자 {NUM}", blk[1])
        if mm:
            row["box_walls"], row["box_boxes"] = float(mm.group(1)), float(mm.group(2))
        mm = re.search(rf"한 바퀴\(세션 중앙값\) {NUM} s", blk[1])
        if mm:
            row["lap_box"] = float(mm.group(1))
    for key, lab in BOX_ROWS:
        mm = re.search(rf"^\s+{re.escape(key)}\s+(\d+)번 \| 세게 닿음\s+{NUM} %", boxes_txt or txt, re.M)
        if mm:
            row[lab] = [int(mm.group(1)), float(mm.group(2))]
    return row


def exam_rows(name, adir=None):
    adir = adir or os.path.join(OUT, "analysis")
    out = []
    for p in glob.glob(os.path.join(adir, f"{name}_u*_exam[23].txt")):
        m = re.search(rf"{re.escape(name)}_u(\d+)_(exam[23])\.txt$", p)
        if not m:
            continue
        bp = os.path.join(adir, f"{name}_u{m.group(1)}_boxes.txt")
        with open(p, errors="replace") as f:
            txt = f.read()
        bt = ""
        if os.path.exists(bp):
            with open(bp, errors="replace") as f:
                bt = f.read()
        out.append(dict(parse_exam(txt, bt), u=int(m.group(1)), exam=m.group(2)))
    return sorted(out, key=lambda r: r["u"])


_BASE = {"key": None, "val": None}


def baseline():
    """what launch_run.sh gives plan_rl with no extra flags, parsed by plan_rl itself (cached until either changes):
    the presets show only the flags a run differs by"""
    key = tuple(os.path.getmtime(p) for p in (LAUNCH, PLAN_RL))
    if _BASE["key"] != key:
        world = sorted(glob.glob(os.path.join(CFG, "plan_world_*.json")))[0]
        ok, res = check_args(launch_argv(BASE_NAME, world, 100, "", []), "both")
        _BASE["key"], _BASE["val"] = key, (res if ok else None)
        if not ok:
            log(f"기준 인자 확인 실패: {res}")
    return _BASE["val"]


def knob_defaults(base):
    out = {}
    for f, *_ in TR.KNOBS:
        k = f[2:].replace("-", "_")
        if base and k in base["args"]:
            v = base["args"][k]
            out[f] = "맵 설정값" if (f == "--removal-p" and v is not None and float(v) < 0) else ("" if v is None else str(v))
    return out


def build_status(done):
    alive = plan_runs()
    jobs = load(JOBS, {})
    names = set(alive) | {n for n, j in jobs.items() if time.time() - j.get("started", 0) < 7 * 86400}
    recent = sorted(glob.glob(os.path.join(RUNS, "*", "progress.jsonl")), key=os.path.getmtime, reverse=True)[:6]
    names |= {os.path.basename(os.path.dirname(p)) for p in recent}
    base = baseline()
    bargs, bopt = (base["args"], base["opt"]) if base else (None, None)
    runs = {}
    for n in sorted(names):
        if not TR.NAME_RE.match(n) or not os.path.isdir(os.path.join(RUNS, n)):
            continue
        a = TR.run_args(n, RUNS)
        init = str(a.get("init") or "")
        runs[n] = {"alive": n in alive, "progress": last_progress(n), "checkpoints": TR.checkpoints(n, RUNS),
                   "exams": exam_rows(n), "world": os.path.basename(str(a.get("world") or "")),
                   "init": os.path.relpath(init, RUNS) if init.startswith(RUNS) else init,
                   "flags": TR.preset(n, "scratch", RUNS, bargs, bopt).get("flags", []) if a else [],
                   "updates": a.get("updates"), "steer_floor": a.get("steer_std_floor")}
    by_age = sorted(runs, key=lambda n: (runs[n]["alive"], (runs[n]["progress"] or {}).get("at", 0)), reverse=True)
    presets = []
    for label, spec in load(PRESETS, {}).items():
        src = spec.get("from")
        if src == "@alive":
            src = sorted(alive)[0] if alive else (by_age[0] if by_age else None)
        pr = TR.preset(src, spec.get("mode", "scratch"), RUNS, bargs, bopt) if src else {}
        if not pr:
            continue
        sets = {str(f): str(v) for f, v in (spec.get("set") or {}).items()}
        fl = [(f, v) for f, v in pr["flags"] if f not in sets] + list(sets.items())
        presets.append(dict(pr, label=label.replace("@alive", src), flags=fl))
    worlds = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CFG, "plan_world_*.json")))
    taken = sorted(n for n in os.listdir(RUNS) if TR.NAME_RE.match(n))
    return {"at": time.time(), "host": os.uname().nodename, "runs": runs, "order": by_age, "presets": presets,
            "worlds": worlds, "busy": sorted(alive), "taken": taken,
            "next": TR.next_name(by_age[0], RUNS, taken) if by_age else "", "knob_defaults": knob_defaults(base),
            "done": done[-20:]}


def push_lab(st):
    """the status, and each reported run's progress.jsonl and args.json, to the lab's runs folder (one tar stream)"""
    files = ["_train_status.json"]
    for n in st["runs"]:
        for f in ("progress.jsonl", "args.json"):
            if os.path.exists(os.path.join(RUNS, n, f)):
                files.append(f"{n}/{f}")
    try:
        tar = subprocess.Popen(["tar", "-cf", "-", "-C", RUNS] + files, stdout=subprocess.PIPE)
        r = subprocess.run(SSH + [LAB, "mkdir -p ~/f1sim_runs && tar -x -C ~/f1sim_runs"], stdin=tar.stdout, timeout=120,
                           capture_output=True)
        tar.stdout.close()
        tar.wait(timeout=30)
        if r.returncode != 0:
            log(f"연구실로 보내기 실패: rc {r.returncode} {r.stderr.decode(errors='replace').strip()[-200:]}")
    except (subprocess.SubprocessError, OSError) as e:
        log(f"연구실로 보내기 실패: {e}")


def main():
    once = "--once" in sys.argv
    no_lab = "--no-lab" in sys.argv
    done = load(TR.status_path(RUNS), {}).get("done", [])
    t_lab = t_push = 0.0
    log(f"시작 (pid {os.getpid()})")
    while True:
        try:
            reqs = local_requests()
            if not no_lab and time.time() - t_lab >= 15:
                reqs += lab_requests()
                t_lab = time.time()
            for r, src in reqs:
                if not isinstance(r, dict):
                    done.append({"at": time.time(), "from": src, "result": "읽을 수 없는 요청 파일"})
                    continue
                res = act(r, src)
                log(f"{src} {r.get('action')} {r.get('name')}: {res}")
                done.append({"at": time.time(), "req_at": r.get("at"), "from": src, "action": r.get("action"),
                             "name": r.get("name"), "result": res})
            done = done[-50:]
            if reqs or once or time.time() - t_push >= 60:
                st = build_status(done)
                save(TR.status_path(RUNS), st)
                if not no_lab:
                    push_lab(st)
                t_push = time.time()
        except Exception as e:                                   # noqa: BLE001 -- keep serving; the log says why
            log(f"오류: {type(e).__name__}: {e}")
            if once:
                raise
        if once:
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
