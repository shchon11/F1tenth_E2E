"""The console's training requests and the training PC's answer, as files (11-24, the user: "콘솔에서도 내가 학습할 수 있게
... 학습 패널로 퀄리나 지금 학습세팅들 좀 만질 수 있게", and where it trains: "여가서 학습" = the two-GPU PC).

Torch-free: the console page imports it, and so does the training PC's queue (tools/train_queue.py).

  - a request: ``<runs>/_train_requests/<stamp>_<action>_<name>.json`` -- ``{"action": "start" | "stop", "name": ...,
    "world": <configs file name>, "init": <run>/<plan_uN.pt> or "", "updates": N, "gpu": "both" | "0" | "1",
    "exam": "lab3" | "none", "replace": [names to stop first], "flags": [[flag, value], ...], "at": time}``
  - the answer: ``<runs>/_train_status.json`` -- what runs, how far, the exam rows per checkpoint, and what became of
    every request (``done``)

The training PC takes requests from its own folder and from the lab PC's over the ssh it already uses (it reaches the
lab, the lab cannot reach it), so the console can run on either; the status goes back the same way. A request names
plan_rl flags and values only -- both checked against a strict pattern here and again where they are run -- never a
command line."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
FLAG_RE = re.compile(r"^--[a-z][a-z0-9-]{1,40}$")
VALUE_RE = re.compile(r"^[A-Za-z0-9._,:/+-]{0,200}$")
WORLD_RE = re.compile(r"^plan_world_[A-Za-z0-9_]+\.json$")
INIT_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}/plan_u[0-9]+\.pt$")

#: The settings the page shows as fields (the rest of a preset rides along as "그 밖의 인자"): (flag, label, kind, hint)
KNOBS: Tuple[Tuple[str, str, str, str], ...] = (
    ("--steer-std-floor", "조향 탐색 바닥", "float", "첫 걸음 조향을 흔드는 폭의 최소값. 줄어들면 라인이 굳음 (scan31b 0.3)"),
    ("--speed-std-floor", "속도 탐색 바닥", "float", "첫 걸음 속도를 흔드는 폭의 최소값 (0.4)"),
    ("--online-boxes", "자기 길 위 상자 비율", "float", "상자 있는 판 중 상자를 자기가 달리는 길 위에 먼저 놓는 비율 (0.5)"),
    ("--online-late", "늦게 보이는 자리 우선 [m]", "float", "그중 이 거리 안에서야 보이는 자리(코너 뒤)를 가장 먼저 (4, 0 = 끔)"),
    ("--removal-p", "상자 치우는 판 비율", "float", "상자 있는 판 중 도중에 상자를 치우는 비율 (0 = 판 끝까지 둠)"),
    ("--shaping-fade", "진행 보상 끄는 구간", "str", "update a,b: a까지 진행 보상, b에서 0 -- 그 뒤 한 바퀴 시간 + 접촉 벌점만"),
    ("--scene-replay", "세게 박은 장면 다시 하기", "float", "판 시작의 이 비율을 세게 박기 직전 장면에서 (0.25)"),
    ("--lr", "학습률", "float", "처음 값 (끝 값은 --lr-end)"),
)

#: args.json keys the launcher sets itself (or that only mean something inside one process)
_NOT_FLAGS = {"name", "world", "envs", "updates", "init", "update_offset", "init_critic", "device", "runs", "seed",
              "dist_world", "dist_rank", "dist_init", "dist_backend", "dist_same_seed", "live_tap", "live_taps",
              "check_args"}
#: flags a request may not carry: the launcher sets them from the request's own fields (name, world, end update,
#: checkpoint, GPUs)
LAUNCHER_FLAGS = {"--" + k.replace("_", "-") for k in _NOT_FLAGS if k not in ("seed",)}
#: flags a run started from a checkpoint must not carry (plan_rl refuses --mem-keep-s with --init; the other two are
#: starting values the checkpoint already holds) -- the queue drops them from a continued run
SCRATCH_ONLY = {"--mem-keep-s", "--mem-keep-log", "--noise-corr-ramp"}


def runs_dir() -> str:
    return os.environ.get("F1SIM_RUNS") or os.path.expanduser("~/f1sim_runs")


def req_dir(runs: Optional[str] = None) -> str:
    return os.path.join(runs or runs_dir(), "_train_requests")


def status_path(runs: Optional[str] = None) -> str:
    return os.path.join(runs or runs_dir(), "_train_status.json")


def check_flags(flags: Sequence[Sequence[str]]) -> List[Tuple[str, str]]:
    """[(flag, value)] as given, or ValueError naming the first that is not a plain flag / value"""
    out = []
    for fv in flags:
        if len(fv) != 2:
            raise ValueError(f"인자 {fv!r}: [이름, 값] 이어야 함")
        f, v = str(fv[0]), str(fv[1])
        if not FLAG_RE.match(f):
            raise ValueError(f"인자 이름 {f!r}가 --소문자-이름 꼴이 아님")
        if not VALUE_RE.match(v):
            raise ValueError(f"{f} 값 {v!r}에 쓸 수 없는 글자가 있음")
        if f in LAUNCHER_FLAGS:
            raise ValueError(f"{f}는 이름·맵·출발·끝 update·GPU 칸으로 정함")
        out.append((f, v))
    return out


def check_request(r: Dict) -> Dict:
    """the request itself when it is well-formed, else ValueError (the queue writes the reason into the status)"""
    act = r.get("action")
    name = str(r.get("name", ""))
    if act not in ("start", "stop"):
        raise ValueError(f"action {act!r}")
    if not NAME_RE.match(name):
        raise ValueError(f"이름 {name!r}: 소문자로 시작, 소문자·숫자·_ 2-31자")
    if act == "stop":
        return r
    if not WORLD_RE.match(str(r.get("world", ""))):
        raise ValueError(f"세계 {r.get('world')!r}")
    init = str(r.get("init") or "")
    if init and not INIT_RE.match(init):
        raise ValueError(f"출발 checkpoint {init!r}")
    if not (1 <= int(r.get("updates", 0)) <= 100000):
        raise ValueError(f"끝 update {r.get('updates')!r}")
    if init and int(r["updates"]) <= init_update(init):
        raise ValueError(f"끝 update {r['updates']}가 출발 u{init_update(init)}보다 커야 함")
    if str(r.get("gpu", "both")) not in ("both", "0", "1"):
        raise ValueError(f"GPU {r.get('gpu')!r}")
    if str(r.get("exam", "lab3")) not in ("lab3", "none"):
        raise ValueError(f"시험 {r.get('exam')!r}")
    for n in r.get("replace") or []:
        if not NAME_RE.match(str(n)):
            raise ValueError(f"멈출 run 이름 {n!r}")
    check_flags(r.get("flags") or [])
    return r


def init_update(init: str) -> int:
    """'scan31a/plan_u325.pt' -> 325"""
    m = re.search(r"plan_u(\d+)\.pt$", init)
    return int(m.group(1)) if m else 0


def write_request(req: Dict, runs: Optional[str] = None) -> str:
    """write one request atomically; returns its path (the queue takes them in name order = time order)"""
    check_request(req)
    d = req_dir(runs)
    os.makedirs(d, exist_ok=True)
    now = float(req.get("at") or time.time())          # the page may stamp it first, to match the queue's answer
    req = dict(req, at=now)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + f"_{int(now * 1e6) % 1000000:06d}"
    p = os.path.join(d, f"{stamp}_{req['action']}_{req['name']}.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(req, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    return p


def read_status(runs: Optional[str] = None) -> Dict:
    try:
        with open(status_path(runs)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def flags_from_args(args: Dict, base: Optional[Dict] = None, opt: Optional[Dict] = None) -> List[Tuple[str, str]]:
    """a run's args.json (its 'args') as plan_rl flags, launcher-owned keys left out (bools as flags when on; off is
    the default of every store_true).

    `base`: what the launcher gives plan_rl with no extra flags (its fixed flags over plan_rl's defaults, as
    ``plan_rl --check-args`` parses them): then only what differs from it -- the run again in the few flags it was
    started with, not 150 -- and keys plan_rl no longer has are dropped. `opt`: dest -> its flag (default --dest)"""
    out = []
    for k, v in args.items():
        if k in _NOT_FLAGS or v is None or v is False or v == "":
            continue
        if base is not None and (k not in base or base[k] == v):
            continue
        f = (opt or {}).get(k) or "--" + k.replace("_", "-")
        if v is True:
            out.append((f, ""))
        elif isinstance(v, (int, float, str)):
            s = str(v)
            if VALUE_RE.match(s):
                out.append((f, s))
    return out


def run_args(name: str, runs: Optional[str] = None) -> Dict:
    p = os.path.join(runs or runs_dir(), name, "args.json")
    try:
        with open(p) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    return d.get("args", d)


def checkpoints(name: str, runs: Optional[str] = None) -> List[int]:
    d = os.path.join(runs or runs_dir(), name)
    try:
        fs = os.listdir(d)
    except OSError:
        return []
    return sorted(int(m.group(1)) for f in fs for m in [re.match(r"^plan_u(\d+)\.pt$", f)] if m)


def world_contact_scale(world_path: str) -> Optional[float]:
    try:
        with open(world_path) as f:
            return float(json.load(f).get("contact_scale", 1.0))
    except (OSError, ValueError, TypeError):
        return None


def preset(name: str, mode: str, runs: Optional[str] = None, base: Optional[Dict] = None,
           opt: Optional[Dict] = None) -> Dict:
    """the page's fields from run `name`: mode "scratch" = its settings from update 0, "continue" = from its last
    checkpoint on (`base`/`opt`: see flags_from_args). {} when the run has no args.json"""
    a = dict(run_args(name, runs))
    if not a:
        return {}
    wp = str(a.get("world") or "")
    if a.get("contact_scale") is not None and a.get("contact_scale") == world_contact_scale(wp):
        a.pop("contact_scale")              # the world's own (plan_rl reads it from the world): another world, its own
    world = os.path.basename(wp)
    out = {"world": world, "updates": int(a.get("updates") or 1500), "flags": flags_from_args(a, base, opt),
           "from": name, "mode": mode}
    if mode == "continue":
        cks = checkpoints(name, runs)
        if cks:
            out["init"] = f"{name}/plan_u{cks[-1]}.pt"
            out["offset"] = cks[-1]
    return out


def next_name(base: str, runs: Optional[str] = None, taken: Optional[Sequence[str]] = None) -> str:
    """the next free name in a series: scan31b -> scan31c (or scan32a after z). Free = no folder in `runs` and not in
    `taken` (the training PC's run names, from the status: the console may run on another PC)"""
    m = re.match(r"^([a-z]+)(\d+)([a-z]?)$", base)
    if not m:
        return base + "x"
    stem, num, suf = m.group(1), int(m.group(2)), m.group(3) or "a"
    r = runs or runs_dir()
    used = set(taken or ())
    for _ in range(60):
        suf = chr(ord(suf) + 1) if suf < "z" else "a"
        if suf == "a":
            num += 1
        n = f"{stem}{num}{suf}"
        if n not in used and not os.path.exists(os.path.join(r, n)):
            return n
    return base + "x"


def flags_text(flags: Sequence[Sequence[str]], sep: str = " ") -> str:
    """[(flag, value)] as the page's text box shows them: '--a 1 --b --c x' (sep "\n": one flag a line)"""
    return sep.join(str(f) + (" " + str(v) if str(v) != "" else "") for f, v in flags)


def parse_flags_text(text: str) -> List[List[str]]:
    """the text box back to [[flag, value]]: a word that is not a --flag is the value of the flag before it (a lone
    flag is a switch). ValueError on a value with no flag, or on anything check_flags refuses"""
    out: List[List[str]] = []
    for w in text.split():
        if w.startswith("--"):
            out.append([w, ""])
        elif out and out[-1][1] == "":
            out[-1][1] = w
        else:
            raise ValueError(f"'{w}' 앞에 --인자 이름이 없음")
    check_flags(out)
    return out


def merge_flags(*groups: Sequence[Sequence[str]]) -> List[List[str]]:
    """later groups win per flag; the first place a flag appeared keeps its order"""
    out: Dict[str, str] = {}
    for g in groups:
        for f, v in g:
            out[str(f)] = str(v)
    return [[f, v] for f, v in out.items()]
