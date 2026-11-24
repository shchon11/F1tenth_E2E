"""학습 현황 사이트 (11-14, 사용자: "실시간 학습 진행도나 평가들 모아놓은 사이트좀 tailscale로좀 만들어조").
Tailscale 주소에만 열림(공개 인터넷 X). 페이지는 30 s마다 스스로 새로 읽음.
  - 학습 진행: run의 progress.jsonl (rank 0) -- 바퀴/분, 속도, 접촉, 학습 보상, KL, update 시간; 비교 run을 겹쳐 그림
  - 학습 표: plan_table.py (칸 5개)
  - checkpoint마다: frozen exam 결과(analysis/RUN_uU_exam3.txt, 없으면 exam2/exam), 경주 시험(exam4, 시작 checkpoint와 함께), 비교 run 같은 update와 세션끼리 짝지은 차(grad_step_table.py, 캐시),
    네 경우 영상(videos/), 기억 비교 요약, 추세 그림
  - 그림 모음: out/netaudit, out/drift, out/edge*, out/rwexam, out/batchdump (최근 것부터)
  - 연구 기록: out/research_log.md (날짜별 분석·결정, 최근 것부터) · run 상태 메모: out/run_notes.json {run: 멈춘 이유 등}
  - 이어 시작한 run(--init ...plan_uN.pt): 학습 표에 부모 run의 u<=N 행을 '이어 받음'으로 채움
실행: python3 dash_server.py [주소 100.99.253.65] [포트 8765]"""
import gzip
import json, os, re, subprocess, sys, time, threading, html, glob
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote

R = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01"
OUT, TOOLS, RUNS = R + "/out", R + "/tools", "/home/shchon11/f1sim_runs"
PY = "/home/shchon11/F1tenth_E2E/.venv/bin/python3"
HOST = sys.argv[1] if len(sys.argv) > 1 else "100.99.253.65"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
CACHE = OUT + "/dash_cache"
os.makedirs(CACHE, exist_ok=True)
FIELDS = ("update", "min", "t_roll", "t_upd", "laps_per_min", "speed", "contacts_wall_per_min", "contacts_prop_per_min",
          "contacts_car_per_min", "reward", "contact_cost", "kl", "clipfrac", "vloss", "accum_actor", "stopped", "noise_rho_steer",
          "plan_hit_share", "tail_next_05", "tail_driven_10", "vest_mae",      # (11-16, the user: "plan관련 지표는 못보여주나?")
          "noise_scale", "aux_auc",       # (11-19, the user: "잡음 크기는 사이트에 plot 안되나?"; + the box memory's side task)
          "car_fault_per_min", "near_miss_per_min", "pred_err_05", "pred_err_10",   # (11-24) the race phase (1v1)
          "plan_car_share", "plan_car_share_lt07", "plan_car_share_07_09", "plan_car_share_ge09")  # (11-25) --plan-car-cost
TYPES = {".png": "image/png", ".mp4": "video/mp4", ".txt": "text/plain; charset=utf-8", ".log": "text/plain; charset=utf-8",
         ".json": "application/json", ".jpg": "image/jpeg", ".md": "text/plain; charset=utf-8"}
_memo, _lock = {}, threading.Lock()


def memo(key, ttl, fn):
    now = time.time()
    with _lock:
        hit = _memo.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _memo[key] = (now, val)
    return val


def run_notes():
    try:
        return json.load(open(f"{OUT}/run_notes.json"))
    except Exception:
        return {}


def parent(name):
    """(run, update) this run was initialised from (--init .../RUN/plan_uN.pt), else (None, None)"""
    try:
        ini = json.load(open(f"{RUNS}/{name}/args.json"))["args"].get("init") or ""
    except Exception:
        return None, None
    m = re.search(r"/([^/]+)/plan_u(\d+)\.pt$", ini)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def is_plan_run(name):
    """a planner run (plan_rl; 11-15: any name -- kr26a, kr26b, ... -- not only plan2*), not a console watch folder"""
    if name.startswith("watch_"):
        return False
    if name.startswith("plan2"):
        return True
    try:
        return json.load(open(f"{RUNS}/{name}/args.json"))["args"].get("format") in ("v2", "v3")
    except Exception:
        return False


def runs():
    out = []
    ps = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
    notes = run_notes()
    # (11-21) the run each run is judged against (out/run_refs.json {run: control}); the page's default comparison
    try:
        refs = json.load(open(f"{OUT}/run_refs.json"))
    except (OSError, ValueError):
        refs = {}
    for p in glob.glob(RUNS + "/*/progress.jsonl"):
        name = p.split("/")[-2]
        if not memo(("isplan", name), 3600, lambda: is_plan_run(name)):
            continue
        par, cut = parent(name)
        out.append({"name": name, "mtime": os.path.getmtime(p), "map": run_map(name), "alive": f"plan_rl --name {name} " in ps,
                    "parent": par, "cut": cut, "note": notes.get(name, ""), "ref": refs.get(name, "")})
    out.sort(key=lambda r: (not r["alive"], -r["mtime"]))       # the runs training now first
    return out


def overview():
    """every run training now: its status and its newest progress row (the page's top strip, 11-15)"""
    out = []
    for r in memo("runs", 20, runs):
        if not r["alive"]:
            continue
        rows = memo(("p", r["name"]), 10, lambda n=r["name"]: progress(n))
        out.append({**r, "status": memo(("s", r["name"]), 10, lambda n=r["name"]: status(n)), "last": rows[-1] if rows else None})
    return out


def run_map(name):
    """the map a run trained on (its args.json world; 11-14). Runs from before --world was used are Korea 2025."""
    try:
        w = json.load(open(f"{RUNS}/{name}/args.json"))["args"].get("world") or ""
        return str(json.load(open(w)).get("tracks", "")).split(",")[0].split(":")[-1] or "korea_2025_iccas"
    except Exception:
        return "korea_2025_iccas"


def ok_run(name):
    return bool(re.fullmatch(r"[A-Za-z0-9_]+", name or "")) and os.path.exists(f"{RUNS}/{name}/progress.jsonl")


def progress(name):
    rows = []
    with open(f"{RUNS}/{name}/progress.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            row = {k: (float(f"{r[k]:.5g}") if isinstance(r.get(k), float) else r.get(k)) for k in FIELDS}   # 5 digits: half the bytes
            sd = r.get("std")
            row["std_steer"] = sd[0] if isinstance(sd, list) and sd else None
            rows.append(row)
    return rows


def chain_rows(name):
    """(11-16, the user: "사이트에두") the runs this one was continued from, oldest first, each up to the update it was taken
    at -- drawn before the run's own rows with a dashed line where it resumed. Only a run that numbers its updates on from
    its parent's (--update-offset) and trains on the same map (else the x axis would not line up)."""
    out, cur = [], name
    for _ in range(10):
        par, cut = parent(cur)
        if not par or not ok_run(par) or run_map(par) != run_map(cur):
            break
        own = memo(("p", cur), 10, lambda n=cur: progress(n))
        if not own or (own[0].get("update") or 0) <= cut:
            break
        rows = [r for r in memo(("p", par), 10, lambda n=par: progress(n)) if (r.get("update") or 0) <= cut]
        out.append({"run": par, "cut": cut, "next": cur, "rows": rows})
        cur = par
    return out[::-1]


def status(name):
    rows = progress(name)
    try:
        total = int(json.load(open(f"{RUNS}/{name}/args.json"))["args"]["updates"])
    except Exception:
        total = None
    ps = subprocess.run(["ps", "-eo", "pid,etimes,cmd"], capture_output=True, text=True).stdout.splitlines()
    alive = [l for l in ps if f"plan_rl --name {name} " in l and "bash -c" not in l]
    jobs = [re.sub(r".*/tools/", "", l)[:60] for l in ps if name in l and re.search(r"plan_validate|exam_record|train_video|lap_memory", l)
            and "timeout" not in l and "slot5060" not in l]
    st = {"run": name, "update": rows[-1]["update"] if rows else 0, "total": total, "alive": len(alive), "jobs": jobs,
          "last_mtime": os.path.getmtime(f"{RUNS}/{name}/progress.jsonl")}
    if len(rows) >= 2:
        k = [r for r in rows[-31:] if r["min"] is not None]
        d = sorted((b["min"] - a["min"]) * 60.0 / (b["update"] - a["update"]) for a, b in zip(k, k[1:]) if b["update"] > a["update"])
        if d:
            spu = d[len(d) // 2]                       # the median: an analysis or a measurement pausing the run does not set the ETA
            st["s_per_update"] = spu
            if total:
                st["eta"] = time.time() + spu * (total - rows[-1]["update"])
    return st


def header(name):
    p = f"{TOOLS}/launch_{name}.sh"
    if not os.path.exists(p):
        return ""
    out = []
    for line in open(p).read().splitlines()[1:]:
        if not line.startswith("#"):
            break
        out.append(line[1:].strip())
    return "\n".join(out)


def table_rows(name):
    r = subprocess.run([PY, "../tools/plan_table.py", name], cwd=OUT, capture_output=True, text=True, timeout=120)
    lines = [l for l in r.stdout.splitlines() if l.startswith("|")]
    return [[c.strip() for c in l.strip("|").split("|")] for l in lines if not re.fullmatch(r"\|[-| ]+\|", l.strip())]


def table(name):
    """{rows, inherited, parent, cut}: a run started from a parent's checkpoint gets the parent's rows up to that update (its
    "시작" row is the parent's u<cut>), marked as inherited"""
    rows = table_rows(name)
    par, cut = parent(name)
    inh = []
    if par and rows and os.path.exists(f"{RUNS}/{par}/progress.jsonl"):
        prow = [r for r in table_rows(par)[1:] if re.fullmatch(r"u\d+", r[0]) and int(r[0][1:]) < cut]
        body = [([f"u{cut}"] + r[1:]) if r[0] == "시작" else r for r in rows[1:]]
        inh = [r[0] for r in prow] + [f"u{cut}"]
        rows = [rows[0]] + [r[:len(rows[0])] + [""] * max(0, len(rows[0]) - len(r)) for r in prow] + body
    return {"rows": rows, "inherited": inh, "parent": par, "cut": cut}


def exam_file(name, u):
    """the checkpoint's solo exam text (out/-relative): exam3 (11-23 on), else exam2, else the first exam's"""
    for k in ("exam3", "exam2", "exam"):
        rel = f"analysis/{name}_u{u}_{k}.txt"
        if os.path.exists(f"{OUT}/{rel}"):
            return rel
    return None


RACE_ROWS = ("전체", "상대 0.60배", "상대 0.80배", "상대 0.95배", "상대 제 페이스")


def race_rows(name, u):
    """the 1v1 race exam (analysis/RUN_uU_exam4.txt, tools/exam_race_table.py), overall + by the opponent's pace:
    {condition: {laps, opp_laps, pass, fault, hard_car, stall}} (per 90 s session; stall = % of sessions)"""
    p = f"{OUT}/analysis/{name}_u{u}_exam4.txt"
    if not os.path.exists(p):
        return None
    cols, out = None, {}
    num = lambda s: [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]
    for line in open(p, errors="replace"):
        c = [x.strip() for x in line.split("|")]
        if len(c) < 6:
            continue
        if c[0] == "조건":
            cols = {k: i for i, k in enumerate(c)}
            continue
        if not cols or c[0] not in RACE_ROWS or c[0] in out:
            continue
        try:
            lp = num(c[cols["바퀴 (상대)"]]); hd = num(c[cols["세게 닿음 벽 · 상자 · 차"]])
            out[c[0]] = {"laps": lp[0], "opp_laps": lp[1] if len(lp) > 1 else None,
                         "pass": num(c[cols["앞지름"]])[0], "fault": num(c[cols["우리 책임 차 접촉"]])[0],
                         "hard_car": hd[2] if len(hd) > 2 else None, "stall": num(c[cols["멈춘 세션"]])[0]}
        except (KeyError, IndexError):
            continue
    return out or None


def ckpt_entry(name, u):
    e = exam_file(name, u); v = f"videos/{name}_u{u}_train_4cases.mp4"; t = f"analysis/{name}_u{u}_trend.png"
    e4 = f"analysis/{name}_u{u}_exam4.txt"
    lm = f"{OUT}/analysis/{name}_u{u}_lapmem.log"
    summ = ""
    if os.path.exists(lm):
        s_ = [l for l in open(lm, errors="replace") if l.startswith("요약")]
        summ = s_[-1].strip() if s_ else ""
    return {"u": u, "run": name, "exam": e, "exam_kind": e.rsplit("_", 1)[-1][:-4] if e else None,
            "exam4": e4 if os.path.exists(f"{OUT}/{e4}") else None, "race": race_rows(name, u),
            "video": v if os.path.exists(f"{OUT}/{v}") and os.path.exists(f"{OUT}/videos/{name}_u{u}_train_4cases.done") else None,
            "trend": t if os.path.exists(f"{OUT}/{t}") else None, "lapmem": summ,
            "npz": os.path.exists(f"{OUT}/exam/exam1_{name}_u{u}_ideal_uniform_n256.npz")}


def ckpts(name):
    """the run's checkpoints, newest first; a run started from a checkpoint (--init RUN/plan_uN.pt) gets that start
    checkpoint last, marked `start` (11-24: the race exam is read against it)"""
    us = set()
    for p in glob.glob(f"{OUT}/analysis/{name}_u*") + glob.glob(f"{OUT}/videos/{name}_u*") + glob.glob(f"{OUT}/exam/exam1_{name}_u*_ideal_uniform_n256.npz"):
        m = re.search(rf"{name}_u(\d+)", p)
        if m:
            us.add(int(m.group(1)))
    out = [ckpt_entry(name, u) for u in sorted(us, reverse=True)]
    par, cut = parent(name)
    if par and re.fullmatch(r"[A-Za-z0-9_]+", par):
        s = ckpt_entry(par, cut)
        if s["exam"] or s["exam4"]:
            out.append({**s, "start": True, "npz": False, "video": None, "trend": None, "lapmem": ""})
    return out


def exam_summary(name, u):
    e = exam_file(name, u)
    if not e:
        return ""
    p = f"{OUT}/{e}"
    keep, case = [], None
    for line in open(p, errors="replace"):
        if line.startswith("["):
            case = line.strip(); keep.append(case)
        elif case and ("세션당 세게 닿음" in line or "깨끗한 세션(" in line):
            keep.append("  " + line.strip()[:170])
    return "\n".join(keep)


def compare(name, u, ref, ru=None):
    """`ru`: the reference run's update to pair with (11-15: a run started from another map's checkpoint is compared at the
    same number of updates on this map, e.g. kr26b u925 with kr26a u25)"""
    ru = u if ru is None else ru
    a = f"{OUT}/exam/exam1_{ref}_u{ru}_ideal_uniform_n256.npz"; b = f"{OUT}/exam/exam1_{name}_u{u}_ideal_uniform_n256.npz"
    if not (os.path.exists(a) and os.path.exists(b)):
        return {"text": f"{ref} u{ru} 또는 {name} u{u}의 frozen exam 기록이 없음"}
    ma, mb = run_map(ref), run_map(name)
    if ma != mb:                                           # 11-14: sessions are paired by seed; another map's run is no comparison
        return {"text": f"{ref}는 {ma}, {name}는 {mb}에서 학습 -- 다른 지도끼리는 같은 세션 비교가 안 됨"}
    c = f"{CACHE}/{name}_u{u}_vs_{ref}" + (f"_u{ru}" if ru != u else "") + ".txt"
    if not os.path.exists(c) or os.path.getmtime(c) < max(os.path.getmtime(a), os.path.getmtime(b)):
        r = subprocess.run([PY, "-W", "ignore", "grad_step_table.py", a, f"{name} u{u}={b}"], cwd=TOOLS, capture_output=True,
                           text=True, timeout=300)
        open(c, "w").write(r.stdout[-3000:] if r.returncode == 0 else "계산 실패\n" + r.stderr[-1500:])
    return {"text": open(c).read()}


EDGE = (("거의 안 보인 상자 (0.5 s 안에 보임)", "0.5 s 안 (거의 안 보임)"),
        ("늦게 보인 상자 (0.5-0.9 s 전에 보임)", "0.5-0.9 s 전"), ("보통 (0.9-1.5 s 전에 보임)", "0.9-1.5 s 전"),
        ("일찍 보인 상자 (1.5-3 s 전에 보임)", "1.5-3.0 s 전"), ("아주 일찍 보인 상자 (3 s 넘게 전)", "3 s 넘게 전부터"),
        ("같은 상자 첫째 통과", "첫째"), ("같은 상자 넷째 통과 이후", "넷째 이후"),
        ("지난번에 세게 닿은 상자", "지난번에 세게 닿음"), ("지난번 깨끗했던 상자", "지난번에 안 닿거나 약하게"),
        ("급하게 줄인 통과 (1 s에 1.5 m/s 넘게)", "1.5 m/s 넘게 (급하게)"))


def edge_one(name, u):
    """{situation: [passes, hard %]} from analysis/RUN_uU_boxes.txt (boxes on the driven path), + start / recovery from the exam text"""
    out = {}
    p = f"{OUT}/analysis/{name}_u{u}_boxes.txt"
    if os.path.exists(p):
        txt = open(p, errors="replace").read()
        for lab, key in EDGE:
            m = re.search(r"^\s+" + re.escape(key) + r"\s+(\d+)번 \| 세게 닿음\s+([\d.]+) %", txt, re.M)
            if m:
                out[lab] = [int(m.group(1)), float(m.group(2))]
    e = f"{OUT}/{exam_file(name, u)}" if exam_file(name, u) else ""
    if e:
        case = None
        for line in open(e, errors="replace"):
            if line.startswith("["):
                case = line.split("]")[0][1:]
            m = re.search(r"출발\(처음 5 s\)에 세게 닿은 세션\s+([\d.]+) %", line)
            if m and case == "상대 차 · 상자":
                out["출발 5 s 안에 세게 닿은 세션 (상대 차 · 상자)"] = [None, float(m.group(1))]
            m = re.search(r"세게 닿은 뒤\((\d+)건\): 5 s 안에 선 것\s+([\d.]+) %", line)
            if m and case == "혼자 · 상자":
                out["상자에 세게 닿은 뒤 5 s 안에 선 것 (혼자 · 상자)"] = [int(m.group(1)), float(m.group(2))]
            m2 = re.search(r"5 s 안에 다시 못 달린 것\s+([\d.]+) %", line)
            if m and m2 and case in ("혼자 · 상자", "상대 차 · 상자"):
                k_ = f"선 뒤 5 s 안에 다시 못 달린 것 ({case})"
                out[k_] = [None, float(m2.group(1))]
    return out


def edge(name, ref, shift=0):
    us = sorted({int(m.group(1)) for p in glob.glob(f"{OUT}/analysis/{name}_u*_boxes.txt")
                 for m in [re.search(rf"{name}_u(\d+)_boxes", p)] if m}, reverse=True)
    return {"labels": [l for l, _ in EDGE] + ["출발 5 s 안에 세게 닿은 세션 (상대 차 · 상자)", "상자에 세게 닿은 뒤 5 s 안에 선 것 (혼자 · 상자)",
                                              "선 뒤 5 s 안에 다시 못 달린 것 (혼자 · 상자)", "선 뒤 5 s 안에 다시 못 달린 것 (상대 차 · 상자)"],
            "rows": [{"u": u, "ru": u - shift, "run": edge_one(name, u), "ref": edge_one(ref, u - shift) if ref else {}} for u in us]}


def figs():
    """the newest analysis figures of every out/ folder (11-15: not a fixed list -- mapeta, maptune, ... appeared); the per-
    checkpoint trend plots stay with their checkpoint"""
    out = []
    for p in glob.glob(f"{OUT}/*/*.png"):
        d = os.path.basename(os.path.dirname(p))
        if d in ("analysis", "dash_cache", "videos"):
            continue
        out.append({"path": f"{d}/{os.path.basename(p)}", "dir": d, "mtime": os.path.getmtime(p)})
    out.sort(key=lambda r: -r["mtime"])
    return out[:60]



def live_meta(name):
    """the newest progress row's `horizon` (decisions a rollout) and the run's episode lengths [s]"""
    d = os.path.join(RUNS, os.path.basename(name))
    out = {}
    try:
        with open(os.path.join(d, "progress.jsonl"), "rb") as f:
            f.seek(max(0, os.path.getsize(f.name) - 65536))
            last = [ln for ln in f.read().splitlines() if ln.strip()][-1]
        out["horizon"] = int(json.loads(last).get("horizon") or 0) or None
    except Exception:
        pass
    try:
        a = json.load(open(os.path.join(d, "args.json")))
        w, ar = a.get("world") or {}, a.get("args") or {}
        out["episode_s"] = w.get("episode_s") or ar.get("episode_s")
        out["scene_episode_s"] = ar.get("scene_episode_s") or w.get("scene_episode_s")
    except Exception:
        pass
    return out


def spots_runs():
    """(11-20, the user: "쭈르르륵 드래그하면 어디박았는지 볼 수 있게") runs with exam collision maps (tools/spots_frames.py), newest first"""
    d = os.path.join(OUT, "spots")
    out = []
    for r in os.listdir(d) if os.path.isdir(d) else []:
        p = os.path.join(d, r, "index.json")
        if os.path.isfile(p):
            out.append({"name": r, "age": time.time() - os.path.getmtime(p)})
    return sorted(out, key=lambda r: r["age"])


def spots(name):
    """a run's exam collision maps, the runs it was continued from first (each up to the update it was taken at)"""
    from run_chain import parent
    frames, cur, cut = [], name, None
    for _ in range(10):
        p = os.path.join(OUT, "spots", cur, "index.json")
        rows = json.load(open(p)) if os.path.isfile(p) else []
        frames = [r for r in rows if cut is None or r["u"] <= cut] + frames
        par, c = parent(cur)
        if not par:
            break
        cur, cut = par, c
    return {"frames": frames}


def live_cars(name):
    """(11-22, the user: "이차밖에 못보나? 다른 차들을 볼 수 없나") the cars a run writes: 0 = live_tap.json, k = live_tap_<k>.json"""
    d = os.path.join(RUNS, os.path.basename(name))
    out = []
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        m = re.fullmatch(r"live_tap(?:_(\d+))?\.json", f)
        if not m:
            continue
        try:
            lab = json.load(open(os.path.join(d, f))).get("now", {}).get("label", "")
        except (ValueError, OSError):
            lab = ""
        out.append({"car": int(m.group(1) or 0), "label": lab})
    return {"cars": sorted(out, key=lambda c: c["car"])}


def live_runs():
    """runs that write the live view, newest first"""
    out = []
    for d in os.listdir(RUNS):
        p = os.path.join(RUNS, d, "live_tap.json")
        if os.path.isfile(p):
            out.append({"name": d, "age": time.time() - os.path.getmtime(p)})
    return sorted(out, key=lambda r: r["age"])


def live_track(name, tid):
    """one track of a live run: its walls as a PNG (data URL) and where the image sits; the box footprints by shape id"""
    import base64, io
    import numpy as np
    from PIL import Image
    z = np.load(os.path.join(RUNS, os.path.basename(name), "live_tracks.npz"))
    off, H, W = int(z["t_off"][tid]), int(z["t_H"][tid]), int(z["t_W"][tid])
    res = float(np.asarray(z["t_res"]).reshape(-1)[tid] if np.asarray(z["t_res"]).size > 1 else np.asarray(z["t_res"]).reshape(-1)[0])
    org = np.asarray(z["t_origin"]).reshape(-1, 2)
    org = org[tid] if len(org) > tid else org[0]
    g = z["edt"][off:off + H * W].reshape(H, W)
    # (11-18, the user: "그 폭이 장애물이나 벽에 보이게끔 ... plan이 닿는걸 좀 보게") the car's footprint (the plan-hit cost's:
    # `sim.corners`, symmetric about the CoG) and, on the image, the band within half its width of a wall: lane 246, band 160,
    # wall 70 -- a car centre in the band has its side in the wall
    car = np.asarray(z["car"], float).tolist() if "car" in z.files else [0.58, 0.31]
    img = np.where(g <= 0, 70, np.where(g <= 0.5 * car[1], 160, 246)).astype(np.uint8)[::-1]   # row 0 = top (largest y)
    buf = io.BytesIO(); Image.fromarray(img, "L").save(buf, "PNG")
    fps = {k[3:]: np.asarray(z[k]).round(3).tolist() for k in z.files if k.startswith("fp_")}
    # how far into a plan the run's plan-hit cost looks (boxes / walls) and its weight (0 = off: drawn all the same)
    a = {}
    try:
        a = json.load(open(os.path.join(RUNS, os.path.basename(name), "args.json"))).get("args", {})
    except Exception:
        pass
    hit_s = float(a.get("plan_hit_s") or 1.0)
    hit_wall_s = float(a["plan_hit_wall_s"]) if a.get("plan_hit_wall_s") is not None else hit_s
    return {"png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "x0": float(org[0]),
            "y0": float(org[1]), "res": res, "W": W, "H": H, "fp": fps, "car": car,
            "hit_s": hit_s, "hit_wall_s": min(hit_wall_s, hit_s), "hit_share": float(a.get("plan_hit_share") or 0.0)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode()
        # (11-16, the user: "불러오는데 시간이 5초정도 걸리는거 같은데") the progress of a run with its earlier runs is ~1 MB of
        # JSON a request, over Tailscale; it compresses ~8x -- the server answers in < 0.2 s, the transfer was the wait
        gz = (len(body) > 2048 and ("json" in ctype or "text" in ctype or "html" in ctype)
              and "gzip" in (self.headers.get("Accept-Encoding") or ""))
        if gz:
            body = gzip.compress(body, 5)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if gz:
            self.send_header("Content-Encoding", "gzip"); self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj, ensure_ascii=False))

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        name = q.get("run", "")
        try:
            if u.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if u.path == "/api/sleep":            # holds a page's load event (headless screenshots after the data arrived)
                time.sleep(min(10.0, float(q.get("s", "1"))))
                return self._send(200, b"GIF89a\x01\x00\x01\x00\x00\x00\x00;", "image/gif")
            if u.path == "/api/runs":
                return self._json(memo("runs", 20, runs))
            if u.path == "/api/overview":
                return self._json(memo("overview", 10, overview))
            if u.path == "/api/notes":            # the research log (11-15, the user: "사이트에 제대로좀 반영좀")
                p_ = f"{OUT}/research_log.md"
                return self._json({"text": open(p_, errors="replace").read() if os.path.exists(p_) else "",
                                   "mtime": os.path.getmtime(p_) if os.path.exists(p_) else None})
            if u.path == "/api/live_runs":
                return self._json(memo("live_runs", 5, live_runs))
            if u.path == "/api/spots_runs":
                return self._json(memo("spots_runs", 20, spots_runs))
            if u.path == "/api/spots":
                if not re.fullmatch(r"[A-Za-z0-9_]+", name or ""):
                    return self._send(404, json.dumps({"error": "no such run"}))
                return self._json(memo(("spots", name), 20, lambda: spots(name)))
            if u.path.startswith("/api/") and u.path not in ("/api/figs", "/api/notes", "/api/overview") and not ok_run(name):
                return self._send(404, json.dumps({"error": "no such run"}))
            if u.path == "/api/progress":
                return self._json({"rows": memo(("p", name), 10, lambda: progress(name)), "status": memo(("s", name), 10, lambda: status(name)),
                                   "chain": memo(("chain", name), 30, lambda: chain_rows(name))})
            if u.path == "/api/table":
                return self._json(memo(("t", name), 60, lambda: table(name)))
            if u.path == "/api/header":
                return self._json({"text": header(name)})
            if u.path == "/api/ckpts":
                cs = memo(("c", name), 30, lambda: ckpts(name))
                for c in cs:
                    c["summary"] = memo(("e", c["run"], c["u"]), 60, lambda c=c: exam_summary(c["run"], c["u"]))
                return self._json(cs)
            if u.path == "/api/compare":
                ref = q.get("ref", "")
                if not ok_run(ref):
                    return self._send(404, json.dumps({"error": "no such ref"}))
                return self._json(compare(name, int(q.get("u", "0")), ref, int(q["ru"]) if q.get("ru") else None))
            if u.path == "/api/edge":
                ref = q.get("ref", "")
                sh = int(q.get("shift", "0") or 0)
                return self._json(memo(("edge", name, ref, sh), 60, lambda: edge(name, ref if ok_run(ref) else "", sh)))
            if u.path == "/api/figs":
                return self._json(memo("figs", 60, figs))
            if u.path == "/file":
                return self._file(q.get("path", ""))
            # (11-17, the user: "학습중인 차 ... 실시간으로", "네트워크 반응이나 기억값 ... 재밌게") the training's own car
            if u.path == "/live":
                return self._send(200, open(os.path.join(TOOLS, "live_page.html"), "rb").read(), "text/html; charset=utf-8")
            if u.path == "/api/live":
                return self._live(name, q.get("car", "0"))
            if u.path == "/api/live_cars":
                return self._json(memo(("live_cars", name), 5, lambda: live_cars(name)))
            if u.path == "/api/live_track":
                return self._json(memo(("live_track", name, q.get("tid", "0")), 3600, lambda: live_track(name, int(q.get("tid", "0")))))
            if u.path == "/api/live_runs":
                return self._json(memo("live_runs", 5, live_runs))
        except Exception as exc:                       # one bad file must not take the page down
            return self._send(500, json.dumps({"error": str(exc)[:300]}))
        self._send(404, "{}")

    def _live(self, name, car="0"):
        k = int(car) if str(car).isdigit() else 0
        p = os.path.join(RUNS, os.path.basename(name), "live_tap.json" if k == 0 else f"live_tap_{k}.json")
        if not name or not os.path.isfile(p):
            return self._send(404, json.dumps({"error": "이 run에는 실시간 기록이 없습니다 (plan_rl --live-tap)"}))
        body = open(p, "rb").read()
        # (11-17, the user: "update에 필요한 결정 수 비율이랑 현재 에피소드 진행 비율만 상단에") the rollout's length (the last
        # update's `horizon`, which --horizon-auto changes) and the episode lengths (`episode_s`, a replayed scene's `scene_episode_s`)
        meta = memo(("live_meta", name), 10, lambda: live_meta(name))
        return self._send(200, body, extra={"X-Age": f"{time.time() - os.path.getmtime(p):.2f}",
                                            "X-Horizon": str(meta.get("horizon") or ""), "X-Episode-S": str(meta.get("episode_s") or ""),
                                            "X-Scene-S": str(meta.get("scene_episode_s") or "")})

    def _file(self, rel):
        p = os.path.realpath(os.path.join(OUT, rel))
        ext = os.path.splitext(p)[1].lower()
        if not p.startswith(OUT + "/") or ext not in TYPES or not os.path.isfile(p):
            return self._send(404, "{}")
        size = os.path.getsize(p)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m:
            if m.group(1):
                start = int(m.group(1)); end = int(m.group(2)) if m.group(2) else size - 1
            elif m.group(2):
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
        with open(p, "rb") as f:
            f.seek(start)
            data = f.read(end - start + 1)
        extra = {"Accept-Ranges": "bytes"}
        if m:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            return self._send(206, data, TYPES[ext], extra)
        self._send(200, data, TYPES[ext], extra)


PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dash_page.html"), encoding="utf-8").read()

if __name__ == "__main__":
    srv = ThreadingHTTPServer((HOST, PORT), H)
    print(f"학습 현황 사이트: http://{HOST}:{PORT}/", flush=True)
    srv.serve_forever()
