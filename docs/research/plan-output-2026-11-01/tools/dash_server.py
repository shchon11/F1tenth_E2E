"""학습 현황 사이트 (11-14, 사용자: "실시간 학습 진행도나 평가들 모아놓은 사이트좀 tailscale로좀 만들어조").
Tailscale 주소에만 열림(공개 인터넷 X). 페이지는 30 s마다 스스로 새로 읽음.
  - 학습 진행: run의 progress.jsonl (rank 0) -- 바퀴/분, 속도, 접촉, 학습 보상, KL, update 시간; 비교 run을 겹쳐 그림
  - 학습 표: plan_table.py (칸 5개)
  - checkpoint마다: frozen exam 결과(analysis/RUN_uU_exam.txt), 비교 run 같은 update와 세션끼리 짝지은 차(grad_step_table.py, 캐시),
    네 경우 영상(videos/), 기억 비교 요약, 추세 그림
  - 그림 모음: out/netaudit, out/drift (최근 것부터)
실행: python3 dash_server.py [주소 100.99.253.65] [포트 8765]"""
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
          "contacts_car_per_min", "reward", "contact_cost", "kl", "clipfrac", "vloss", "accum_actor", "stopped")
TYPES = {".png": "image/png", ".mp4": "video/mp4", ".txt": "text/plain; charset=utf-8", ".log": "text/plain; charset=utf-8",
         ".json": "application/json", ".jpg": "image/jpeg"}
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


def runs():
    out = []
    for p in glob.glob(RUNS + "/plan2*/progress.jsonl"):
        name = p.split("/")[-2]
        out.append({"name": name, "mtime": os.path.getmtime(p), "map": run_map(name)})
    out.sort(key=lambda r: -r["mtime"])
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
            rows.append({k: r.get(k) for k in FIELDS})
    return rows


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


def table(name):
    r = subprocess.run([PY, "../tools/plan_table.py", name], cwd=OUT, capture_output=True, text=True, timeout=120)
    lines = [l for l in r.stdout.splitlines() if l.startswith("|")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines if not re.fullmatch(r"\|[-| ]+\|", l.strip())]
    return rows


def ckpts(name):
    us = set()
    for p in glob.glob(f"{OUT}/analysis/{name}_u*") + glob.glob(f"{OUT}/videos/{name}_u*") + glob.glob(f"{OUT}/exam/exam1_{name}_u*_ideal_uniform_n256.npz"):
        m = re.search(rf"{name}_u(\d+)", p)
        if m:
            us.add(int(m.group(1)))
    out = []
    for u in sorted(us, reverse=True):
        e = f"analysis/{name}_u{u}_exam.txt"; v = f"videos/{name}_u{u}_train_4cases.mp4"; t = f"analysis/{name}_u{u}_trend.png"
        lm = f"{OUT}/analysis/{name}_u{u}_lapmem.log"
        summ = ""
        if os.path.exists(lm):
            s_ = [l for l in open(lm, errors="replace") if l.startswith("요약")]
            summ = s_[-1].strip() if s_ else ""
        out.append({"u": u, "exam": e if os.path.exists(f"{OUT}/{e}") else None,
                    "video": v if os.path.exists(f"{OUT}/{v}") and os.path.exists(f"{OUT}/videos/{name}_u{u}_train_4cases.done") else None,
                    "trend": t if os.path.exists(f"{OUT}/{t}") else None, "lapmem": summ,
                    "npz": os.path.exists(f"{OUT}/exam/exam1_{name}_u{u}_ideal_uniform_n256.npz")})
    return out


def exam_summary(name, u):
    p = f"{OUT}/analysis/{name}_u{u}_exam.txt"
    if not os.path.exists(p):
        return ""
    keep, case = [], None
    for line in open(p, errors="replace"):
        if line.startswith("["):
            case = line.strip(); keep.append(case)
        elif case and ("세션당 세게 닿음" in line or "깨끗한 세션(" in line):
            keep.append("  " + line.strip()[:170])
    return "\n".join(keep)


def compare(name, u, ref):
    a = f"{OUT}/exam/exam1_{ref}_u{u}_ideal_uniform_n256.npz"; b = f"{OUT}/exam/exam1_{name}_u{u}_ideal_uniform_n256.npz"
    if not (os.path.exists(a) and os.path.exists(b)):
        return {"text": f"{ref} u{u} 또는 {name} u{u}의 frozen exam 기록이 없음"}
    ma, mb = run_map(ref), run_map(name)
    if ma != mb:                                           # 11-14: sessions are paired by seed; another map's run is no comparison
        return {"text": f"{ref}는 {ma}, {name}는 {mb}에서 학습 -- 다른 지도끼리는 같은 세션 비교가 안 됨"}
    c = f"{CACHE}/{name}_u{u}_vs_{ref}.txt"
    if not os.path.exists(c) or os.path.getmtime(c) < max(os.path.getmtime(a), os.path.getmtime(b)):
        r = subprocess.run([PY, "-W", "ignore", "grad_step_table.py", a, f"{name} u{u}={b}"], cwd=TOOLS, capture_output=True,
                           text=True, timeout=300)
        open(c, "w").write(r.stdout[-3000:] if r.returncode == 0 else "계산 실패\n" + r.stderr[-1500:])
    return {"text": open(c).read()}


EDGE = (("늦게 보인 상자 (0.5-0.9 s 전에 보임)", "0.5-0.9 s 전"), ("보통 (0.9-1.5 s 전에 보임)", "0.9-1.5 s 전"),
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
    e = f"{OUT}/analysis/{name}_u{u}_exam.txt"
    if os.path.exists(e):
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


def edge(name, ref):
    us = sorted({int(m.group(1)) for p in glob.glob(f"{OUT}/analysis/{name}_u*_boxes.txt")
                 for m in [re.search(rf"{name}_u(\d+)_boxes", p)] if m}, reverse=True)
    return {"labels": [l for l, _ in EDGE] + ["출발 5 s 안에 세게 닿은 세션 (상대 차 · 상자)", "상자에 세게 닿은 뒤 5 s 안에 선 것 (혼자 · 상자)",
                                              "선 뒤 5 s 안에 다시 못 달린 것 (혼자 · 상자)", "선 뒤 5 s 안에 다시 못 달린 것 (상대 차 · 상자)"],
            "rows": [{"u": u, "run": edge_one(name, u), "ref": edge_one(ref, u) if ref else {}} for u in us]}


def figs():
    out = []
    for d in ("netaudit", "drift"):
        for p in glob.glob(f"{OUT}/{d}/*.png"):
            out.append({"path": f"{d}/{os.path.basename(p)}", "mtime": os.path.getmtime(p)})
    out.sort(key=lambda r: -r["mtime"])
    return out[:40]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
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
            if u.path.startswith("/api/") and u.path != "/api/figs" and not ok_run(name):
                return self._send(404, json.dumps({"error": "no such run"}))
            if u.path == "/api/progress":
                return self._json({"rows": memo(("p", name), 10, lambda: progress(name)), "status": memo(("s", name), 10, lambda: status(name))})
            if u.path == "/api/table":
                return self._json(memo(("t", name), 60, lambda: table(name)))
            if u.path == "/api/header":
                return self._json({"text": header(name)})
            if u.path == "/api/ckpts":
                cs = memo(("c", name), 30, lambda: ckpts(name))
                for c in cs:
                    c["summary"] = memo(("e", name, c["u"]), 60, lambda c=c: exam_summary(name, c["u"]))
                return self._json(cs)
            if u.path == "/api/compare":
                ref = q.get("ref", "")
                if not ok_run(ref):
                    return self._send(404, json.dumps({"error": "no such ref"}))
                return self._json(compare(name, int(q.get("u", "0")), ref))
            if u.path == "/api/edge":
                ref = q.get("ref", "")
                return self._json(memo(("edge", name, ref), 60, lambda: edge(name, ref if ok_run(ref) else "")))
            if u.path == "/api/figs":
                return self._json(memo("figs", 60, figs))
            if u.path == "/file":
                return self._file(q.get("path", ""))
        except Exception as exc:                       # one bad file must not take the page down
            return self._send(500, json.dumps({"error": str(exc)[:300]}))
        self._send(404, "{}")

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
