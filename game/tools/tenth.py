#!/usr/bin/env python3
"""TENTH 개발 루프 실행기 — 에이전트가 명령 한 줄로 쓰는 입구. 에디터 GUI 없음.

  python3 game/tools/tenth.py import                     # 자산 가져오기(.godot/ 캐시, 전역 클래스 목록)
  python3 game/tools/tenth.py test                       # GDScript 단위 시험(헤드리스) + 서버 파이썬 시험
  python3 game/tools/tenth.py shot --seconds 8 --out game/out/race.png [--camera top]
                                                         # 서버 + 앱(화면 밖) -> N초 뒤 PNG -> 둘 다 종료
  python3 game/tools/tenth.py bench --transports tcp,ws,udp,shm --seconds 15
                                                         # 합성 프레임(차 2대 × 1081점 × 40 Hz)으로 전송 비교
  python3 game/tools/tenth.py run                        # 사람용: 보통 창으로 서버 + 앱

화면을 가로채지 않는 방법(--offscreen, shot/bench):
  - game/.capture_host/ 에 프로젝트를 심볼릭 링크로 비추고 override.cfg 한 줄(display/window/size/no_focus=true)을 둔다.
    Godot X11은 no_focus 창을 override-redirect로 만들어 창 관리자가 손대지 않고 포커스도 가져가지 않는다.
    (원본 project.godot은 그대로라 사람이 여는 앱은 보통 창이다.)
  - 창은 8×8 px로 주 모니터 오른쪽 아래 구석에 생기고, 장면은 content_scale(viewport)로 캡처 해상도 그대로 렌더된다.
  - 창을 최소화하면 Godot은 그리기를 멈추므로(can_any_window_draw) 최소화는 쓰지 않는다.
띄운 프로세스는 PID로만 끝낸다(pkill 금지).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GAME = os.path.dirname(HERE)
REPO = os.path.dirname(GAME)
GODOT = os.path.join(HERE, "godot.sh")
PY = os.path.join(REPO, ".venv", "bin", "python")
HOST = os.path.join(GAME, ".capture_host")
OUT = os.path.join(GAME, "out")
LINKED = ("project.godot", "scenes", "scripts", "assets", "theme", "tests", "bench", ".godot")


def log(msg: str) -> None:
    print(f"[tenth] {msg}", flush=True)


# ------------------------------------------------------------------ 프로세스
def cpu_ticks(pid: int):
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        return int(parts[11]) + int(parts[12])           # utime + stime
    except OSError:
        return None


def rss_mb(pid: int):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None


class Proc:
    """setsid로 띄우고 PID로만 끝내는 자식 프로세스 + 로그 파일 + CPU 측정."""

    def __init__(self, name: str, argv, env=None, log_path=None):
        self.name = name
        self.log_path = log_path or os.path.join(OUT, f"{name}.log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.logf = open(self.log_path, "w")
        self.p = subprocess.Popen(argv, stdout=self.logf, stderr=subprocess.STDOUT, env=env,
                                  start_new_session=True, cwd=REPO)
        self.pid = self.p.pid
        self.t0 = time.time()
        self.tick0 = cpu_ticks(self.pid)
        self.peak_rss = 0.0

    def sample(self):
        r = rss_mb(self.pid)
        if r:
            self.peak_rss = max(self.peak_rss, r)
        c = self._cpu_now()
        if c is not None:
            self.last_cpu = c          # 프로세스가 끝난 뒤에도 마지막 값을 쓴다

    def mark(self):
        self.t0 = time.time()
        self.tick0 = cpu_ticks(self.pid)

    def cpu_pct(self):
        c = self._cpu_now()
        return c if c is not None else getattr(self, "last_cpu", None)

    def _cpu_now(self):
        t = cpu_ticks(self.pid)
        if t is None or self.tick0 is None:
            return None
        return 100.0 * (t - self.tick0) / os.sysconf("SC_CLK_TCK") / max(time.time() - self.t0, 1e-6)

    def text(self) -> str:
        if not self.logf.closed:
            self.logf.flush()
        try:
            return open(self.log_path, encoding="utf-8", errors="replace").read()
        except OSError:
            return ""

    def wait_line(self, pattern: str, timeout: float):
        t_end = time.time() + timeout
        while time.time() < t_end:
            m = re.search(pattern + r".*", self.text())
            if m:
                return m.group(0)
            if self.p.poll() is not None:
                return None
            self.sample()
            time.sleep(0.1)
        return None

    def stop(self, timeout: float = 6.0) -> int:
        if self.p.poll() is None:
            try:
                os.kill(self.pid, signal.SIGTERM)       # PID로만 (pkill 금지)
            except ProcessLookupError:
                pass
            try:
                self.p.wait(timeout)
            except subprocess.TimeoutExpired:
                os.kill(self.pid, signal.SIGKILL)
                self.p.wait(3)
        self.logf.close()
        return self.p.returncode


def free_gb() -> float:
    with open("/proc/meminfo") as f:
        mi = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return mi["MemAvailable"] / 1024 / 1024


def primary_corner():
    """주 모니터 오른쪽 아래 구석(8×8 창의 대부분이 화면 밖에 걸치도록)."""
    try:
        out = subprocess.run(["xrandr", "--current"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r" connected primary (\d+)x(\d+)\+(\d+)\+(\d+)", out)
        if m:
            w, h, x, y = map(int, m.groups())
            return x + w - 2, y + h - 2
    except Exception:
        pass
    return 0, 0


def ensure_host() -> str:
    """화면 밖 캡처용 프로젝트 거울(.capture_host): 원본을 링크하고 override.cfg만 다르다."""
    os.makedirs(HOST, exist_ok=True)
    for n in LINKED:
        src = os.path.join(GAME, n)
        dst = os.path.join(HOST, n)
        if os.path.exists(src) and not os.path.islink(dst):
            if os.path.exists(dst):
                shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
            os.symlink(os.path.relpath(src, HOST), dst)
    with open(os.path.join(HOST, "override.cfg"), "w") as f:
        f.write("; tools/tenth.py가 만든다. 에이전트 캡처 전용: 포커스를 받지 않는 override-redirect 창\n"
                "[display]\nwindow/size/no_focus=true\nwindow/size/window_width_override=8\n"
                "window/size/window_height_override=8\n")
    return HOST


def godot_env():
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":0")
    env.pop("WAYLAND_DISPLAY", None)
    return env


def cmd_import(_a=None) -> int:
    t0 = time.time()
    r = subprocess.run([GODOT, "--headless", "--import", "--path", GAME], capture_output=True, text=True, timeout=600)
    errs = [l for l in (r.stdout + r.stderr).splitlines() if "ERROR" in l or "SCRIPT ERROR" in l]
    log(f"import 끝 ({time.time() - t0:.1f} s, 종료 코드 {r.returncode}, 오류 {len(errs)}줄)")
    for l in errs[:20]:
        print("   ", l)
    return 0 if r.returncode == 0 and not errs else 1


# ------------------------------------------------------------------ test
def cmd_test(a) -> int:
    if not os.path.isdir(os.path.join(GAME, ".godot")):
        cmd_import()
    t0 = time.time()
    r = subprocess.run([GODOT, "--headless", "--path", GAME, "-s", "res://tests/run_tests.gd"],
                       capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    for line in out.splitlines():
        if line.startswith(("TEST", "  ", "SCRIPT ERROR", "ERROR", "Parse Error")) or "실패" in line:
            print(line)
    log(f"GDScript 시험: 종료 코드 {r.returncode} ({time.time() - t0:.1f} s)")
    rc = r.returncode
    if not a.godot_only:
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CUDA_VISIBLE_DEVICES="")
        r2 = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                             os.path.join(REPO, "f1sim", "f1sim", "server", "tests")], env=env, cwd=REPO)
        log(f"서버 파이썬 시험: 종료 코드 {r2.returncode}")
        rc = rc or r2.returncode
    return rc


# ------------------------------------------------------------------ shot
def start_server(a, transport="tcp", port=47800, bench=False, extra=()):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    if bench:
        argv = [PY, "-m", "f1sim.server.bench", "--transport", transport, "--port", str(port),
                "--cars", str(a.cars), "--duration", str(a.seconds + 30)]
    else:
        argv = [PY, "-m", "f1sim.server", "--transport", transport, "--port", str(port), "--track", a.track]
        if a.compile:
            argv.append("--compile")
    argv += list(extra)
    return Proc(f"server_{transport}" if bench else "server", argv, env=env)


def start_app(args_after, scene=None, name="godot", offscreen=True, uncapped=False):
    argv = [GODOT, "--path", ensure_host() if offscreen else GAME]
    if uncapped:
        argv += ["--disable-vsync"]
        args_after = list(args_after) + ["--max-fps=0"]
    if offscreen:
        x, y = primary_corner()
        argv += ["--position", f"{x},{y}", "--resolution", "8x8"]
    if scene:
        argv += ["--scene", scene]
    argv += ["--"] + (["--offscreen"] if offscreen else []) + list(args_after)
    return Proc(name, argv, env=godot_env())


def cmd_shot(a) -> int:
    log(f"가용 메모리 {free_gb():.1f} GB")
    if not os.path.isdir(os.path.join(GAME, ".godot")):
        cmd_import()
    srv = start_server(a, extra=["--autostart"] if a.autostart else [])
    ready = srv.wait_line("TENTH_SERVER_READY", 240)
    if not ready:
        log("서버가 준비되지 않았습니다:\n" + srv.text()[-3000:])
        srv.stop()
        return 1
    log(ready)
    out = os.path.abspath(a.out)
    metrics = os.path.splitext(out)[0] + "_metrics.json"
    app = start_app([f"--server=127.0.0.1:47800", "--transport=tcp", "--autostart", f"--camera={a.camera}",
                     f"--capture={out}", f"--capture-after={a.seconds}", f"--metrics={metrics}",
                     f"--quit-after={a.seconds + 0.5}"] + ([f"--inject={a.inject}"] if a.inject else []),
                    uncapped=a.uncapped)
    srv.mark(); app.mark()
    t_end = time.time() + a.seconds + 90
    while app.p.poll() is None and time.time() < t_end:
        srv.sample(); app.sample()
        time.sleep(0.25)
    srv_cpu, app_cpu = srv.cpu_pct(), app.cpu_pct()
    rc_app = app.stop()
    srv.stop()
    cap = re.search(r"TENTH_CAPTURE .*", app.text())
    stats = re.findall(r"TENTH_STATS (.*)", srv.text())
    res = {"capture": cap.group(0) if cap else None, "app_exit": rc_app,
           "server_cpu_pct": round(srv_cpu, 1) if srv_cpu is not None else None,
           "app_cpu_pct": round(app_cpu, 1) if app_cpu is not None else None,
           "server_peak_rss_mb": round(srv.peak_rss), "app_peak_rss_mb": round(app.peak_rss),
           "server_last_stats": json.loads(stats[-1]) if stats else None}
    try:
        res["app_metrics"] = json.load(open(metrics))
    except Exception:
        pass
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if not cap:
        log("캡처가 없습니다. 앱 로그 끝:\n" + app.text()[-3000:])
        return 1
    return 0


# ------------------------------------------------------------------ bench
def cmd_bench(a) -> int:
    log(f"가용 메모리 {free_gb():.1f} GB")
    if not os.path.isdir(os.path.join(GAME, ".godot")):
        cmd_import()
    rows = []
    for i, tr in enumerate(a.transports.split(",")):
        port = 47810 + i
        srv = start_server(a, transport=tr, port=port, bench=True)
        if not srv.wait_line("TENTH_SERVER_READY", 30):
            log(f"{tr}: 서버 준비 실패\n{srv.text()[-2000:]}")
            srv.stop()
            continue
        metrics = os.path.join(OUT, f"bench_{tr}.json")
        app = start_app([f"--server=127.0.0.1:{port}", f"--transport={tr}", f"--metrics={metrics}",
                         f"--quit-after={a.seconds}"], scene="res://bench/transport_bench.tscn", name=f"godot_{tr}")
        time.sleep(2.0)            # 연결·첫 프레임 뒤부터 CPU를 잰다
        srv.mark(); app.mark()
        t_end = time.time() + a.seconds + 60
        while app.p.poll() is None and time.time() < t_end:
            srv.sample(); app.sample()
            time.sleep(0.25)
        srv_cpu, app_cpu = srv.cpu_pct(), app.cpu_pct()
        app.stop(); srv.stop()
        try:
            m = json.load(open(metrics))
        except Exception:
            log(f"{tr}: 측정 파일 없음. 앱 로그:\n{app.text()[-2000:]}")
            continue
        b = re.search(r"TENTH_BENCH (.*)", srv.text())
        sb = json.loads(b.group(1)) if b else {}
        row = {"transport": tr, "server_cpu_pct": round(srv_cpu or 0, 1), "app_cpu_pct": round(app_cpu or 0, 1),
               "frame_bytes": sb.get("frame_bytes"), "encode_ms": sb.get("encode_ms_med"),
               "server_rss_mb": round(srv.peak_rss), "app_rss_mb": round(app.peak_rss)}
        for k in ("state_frames", "fps_median", "frame_ms_p95", "latency_ms_p50", "latency_ms_p95", "rtt_ms_p50",
                  "rtt_ms_p95", "decode_us_p50", "apply_ms_p50", "seq_gaps", "torn"):
            v = m.get(k)
            row[k] = round(v, 3) if isinstance(v, float) else v
        rows.append(row)
        log(json.dumps(row, ensure_ascii=False))
    with open(os.path.join(OUT, "bench_summary.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return 0 if rows else 1


def cmd_probe(a) -> int:
    """전송만의 지연: 합성 서버 + 파이썬 수신기(화면 주기 없음)."""
    rows = []
    for i, tr in enumerate(a.transports.split(",")):
        port = 47830 + i
        srv = start_server(a, transport=tr, port=port, bench=True)
        if not srv.wait_line("TENTH_SERVER_READY", 30):
            srv.stop(); continue
        cl = Proc(f"probe_{tr}", [PY, "-m", "f1sim.server.probe", "--transport", tr, "--port", str(port),
                                   "--seconds", str(a.seconds)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        srv.mark(); cl.mark()
        line = cl.wait_line("TENTH_PROBE", a.seconds + 30)
        srv_cpu, cl_cpu = srv.cpu_pct(), cl.cpu_pct()
        cl.stop(); srv.stop()
        if line:
            row = json.loads(line.split(" ", 1)[1])
            row.update(server_cpu_pct=round(srv_cpu or 0, 1), client_cpu_pct=round(cl_cpu or 0, 1))
            rows.append(row)
            log(json.dumps(row, ensure_ascii=False))
    with open(os.path.join(OUT, "probe_summary.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return 0 if rows else 1


def cmd_run(a) -> int:
    srv = start_server(a)
    if not srv.wait_line("TENTH_SERVER_READY", 240):
        log(srv.text()[-3000:]); srv.stop(); return 1
    app = start_app(["--server=127.0.0.1:47800", "--transport=tcp"], offscreen=False)
    log(f"서버 PID {srv.pid}, 앱 PID {app.pid}. 앱 창을 닫으면 서버도 끝납니다.")
    try:
        app.p.wait()
    except KeyboardInterrupt:
        app.stop()
    srv.stop()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("import")
    t = sub.add_parser("test")
    t.add_argument("--godot-only", action="store_true")
    for name in ("shot", "bench", "run", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--seconds", type=float, default=8.0)
        p.add_argument("--track", default="real/korea26")
        p.add_argument("--cars", type=int, default=2)
        p.add_argument("--compile", action="store_true", help="서버 물리 루프를 CPU inductor로 컴파일(실시간 확보)")
        if name == "shot":
            p.add_argument("--out", default=os.path.join(OUT, "race.png"))
            p.add_argument("--camera", default="chase", choices=("chase", "top"))
            p.add_argument("--autostart", action="store_true", default=True)
            p.add_argument("--inject", default="", help="예: race_toggle@4,race_toggle@5 (동작@초)")
            p.add_argument("--uncapped", action="store_true", help="fps 제한·수직 동기 끔(렌더 여유 측정)")
        if name in ("bench", "probe"):
            p.add_argument("--transports", default="tcp,ws,udp,shm")
    a = ap.parse_args(argv)
    return {"import": cmd_import, "test": cmd_test, "shot": cmd_shot, "bench": cmd_bench, "run": cmd_run, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
