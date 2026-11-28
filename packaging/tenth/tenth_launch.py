#!/usr/bin/python3
"""TENTH 실행기(설치본). 아이콘이 이것을 부른다.

  tenth-launch                 시뮬 서버를 띄우고, 준비되면 앱 창을 띄운다. 앱을 닫으면 서버도 끝낸다(PID로).
  tenth-launch check           서버만 띄워 준비되는지 보고 끝낸다(문제 찾기용).
  tenth-launch flow --out DIR  메인 -> 레이스 준비 -> 주행 -> 결과를 자동으로 밟고 화면마다 PNG를 남긴다.
                               창은 포커스를 받지 않는 8×8 창(화면을 가로채지 않음). 결과 요약: DIR/flow_summary.json
  공통: --no-dialog            오류를 창 대신 표준 오류로만(ssh에서)

설정: ~/.config/tenth/config.yaml 한 곳(파이썬 venv, GPU, 자료·체크포인트·결과 위치, 메모리 상한).
로그: 설정의 log_dir(기본 <설치 폴더>/logs): launcher.log, server-<시각>.log, app-<시각>.log
중복 실행은 <설치 폴더>/run/launcher.lock 으로 막는다. 프로세스는 PID로만 끝낸다(pkill 쓰지 않음).
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

APP_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG = os.path.expanduser(os.environ.get("TENTH_CONFIG") or "~/.config/tenth/config.yaml")   # TENTH_CONFIG: 시험용 다른 설정
RUN_DIR = os.path.join(APP_DIR, "run")
DEFAULTS = {
    "python": "",                 # 서버를 돌릴 파이썬(torch가 있는 venv)
    "device": "cuda",             # cuda | cpu
    "gpu": "0",                   # 서버의 CUDA_VISIBLE_DEVICES
    "app_gpu_index": "",          # 앱(Godot) --gpu-index. 비우면 Godot이 고름
    "library": os.path.join(APP_DIR, "library"),
    "data": os.path.join(APP_DIR, "data"),
    "external": "",               # F1SIM_EXTERNAL(팀 지도 폴더). 비우면 설정 안 함
    "server_home": "",            # 서버의 HOME. 비우면 보통 HOME
    "port": 47800,
    "server_memory_max": "",      # 예: 3G. 비우면 상한 없음
    "app_memory_max": "",
    "min_free_gb": 0,
    "ready_timeout_s": 300,
    "log_dir": os.path.join(APP_DIR, "logs"),
    "keep_logs": 20,
}
NO_DIALOG = False
_children: list = []


# ------------------------------------------------------------------ 설정·로그
def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if os.path.isfile(CONFIG):
        text = open(CONFIG, encoding="utf-8").read()
        try:
            import yaml
            doc = yaml.safe_load(text) or {}
        except ImportError:                       # 평평한 key: value 만 쓰므로 간단히 읽는다
            doc = {}
            for line in text.splitlines():
                line = line.split(" #", 1)[0].strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                doc[k.strip()] = v.strip().strip('"').strip("'")
        cfg.update({k: v for k, v in doc.items() if v is not None})
    for k in ("python", "library", "data", "external", "server_home", "log_dir"):
        if cfg.get(k):
            cfg[k] = os.path.expanduser(str(cfg[k]))
    return cfg


class Log:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
        self.f.write(line + "\n")
        self.f.flush()
        print(f"[tenth] {msg}", file=sys.stderr, flush=True)


def rotate_logs(log_dir: str, keep: int) -> None:
    for pat in ("server-*.log", "app-*.log"):
        files = sorted(glob.glob(os.path.join(log_dir, pat)))
        for f in files[:-keep] if keep > 0 else []:
            try:
                os.remove(f)
            except OSError:
                pass


# ------------------------------------------------------------------ 사용자에게 보이는 알림
def _gui_env() -> dict:
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":0")
    return env


def error_box(title: str, text: str) -> None:
    print(f"[tenth] 오류: {title}\n{text}", file=sys.stderr, flush=True)
    if NO_DIALOG:
        return
    for argv in (["zenity", "--error", "--no-wrap", "--title=TENTH", f"--text=<b>{_esc(title)}</b>\n\n{_esc(text)}"],
                 ["xmessage", "-center", f"TENTH: {title}\n\n{text}"],
                 ["notify-send", "-u", "critical", f"TENTH: {title}", text]):
        if shutil.which(argv[0]):
            subprocess.run(argv, env=_gui_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return


def info_box(text: str) -> None:
    print(f"[tenth] {text}", file=sys.stderr, flush=True)
    if NO_DIALOG:
        return
    if shutil.which("zenity"):
        subprocess.run(["zenity", "--info", "--no-wrap", "--title=TENTH", f"--text={_esc(text)}"], env=_gui_env(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif shutil.which("notify-send"):
        subprocess.run(["notify-send", "TENTH", text], env=_gui_env())


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Progress:
    """서버가 뜨는 동안의 작은 진행 창(zenity). 취소를 누르면 cancelled()가 참."""

    def __init__(self, text: str):
        self.p = None
        self.t0 = time.time()
        if NO_DIALOG or not shutil.which("zenity"):
            return
        self.p = subprocess.Popen(["zenity", "--progress", "--pulsate", "--auto-close", "--title=TENTH", "--width=360",
                                   f"--text={text}"], stdin=subprocess.PIPE, env=_gui_env(),
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)

    def cancelled(self) -> bool:
        """사용자가 취소(또는 창 닫기)했으면 참. zenity가 뜨자마자 오류로 끝난 경우(1 s 안)는 취소로 보지 않는다."""
        if self.p is None or self.p.poll() is None:
            return False
        if time.time() - self.t0 < 1.0 or self.p.returncode not in (1, 252):
            self.p = None
            return False
        return True

    def close(self) -> None:
        if self.p is None or self.p.poll() is not None:
            return
        try:
            self.p.stdin.write("100\n")
            self.p.stdin.flush()
            self.p.wait(2)
        except Exception:
            try:
                os.kill(self.p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


# ------------------------------------------------------------------ 프로세스
def _pdeathsig() -> None:
    """실행기가 어떻게 죽든(SIGKILL 포함) 자식이 SIGTERM을 받게 한다."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
    except Exception:
        pass


_scope_ok = None


def scope_ok() -> bool:
    global _scope_ok
    if _scope_ok is None:
        _scope_ok = bool(shutil.which("systemd-run")) and subprocess.run(
            ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=64M", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    return _scope_ok


class Proc:
    def __init__(self, name: str, argv: list, env: dict, log_path: str, memory_max: str = "", cwd: str = APP_DIR):
        self.name = name
        self.memory_max = memory_max
        if memory_max and scope_ok():
            argv = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={memory_max}", "-p", "MemorySwapMax=0"] + list(argv)
        self.log_path = log_path
        self.logf = open(log_path, "w", encoding="utf-8")
        self.p = subprocess.Popen(argv, stdout=self.logf, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                                  start_new_session=True, preexec_fn=_pdeathsig)
        self.pid = self.p.pid
        self.peak_rss = 0.0
        _children.append(self)

    def alive(self) -> bool:
        return self.p.poll() is None

    def text(self) -> str:
        if not self.logf.closed:
            self.logf.flush()
        try:
            return open(self.log_path, encoding="utf-8", errors="replace").read()
        except OSError:
            return ""

    def sample(self) -> None:
        try:
            for line in open(f"/proc/{self.pid}/status"):
                if line.startswith("VmRSS:"):
                    self.peak_rss = max(self.peak_rss, int(line.split()[1]) / 1024.0)
        except OSError:
            pass

    def stop(self, timeout: float = 6.0) -> None:
        if self.p.poll() is None:
            try:
                os.kill(self.pid, signal.SIGTERM)          # PID로만
            except ProcessLookupError:
                pass
            try:
                self.p.wait(timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.kill(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.p.wait(3)
        if not self.logf.closed:
            self.logf.close()


def stop_all() -> None:
    for c in reversed(_children):
        c.stop()


def free_gb() -> float:
    with open("/proc/meminfo") as f:
        mi = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return mi["MemAvailable"] / 1024 / 1024


def free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"{start}~{start + 49} 포트가 모두 쓰이고 있습니다")


def reason_of(text: str, rc, memory_max: str) -> str:
    """서버 로그에서 사용자에게 보일 한 줄 원인."""
    if rc in (-9, 137) and memory_max:
        return f"서버가 메모리 상한 {memory_max}를 넘어 강제 종료됐습니다(설정 server_memory_max를 올리거나 다른 작업을 줄이십시오)."
    m = re.findall(r"TENTH_ERROR (.*)", text)
    if m:
        return m[-1][:300]
    if "CUDA" in text and re.search(r"no CUDA|CUDA unavailable|cuda.*not available|No CUDA GPUs", text, re.I):
        return "CUDA GPU를 쓸 수 없습니다(설정 gpu 번호 또는 드라이버 확인. device: cpu로도 돌 수 있음)."
    m = re.findall(r"^\s*(\w+(?:Error|Exception)\b.*)$", text, re.M)
    if m:
        return m[-1][:300]
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if rc is not None:
        return (lines[-1][:300] if lines else "") + f" (종료 코드 {rc})"
    return lines[-1][:300] if lines else "로그가 비어 있습니다"


def acquire_lock():
    os.makedirs(RUN_DIR, exist_ok=True)
    fd = os.open(os.path.join(RUN_DIR, "launcher.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    return fd


def kill_stale_server(log: Log) -> None:
    """지난 실행기가 비정상으로 끝나 남은 서버(pid 파일)를 PID로 끝낸다."""
    pf = os.path.join(RUN_DIR, "server.pid")
    try:
        pid = int(open(pf).read().split()[0])
        cmd = open(f"/proc/{pid}/cmdline").read()
    except (OSError, ValueError, IndexError):
        return
    if "f1sim.server" in cmd and APP_DIR in open(f"/proc/{pid}/environ", errors="replace").read():
        log(f"지난 실행에서 남은 서버 PID {pid}를 끝냅니다")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


# ------------------------------------------------------------------ 서버·앱 시작
def server_env(cfg: dict) -> dict:
    env = dict(os.environ)
    real_home = os.path.expanduser("~")
    env.update(OMP_NUM_THREADS="1", TORCHINDUCTOR_COMPILE_THREADS="1", F1SIM_RACELINE_CACHE_ONLY="1",
               PYTHONUNBUFFERED="1", PYTHONPATH=os.path.join(APP_DIR, "server"),
               TENTH_LIBRARY=cfg["library"], TENTH_DATA=cfg["data"], TENTH_APP_DIR=APP_DIR)
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"]) if str(cfg["device"]).startswith("cuda") else ""
    if cfg.get("external"):
        env["F1SIM_EXTERNAL"] = cfg["external"]
    if cfg.get("server_home"):
        # 옛 코드의 ~/.cache/f1sim, ~/f1sim_runs, ~/.f1sim/maps 를 이 폴더 안에서 찾게 한다(기존 폴더를 건드리지 않음).
        # 파이썬 사용자 패키지(~/.local, torch가 여기 있을 수 있음)는 진짜 홈의 것을 그대로 쓴다.
        env["HOME"] = cfg["server_home"]
        env.setdefault("PYTHONUSERBASE", os.path.join(real_home, ".local"))
    return env


def check_setup(cfg: dict) -> str:
    """시작 전에 알 수 있는 설정 문제(한 줄). 없으면 빈 문자열."""
    py = cfg.get("python") or ""
    if not py or not os.access(py, os.X_OK):
        return f"설정한 파이썬이 없습니다: {py or '(비어 있음)'} — {CONFIG}의 python을 고치십시오."
    if not os.path.isdir(cfg["library"]):
        return f"자료 폴더가 없습니다: {cfg['library']} — {CONFIG}의 library를 고치십시오."
    if not os.path.isdir(os.path.join(APP_DIR, "server", "f1sim")):
        return f"서버 코드가 없습니다: {APP_DIR}/server/f1sim — 다시 설치하십시오(packaging/tenth/update.sh)."
    if not os.access(os.path.join(APP_DIR, "tenth.x86_64"), os.X_OK):
        return f"앱 실행 파일이 없습니다: {APP_DIR}/tenth.x86_64 — 다시 설치하십시오."
    if cfg.get("server_home") and not os.path.isdir(cfg["server_home"]):
        return f"server_home 폴더가 없습니다: {cfg['server_home']}"
    need = float(cfg.get("min_free_gb") or 0)
    if need > 0 and free_gb() < need:
        return f"가용 메모리 {free_gb():.1f} GB가 필요한 {need:g} GB보다 적습니다. 다른 작업을 줄이고 다시 여십시오."
    return ""


def start_server(cfg: dict, port: int, stamp: str, log: Log) -> Proc:
    os.makedirs(cfg["data"], exist_ok=True)
    argv = [cfg["python"], "-m", "tenth_boot", "f1sim.server", "--device", str(cfg["device"]), "--port", str(port)]
    srv = Proc("server", argv, server_env(cfg), os.path.join(cfg["log_dir"], f"server-{stamp}.log"),
               memory_max=str(cfg.get("server_memory_max") or ""))
    with open(os.path.join(RUN_DIR, "server.pid"), "w") as f:
        f.write(f"{srv.pid}\n")
    log(f"서버 PID {srv.pid} 포트 {port} 장치 {cfg['device']} gpu={cfg['gpu']} 로그 {srv.log_path}")
    return srv


def wait_ready(srv: Proc, timeout: float, progress: Progress = None):
    t_end = time.time() + timeout
    while time.time() < t_end:
        m = re.search(r"TENTH_SERVER_READY (.*)", srv.text())
        if m:
            return json.loads(m.group(1)), ""
        if not srv.alive():
            return None, reason_of(srv.text(), srv.p.returncode, srv.memory_max)
        if progress is not None and progress.cancelled():
            return None, "취소"
        srv.sample()
        time.sleep(0.2)
    return None, f"서버가 {timeout:.0f} s 안에 준비되지 않았습니다. " + reason_of(srv.text(), None, srv.memory_max)


def cpu_fallback(cfg: dict, ready: dict, log: Log) -> None:
    """설정은 cuda인데 서버가 CPU로 떴으면(GPU 번호가 틀렸거나 CUDA를 못 씀) 알린다. 서버는 그대로 돈다(약 0.4배 느림)."""
    if str(cfg["device"]).startswith("cuda") and ready.get("device") != "cuda":
        msg = f"GPU(gpu: {cfg['gpu']})를 쓸 수 없어 시뮬이 CPU로 돕니다(실시간의 약 0.4배). {CONFIG}의 gpu를 확인하십시오."
        log("주의: " + msg)
        if not NO_DIALOG and shutil.which("notify-send"):
            subprocess.Popen(["notify-send", "-u", "critical", "TENTH", msg], env=_gui_env(),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def app_env() -> dict:
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":0")
    if "XAUTHORITY" not in env:
        xa = f"/run/user/{os.getuid()}/gdm/Xauthority"
        if os.path.isfile(xa) and not os.path.isfile(os.path.expanduser("~/.Xauthority")):
            env["XAUTHORITY"] = xa
    env.pop("WAYLAND_DISPLAY", None)              # M1은 X11에서 확인함
    return env


def app_argv(cfg: dict, pre: list, user: list, exe: str = "") -> list:
    argv = [exe or os.path.join(APP_DIR, "tenth.x86_64")] + list(pre)
    if str(cfg.get("app_gpu_index", "")).strip() != "":
        argv += ["--gpu-index", str(cfg["app_gpu_index"])]
    return argv + ["--"] + list(user)


# ------------------------------------------------------------------ 명령
def cmd_run(cfg: dict, log: Log) -> int:
    err = check_setup(cfg)
    if err:
        log("설정 문제: " + err)
        error_box("시작할 수 없습니다", err + f"\n\n로그: {log.path}")
        return 1
    stamp = time.strftime("%Y%m%d-%H%M%S")
    port = free_port(int(cfg["port"]))
    srv = start_server(cfg, port, stamp, log)
    prog = Progress("시뮬 서버를 띄우는 중입니다…")
    ready, why = wait_ready(srv, float(cfg["ready_timeout_s"]), prog)
    prog.close()
    if not ready:
        srv.stop()
        if why == "취소":
            log("사용자가 취소함")
            return 0
        log("서버 실패: " + why)
        error_box("시뮬 서버가 뜨지 않았습니다", f"{why}\n\n서버 로그: {srv.log_path}")
        return 1
    log("서버 준비: " + json.dumps(ready, ensure_ascii=False))
    cpu_fallback(cfg, ready, log)
    app = Proc("app", app_argv(cfg, [], [f"--server=127.0.0.1:{port}", "--transport=tcp"]), app_env(),
               os.path.join(cfg["log_dir"], f"app-{stamp}.log"), memory_max=str(cfg.get("app_memory_max") or ""))
    log(f"앱 PID {app.pid} 로그 {app.log_path}")
    t0 = time.time()
    warned = False
    while app.alive():
        if not srv.alive() and not warned:
            warned = True
            why = reason_of(srv.text(), srv.p.returncode, srv.memory_max)
            log("주행 중 서버가 끝남: " + why)
            if not NO_DIALOG and shutil.which("zenity"):
                subprocess.Popen(["zenity", "--error", "--no-wrap", "--title=TENTH",
                                  f"--text=<b>시뮬 서버가 끝났습니다</b>\n\n{_esc(why)}\n\n서버 로그: {_esc(srv.log_path)}"],
                                 env=_gui_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(0.5)
    rc = app.p.returncode
    log(f"앱 종료(코드 {rc}, {time.time() - t0:.0f} s) -> 서버 정리")
    srv.stop()
    if rc not in (0, None) and time.time() - t0 < 10:
        error_box("앱이 바로 끝났습니다", reason_of(app.text(), rc, app.memory_max) + f"\n\n앱 로그: {app.log_path}")
        return 1
    return 0


def cmd_check(cfg: dict, log: Log) -> int:
    err = check_setup(cfg)
    if err:
        log("설정 문제: " + err)
        return 1
    stamp = time.strftime("%Y%m%d-%H%M%S")
    srv = start_server(cfg, free_port(int(cfg["port"])), stamp, log)
    ready, why = wait_ready(srv, float(cfg["ready_timeout_s"]))
    srv.stop()
    if not ready:
        log("서버 실패: " + why)
        return 1
    cpu_fallback(cfg, ready, log)
    env = server_env(cfg)
    mod = subprocess.run([cfg["python"], "-m", "tenth_boot", "--where"],
                         env=env, capture_output=True, text=True).stdout.strip()
    print(json.dumps({"ready": ready, "f1sim_torch": mod}, ensure_ascii=False))
    return 0


def primary_corner():
    try:
        out = subprocess.run(["xrandr", "--current"], capture_output=True, text=True, timeout=5, env=app_env()).stdout
        m = re.search(r" connected primary (\d+)x(\d+)\+(\d+)\+(\d+)", out) or re.search(r" connected (\d+)x(\d+)\+(\d+)\+(\d+)", out)
        if m:
            w, h, x, y = map(int, m.groups())
            return x + w - 2, y + h - 2
    except Exception:
        pass
    return 0, 0


def capture_host() -> str:
    """화면 밖 캡처용 실행 파일. 내보낸 앱은 실행 파일 폴더의 override.cfg를 읽는다(릴리스 템플릿은 --main-pack/--path를
    막아 둠). 그래서 run/capture_host/ 에 실행 파일·pck를 하드 링크로 두고(심볼릭 링크는 /proc/self/exe가 원래 폴더로 풀려
    안 됨) 그 옆에만 no_focus(override-redirect, 포커스 없음) 8×8 창 설정을 둔다. 사람이 여는 앱은 보통 창 그대로다."""
    host = os.path.join(RUN_DIR, "capture_host")
    os.makedirs(host, exist_ok=True)
    for n in ("tenth.x86_64", "tenth.pck"):              # 갱신 뒤 새 파일을 가리키도록 매번 다시 건다
        dst = os.path.join(host, n)
        if os.path.lexists(dst):
            os.remove(dst)
        try:
            os.link(os.path.join(APP_DIR, n), dst)
        except OSError:
            shutil.copy2(os.path.join(APP_DIR, n), dst)
    with open(os.path.join(host, "override.cfg"), "w") as f:
        f.write("; tenth-launch flow 전용: 포커스를 받지 않는 override-redirect 8x8 창\n"
                "[display]\nwindow/size/no_focus=true\nwindow/size/window_width_override=8\n"
                "window/size/window_height_override=8\n")
    return os.path.join(host, "tenth.x86_64")


def cmd_flow(cfg: dict, log: Log, a) -> int:
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(out, exist_ok=True)
    err = check_setup(cfg)
    if err:
        log("설정 문제: " + err)
        return 1
    stamp = time.strftime("%Y%m%d-%H%M%S")
    port = free_port(int(cfg["port"]) + 1)
    srv = start_server(cfg, port, stamp, log)
    ready, why = wait_ready(srv, float(cfg["ready_timeout_s"]))
    if not ready:
        srv.stop()
        log("서버 실패: " + why)
        return 1
    log("서버 준비: " + json.dumps(ready, ensure_ascii=False))
    cpu_fallback(cfg, ready, log)
    src, _, idx = a.layout.partition(":")
    boxes = {"source": src, "count": a.boxes}
    if src == "saved":
        boxes["index"] = int(idx or 0)
    elif src == "random":
        boxes["seed"] = int(idx or 1)
    spec = {"course": a.course, "driver": a.driver, "exec": a.exec_mode, "physics": a.physics, "rules": a.rules,
            "duration_s": a.seconds, "remove_at_s": a.remove if a.remove > 0 else None, "boxes": boxes}
    # 메인 화면은 출발 신호(go_after)의 절반까지, 준비 화면은 그다음 절반 동안 보인다 -> go_after 10 s면 둘 다 3 s에 찍힌다
    plan = a.plan or f"main:3,setup:3,live:{a.live1:g},live:{a.live2:g},result:4"
    x, y = primary_corner()
    metrics = os.path.join(out, "flow_metrics.json")
    user = ["--offscreen", f"--server=127.0.0.1:{port}", "--transport=tcp", f"--spec={json.dumps(spec, ensure_ascii=False)}",
            f"--go-after={a.go_after}", f"--capture-plan={plan}", f"--capture-dir={out}", f"--metrics={metrics}"]
    if a.camera:
        user.append("--camera=" + a.camera)
    pre = ["--position", f"{x},{y}", "--resolution", "8x8"]
    app = Proc("app", app_argv(cfg, pre, user, exe=capture_host()), app_env(), os.path.join(cfg["log_dir"], f"app-{stamp}.log"),
               memory_max=str(cfg.get("app_memory_max") or ""))
    log(f"앱(화면 밖) PID {app.pid} -> {out}")
    t_end = time.time() + a.seconds / 0.4 + 240
    while app.alive() and time.time() < t_end:
        srv.sample(); app.sample()
        if not srv.alive() or "TENTH_ERROR" in srv.text():
            log("서버가 오류를 알렸거나 끝났습니다: " + reason_of(srv.text(), srv.p.returncode, srv.memory_max))
            time.sleep(3)
            break
        time.sleep(0.5)
    rc_app = app.p.poll()
    app.stop()
    stext = srv.text()
    srv.stop()
    atext = app.text()
    caps = re.findall(r"TENTH_CAPTURE (\S+)", atext)
    res_line = re.search(r"TENTH_RESULT (.*)", stext)
    stats = [json.loads(s) for s in re.findall(r"TENTH_STATS (.*)", stext)]
    sess = re.search(r"TENTH_SESSION (.*)", stext)
    rts = [s["rt"] for s in stats if s.get("status") == "running" and s.get("rt", 0) > 0]
    summary = {"installed": APP_DIR, "version": open(os.path.join(APP_DIR, "VERSION")).read().split("\n")[0]
               if os.path.isfile(os.path.join(APP_DIR, "VERSION")) else "", "server": ready,
               "captures": caps, "app_exit": rc_app,
               "server_peak_rss_mb": round(srv.peak_rss), "app_peak_rss_mb": round(app.peak_rss),
               "rt_running": rts, "rt_median": sorted(rts)[len(rts) // 2] if rts else None,
               "session": json.loads(sess.group(1)) if sess else None,
               "result": json.loads(res_line.group(1)) if res_line else None,
               "server_log": srv.log_path, "app_log": app.log_path}
    try:
        summary["app_metrics"] = json.load(open(metrics))
    except Exception:
        pass
    if summary["result"] and summary["result"].get("files", {}).get("json"):
        v = subprocess.run([cfg["python"], "-m", "tenth_boot", "f1sim.server.verify", summary["result"]["files"]["json"]],
                           capture_output=True, text=True, env=server_env(cfg), cwd=APP_DIR)
        try:
            summary["verify"] = json.loads(v.stdout)
            log(f"독립 검산: 일치 {summary['verify'].get('agree')}")
        except Exception:
            summary["verify"] = {"error": (v.stdout + v.stderr)[-2000:]}
            log("독립 검산 못 함: " + reason_of(v.stdout + v.stderr, None, ""))
    with open(os.path.join(out, "flow_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: summary.get(k) for k in ("captures", "app_exit", "server_peak_rss_mb", "app_peak_rss_mb",
                                                  "rt_median", "result")}, ensure_ascii=False, indent=1))
    if len(caps) < 3:
        log("캡처가 모자랍니다. 앱 로그 끝:\n" + atext[-2000:] + "\n서버 로그 끝:\n" + stext[-2000:])
        return 1
    return 0


def main(argv=None) -> int:
    global NO_DIALOG
    ap = argparse.ArgumentParser(prog="tenth-launch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-dialog", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("run")
    sub.add_parser("check")
    f = sub.add_parser("flow")
    f.add_argument("--out", required=True)
    f.add_argument("--course", default="kr26-hall")
    f.add_argument("--driver", default="scan31b-u475")
    f.add_argument("--exec", dest="exec_mode", default="real", choices=("real", "ideal"))
    f.add_argument("--physics", default="default-car-pre-place")
    f.add_argument("--rules", default="kr-event-quali")
    f.add_argument("--seconds", type=float, default=60.0)
    f.add_argument("--remove", type=float, default=30.0)
    f.add_argument("--boxes", type=int, default=3)
    f.add_argument("--layout", default="saved:0", help="saved:<번호> | random:<시드>")
    f.add_argument("--go-after", type=float, default=10.0, help="메인 -> 준비(절반) -> 출발까지 초")
    f.add_argument("--live1", type=float, default=16.0)
    f.add_argument("--live2", type=float, default=45.0)
    f.add_argument("--plan", default="")
    f.add_argument("--camera", default="")
    a = ap.parse_args(argv)
    NO_DIALOG = a.no_dialog or a.cmd in ("check", "flow")
    cfg = load_config()
    os.makedirs(cfg["log_dir"], exist_ok=True)
    log = Log(os.path.join(cfg["log_dir"], "launcher.log"))
    fd = acquire_lock()
    if fd is None:
        info_box("TENTH가 이미 실행 중입니다. 열린 창을 쓰거나, 그 창을 닫은 뒤 다시 여십시오.")
        return 0
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, lambda *_: (stop_all(), sys.exit(130)))
    kill_stale_server(log)
    log(f"시작: {a.cmd or 'run'} (설치 {APP_DIR}, 설정 {CONFIG if os.path.isfile(CONFIG) else '없음 -> 기본값'})")
    try:
        if a.cmd == "check":
            return cmd_check(cfg, log)
        if a.cmd == "flow":
            return cmd_flow(cfg, log, a)
        return cmd_run(cfg, log)
    except Exception as exc:          # 예상 못 한 오류도 사용자에게 한 줄로
        log(f"실행기 오류: {exc!r}")
        error_box("실행기 오류", f"{exc}\n\n로그: {log.path}")
        return 1
    finally:
        stop_all()
        try:
            os.remove(os.path.join(RUN_DIR, "server.pid"))
        except OSError:
            pass
        rotate_logs(cfg["log_dir"], int(cfg.get("keep_logs") or 20))


if __name__ == "__main__":
    sys.exit(main())
