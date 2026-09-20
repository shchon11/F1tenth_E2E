"""Photograph the console for the README, from a real running session.

The previous set of screenshots was produced by a script that lived outside the repository, so
"re-take them" meant finding that script on one particular machine. This is that script, in here,
with no path or device in it that belongs to any one computer.

Nothing is mocked. It builds the real `ConsoleWindow`, wires it to the real
`f1sim.viewer.sim_worker`, waits for the session to actually be driving, and grabs the window. If
a gauge shows a number, the running session produced it.

    # everything, into docs/media (needs a display; use xvfb-run if you have none)
    xvfb-run -s "-screen 0 2048x1280x24" python3 -m scripts.capture_console_media

    python3 scripts/capture_console_media.py --out /tmp/shots --map gen:competition:2
    python3 scripts/capture_console_media.py --checkpoint ~/runs/mine/ppo_latest.pt

By default it renders on the CPU with software GL, so it never competes with a training job for
the GPU; `--gpu` opts out. The figures visible in the panels are then llvmpipe figures and are not
performance numbers, which is what the provenance file it writes says too.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
import platform
import subprocess
import sys
import time
from typing import Callable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

#: Small, always generated, needs no external submodule -- so this runs on a fresh clone.
DEFAULT_MAP = "gen:competition:2"

#: Where the README looks for them.
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "..", "docs", "media")


def sha256(path: str, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


def pick_checkpoint(explicit: str = "") -> str:
    """A checkpoint to drive with: the one asked for, else the repo's own, else the newest run.

    The repository ships two in `checkpoints/`, which is what makes this reproducible for someone
    who has just cloned it and has no runs of their own yet.
    """
    if explicit:
        p = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.exists(p):
            raise SystemExit(f"no checkpoint at {p}")
        return p
    shipped = os.path.join(os.path.dirname(os.path.dirname(HERE)), "checkpoints")
    if os.path.isdir(shipped):
        pts = sorted(f for f in os.listdir(shipped) if f.endswith(".pt"))
        if pts:
            return os.path.join(shipped, pts[0])
    from f1sim.viewer.console.catalog import list_runs
    runs = list_runs()
    if not runs:
        raise SystemExit(
            "no checkpoint found. Pass --checkpoint, or train one first "
            "(the repository normally ships some in checkpoints/).")
    return runs[0].checkpoint


def gl_renderer() -> str:
    """What actually drew the pixels, measured rather than assumed."""
    try:
        out = subprocess.run(["glxinfo", "-B"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "OpenGL renderer string" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown (glxinfo not installed)"


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT, help="directory to write into")
    ap.add_argument("--map", default=DEFAULT_MAP, help=f"track to drive (default {DEFAULT_MAP})")
    ap.add_argument("--checkpoint", default="", help="policy to drive with")
    ap.add_argument("--size", default="1920x1200", help="window size, WxH")
    ap.add_argument("--gpu", action="store_true",
                    help="render and run on the GPU (default: CPU + software GL, so a training "
                         "job keeps its card)")
    ap.add_argument("--settle", type=float, default=12.0,
                    help="seconds of driving before the first frame is kept")
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)

    if not a.gpu:
        # Before anything imports torch or makes a GL context.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise SystemExit(
            "no display. These are screenshots of a real window, so one is required:\n"
            '  xvfb-run -s "-screen 0 2048x1280x24" python3 ' + " ".join(sys.argv))

    width, height = (int(v) for v in a.size.lower().split("x"))
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ckpt = pick_checkpoint(a.checkpoint)

    from PyQt5 import QtCore, QtWidgets
    from f1sim.viewer.console import app as console_app
    from f1sim.viewer.console.protocol import STATE_RUNNING, SessionConfig
    from f1sim.viewer.console.session import SessionController
    from f1sim.viewer.console.window import ConsoleWindow
    import f1sim.viewer.sim_worker as sim_worker

    app = QtWidgets.QApplication.instance() or console_app.create_app(["capture"])
    window = ConsoleWindow()
    window.resize(width, height)
    controller = SessionController(window, worker_target=sim_worker.main)
    window.show()

    def pump(seconds: float = 0.05):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            app.processEvents(QtCore.QEventLoop.AllEvents, 5)
            time.sleep(0.002)

    def wait_for(predicate: Callable[[], bool], what: str):
        end = time.monotonic() + a.timeout
        while time.monotonic() < end:
            if predicate():
                return
            pump(0.02)
        raise SystemExit(f"timed out waiting for {what} (state={window.state})")

    print(f"checkpoint  {ckpt}")
    print(f"map         {a.map}")
    print(f"device      {'gpu' if a.gpu else 'cpu (software GL)'}")

    # Populate the pickers through the window's own API, so the panel names the session that is
    # actually driving rather than contradicting the picture.
    from f1sim.viewer.console.catalog import list_runs
    window.set_runs(list_runs())
    window.run_list.select(ckpt)
    window._on_run_selected(ckpt)

    cfg = SessionConfig(run=ckpt, map_name=a.map, races=1, cars_per_race=1,
                        device="cpu" if not a.gpu else "auto")
    controller.start_worker()
    controller.start_session(cfg)
    wait_for(lambda: window.state == STATE_RUNNING, "the session to start driving")
    print("running; settling…")
    pump(a.settle)

    shots = {}

    def grab(name: str, note: str):
        path = os.path.join(out, name)
        window.grab().save(path)
        shots[name] = {"note": note, "sha256": sha256(path),
                       "bytes": os.path.getsize(path)}
        print(f"  {name}  ({os.path.getsize(path) // 1024} kB)")

    grab("f1tenth-visualizer-overview.png", "whole window, overview camera")

    window._on_camera("chase")
    pump(2.5)
    window.btn_left_panel.click()          # the app's own 설정 패널 toggle
    pump(1.5)
    grab("f1tenth-visualizer-chase.png", "chase camera, setup panel collapsed by the app's toggle")
    window.btn_left_panel.click()
    pump(1.0)

    strip = window.policy_panel
    strip.grab().save(os.path.join(out, "f1tenth-visualizer-policy-panel.png"))
    shots["f1tenth-visualizer-policy-panel.png"] = {
        "note": "detail crop of the policy input/output strip, at that widget's own geometry",
        "sha256": sha256(os.path.join(out, "f1tenth-visualizer-policy-panel.png")),
        "bytes": os.path.getsize(os.path.join(out, "f1tenth-visualizer-policy-panel.png"))}
    print("  f1tenth-visualizer-policy-panel.png")

    controller.stop_session()
    pump(1.5)

    manifest = {
        "captured": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": "f1sim/scripts/capture_console_media.py",
        "argv": sys.argv[1:],
        "checkpoint": {"path": ckpt, "sha256": sha256(ckpt), "bytes": os.path.getsize(ckpt)},
        "map": a.map,
        "window": f"{width}x{height}",
        "device": "gpu" if a.gpu else "cpu",
        "gl_renderer": gl_renderer(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "files": shots,
        "read_honestly": (
            "With --gpu absent these are software-rendered on the CPU. The sim 배속 and 렌더 fps "
            "figures visible in the panels are llvmpipe figures and are NOT performance numbers."),
    }
    mpath = os.path.join(out, "console-capture-manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"manifest    {mpath}")

    window.allow_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
