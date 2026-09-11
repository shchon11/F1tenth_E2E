"""The shipped entry point, launched as a real process.

Everything else about the console is tested in-process, which is faster and more precise but shares
the test runner's interpreter. This one starts `python -m f1sim.learn.watch` the way a person does,
waits for a window, screenshots it, and checks that a terminal Ctrl+C closes it cleanly and leaves
no worker behind. The bugs it catches are the ones that only exist in the real path: a name that is
not in scope at launch, a signal Qt never delivers to Python, a teardown order that aborts.

Skipped without a display or the X tools it needs.
"""
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

pytestmark = pytest.mark.skipif(
    not os.environ.get("DISPLAY") or not shutil.which("xdotool"),
    reason="needs an X display and xdotool")


def _child_pids(pid):
    out = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)],
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


@pytest.fixture
def console():
    # This test is about a REAL X window -- `wait_for_window` looks the console up with `xdotool`.
    # The platform therefore has to be `xcb` explicitly rather than inherited: other console test
    # modules set `QT_QPA_PLATFORM=offscreen` on the shared process environment, and a child built
    # from `os.environ` then opens no window at all and this fails with "no window appeared" after
    # a 90 s wait. Those modules restore it now, but inheriting a display-dependent setting is the
    # fragile part, so this states what it needs.
    env = dict(os.environ, PYTHONPATH=REPO, QT_QPA_PLATFORM="xcb")
    proc = subprocess.Popen([PYTHON, "-m", "f1sim.learn.watch"], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def wait_for_window(proc, timeout=90):
    """Wait for a window belonging to THIS process.

    `xdotool search --name f1sim` matches by title across the whole display, so it can return a
    window opened by another test's console, or by a real desktop session on the same X server --
    and then this passes while the process under test never opened anything. `--pid` scopes the
    search to the child actually launched here.
    """
    end = time.time() + timeout
    while time.time() < end:
        r = subprocess.run(["xdotool", "search", "--pid", str(proc.pid), "--name", "f1sim"],
                           capture_output=True, text=True)
        if r.stdout.strip():
            return r.stdout.strip().splitlines()[-1], time.time() - (end - timeout)
        if proc.poll() is not None:
            pytest.fail(f"console exited early ({proc.returncode}):\n{proc.stdout.read()[-4000:]}")
        time.sleep(0.25)
    pytest.fail("no window appeared")


def test_no_argument_entry_point_opens_the_console(console, tmp_path):
    """`python -m f1sim.learn.watch` with no arguments is the console, not the old Tk launcher."""
    win, _ = wait_for_window(console)
    assert win
    time.sleep(5)                        # let the worker hand over the map catalogue
    shot = tmp_path / "console.png"
    if shutil.which("import"):
        subprocess.run(["import", "-window", "root", str(shot)], check=True)
        assert shot.exists() and shot.stat().st_size > 10_000
    assert console.poll() is None, "console should still be running"


def test_ctrl_c_closes_cleanly_and_leaves_no_worker(console):
    """A program people start from a shell has to answer Ctrl+C.

    Qt's loop sits in C, so a Python signal handler is not run until the interpreter is entered
    again -- without the timer that makes that happen, SIGINT was simply ignored and the process
    had to be killed. And quitting from the handler without closing the window aborted during
    teardown, so this also checks the exit status, not merely that it stopped.
    """
    wait_for_window(console)
    time.sleep(4)
    workers = _child_pids(console.pid)
    assert workers, "the console should have spawned a simulation worker"

    console.send_signal(signal.SIGINT)
    end = time.time() + 40
    while console.poll() is None and time.time() < end:
        time.sleep(0.25)
    assert console.poll() is not None, "did not exit on SIGINT"
    assert console.returncode == 0, f"unclean exit {console.returncode}:\n{console.stdout.read()[-4000:]}"

    # and its worker went with it -- exactly the process it started, no orphan
    time.sleep(1.0)
    for pid in workers:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
