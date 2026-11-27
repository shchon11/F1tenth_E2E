"""(11-27, the user: "툭툭 끊기는거 꼴보기 싫은데 몇초전 재생이어도 좋으니까 매끄럽게 좀 보이게좀") an ACE run on /live, smooth:
ace_side/live_tap_log.py on ACE logs every decision of the run's live taps; this pulls the new part of that log every 1.5 s
(through the lab PC, both ssh hops kept open, gzip) and REPLAYS each decision into ~/f1sim_runs/<run>/live_tap*.json at the
moment it was written on ACE plus DELAY seconds (5 s; raised when a pull comes later than that), so the page sees every decision
at its original pace, a few seconds late, and a PPO update still shows as the pause it is. Also fetches args.json and
live_tracks.npz once, starts the logger on ACE if it is not running, and stops when the run on ACE has ended.
usage: ace_live_play.py RUN"""
import collections
import gzip
import os
import shlex
import subprocess
import sys
import threading
import time

NAME = sys.argv[1]
LAB = "shchon11@10.100.0.2"; AB = "/nas/workspace/shchon11_f1sim"; AR = AB + "/runs"
D = os.path.join(os.path.expanduser("~/f1sim_runs"), NAME)
HERE = os.path.dirname(os.path.abspath(__file__))
CM = ["-o", "ControlMaster=auto", "-o", f"ControlPath={os.path.expanduser('~')}/.ssh/cm-ace-play-%r@%h:%p",
      "-o", "ControlPersist=600", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
RC = "ssh -o ControlMaster=auto -o ControlPath=/tmp/cm-ace-play-%r@%h:%p -o ControlPersist=600 -o BatchMode=yes ACE"
DELAY0, DELAY_MAX = 5.0, 30.0


def ace(cmd, inp=None, timeout=60):
    """run `cmd` on ACE through the lab PC -> stdout bytes (b"" on failure)"""
    try:
        return subprocess.run(["ssh", *CM, LAB, f"{RC} {shlex.quote(cmd)}"], input=inp,
                              capture_output=True, timeout=timeout).stdout
    except (subprocess.TimeoutExpired, OSError):
        return b""


def setup():
    os.makedirs(D, exist_ok=True)
    if os.path.exists(os.path.join(D, "args.json")) and not os.path.exists(os.path.join(D, ".from_ace")):
        sys.exit(f"{D} is not an ACE run")
    open(os.path.join(D, ".from_ace"), "a").close()
    for fn in ("live_tap_log.py", "live_log_tail.sh"):              # the ACE-side helpers, beside the runs
        ace(f"mkdir -p {AB}/tools && cat > {AB}/tools/{fn} && chmod +x {AB}/tools/{fn}",
            inp=open(os.path.join(HERE, "ace_side", fn), "rb").read())
    if not os.path.exists(os.path.join(D, "live_tracks.npz")):
        tb = ace(f"cd {AR}/{NAME} && tar -cf - args.json live_tracks.npz", timeout=600)
        subprocess.run(["tar", "-x", "-C", D], input=tb)


def ensure_logger(force=False):
    """start the logger on ACE unless one runs; force: start a new one anyway (it supersedes a hung one through live_log.cur)"""
    start = (f"(cd {AB} && setsid nohup {AB}/venv/bin/python {AB}/tools/live_tap_log.py {AR} {NAME} "
             f">> {AR}/{NAME}.live_log.out 2>&1 < /dev/null &)")
    ace(start if force else f"pgrep -u $(id -u) -f 'live_tap_log.py {AR} {NAME}$' > /dev/null || {start}")


def alive():
    return ace(f"p=$(cat {AR}/{NAME}.pid 2>/dev/null); [ -n \"$p\" ] && ps -p $p > /dev/null && echo y").strip() == b"y"


Q = collections.deque(); lock = threading.Lock(); stop = threading.Event()


def head(raw):
    """(size, log name, seconds since the run's last tap, body) of a live_log_tail.sh reply, or None"""
    try:
        data = gzip.decompress(raw)
        h, body = data.split(b"\n", 1)
        s, n, a = h.decode().split()
        return int(s), n, int(a), body
    except (OSError, ValueError, EOFError):
        return None


def puller():
    off, rest, t_chk, name = 0, b"", time.time(), None
    # start at the log's end: replay from now on, not the run's whole past
    r = head(ace(f"{AB}/tools/live_log_tail.sh {NAME} 999999999999"))
    if r:
        off, name = r[0], r[1]
    t_grow = time.time()
    while not stop.is_set():
        r = head(ace(f"{AB}/tools/live_log_tail.sh {NAME} {off}"))
        if r is None:
            time.sleep(1.5); continue
        size, n, age, body = r
        if n != name:                                               # a new logger (a restart): its log from the start
            name, off, rest = n, 0, b""; t_grow = time.time(); continue
        if size < off:                                              # rotated: read the new file from its start
            off, rest = 0, b""; continue
        if body:
            t_grow = time.time()
        elif age < 10 and time.time() - t_grow > 45:                # the taps change but the log does not: the logger hung
            print(time.strftime("%T"), "logger stalled -> a new one", flush=True)
            ensure_logger(force=True); t_grow = time.time()
        off += len(body)
        buf = rest + body; lines = buf.split(b"\n"); rest = lines.pop()
        fr = []
        for ln in lines:
            p = ln.split(b"\t", 2)
            if len(p) == 3:
                try:
                    fr.append((float(p[0]), int(p[1]), p[2]))
                except ValueError:
                    pass
        with lock:
            Q.extend(fr)
        if time.time() - t_chk > 120:
            t_chk = time.time(); ensure_logger()
            if not alive():
                print(time.strftime("%T"), NAME, "not running on ACE: stop", flush=True); stop.set()
        time.sleep(1.5)


def write(k, body):
    p = os.path.join(D, "live_tap.json" if k == 0 else f"live_tap_{k}.json")
    tmp = p + ".play"
    with open(tmp, "wb") as f:
        f.write(body)
    os.replace(tmp, p)


def player():
    delay = DELAY0
    while not stop.is_set():
        with lock:
            item = Q[0] if Q else None
        if item is None:
            time.sleep(0.05); continue
        t, k, body = item
        due = t + delay - time.time()
        if due > 0:
            time.sleep(min(due, 0.1)); continue
        if -due > 1.0:                                              # a pull came later than the delay covers: wait longer
            delay = min(DELAY_MAX, delay - due + 0.5)
            print(time.strftime("%T"), f"delay -> {delay:.1f} s", flush=True)
        with lock:
            Q.popleft()
            newest = Q[-1][0] if Q else t
        write(k, body)
        # back towards DELAY0 once the pulls keep ahead again: each frame shown 0.05 s early (playback ~12 % fast) while the
        # newest frame already here is more than 2 s inside the delay
        if delay > DELAY0 and time.time() - newest < delay - 2.0:
            delay = max(DELAY0, delay - 0.05)


if __name__ == "__main__":
    setup(); ensure_logger(); time.sleep(2)
    th = threading.Thread(target=puller, daemon=True); th.start()
    player()
