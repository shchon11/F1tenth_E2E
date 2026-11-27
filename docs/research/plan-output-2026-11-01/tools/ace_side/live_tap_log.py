"""(11-27, the user on /live for an ACE run: "툭툭 끊기는거 꼴보기 싫은데 몇초전 재생이어도 좋으니까 매끄럽게") runs ON ACE beside a
training run: every decision the run's live taps write (live_tap*.json, rewritten ~2.5x a second, atomically) is appended to
<run>/live_log.tsv as "t<TAB>k<TAB>json" (t: ACE's clock, k: the tap's car index), so this PC can replay every decision at its own
pace a few seconds late (ace_live_play.py) instead of the one snapshot a 3 s pull used to catch. Rotates at 40 MB (-> .old).
Stops when the run's process (<runs>/<run>.pid) is gone.
(11-27 12:18) one of these hung in an NFS lookup (state D, nfs_lookup_revalidate) for an hour while the run went on: each logger
now writes its own live_log.<pid>.tsv and names it in live_log.cur; ace_live_play.py starts a new logger when the log stops
growing while the taps still change, and a logger that wakes up to find live_log.cur naming another file exits.
usage: live_tap_log.py RUNS_DIR RUN"""
import glob
import json
import os
import sys
import time

R, NAME = sys.argv[1], sys.argv[2]
D = os.path.join(R, NAME); LOG = os.path.join(D, f"live_log.{os.getpid()}.tsv"); CUR = os.path.join(D, "live_log.cur")
seen, paths, t_glob, t_pid = {}, [], 0.0, 0.0
for old in glob.glob(os.path.join(D, "live_log.*.tsv*")):            # earlier loggers' logs (the hung one's included)
    try:
        os.remove(old)
    except OSError:
        pass
f = open(LOG, "a")
with open(CUR + ".tmp", "w") as g:
    g.write(os.path.basename(LOG))
os.replace(CUR + ".tmp", CUR)
while True:
    now = time.time()
    if now - t_glob > 5:
        paths, t_glob = sorted(glob.glob(os.path.join(D, "live_tap*.json"))), now
        try:
            if open(CUR).read().strip() != os.path.basename(LOG):    # superseded by a newer logger
                break
        except OSError:
            pass
    if now - t_pid > 30:
        t_pid = now
        try:
            os.kill(int(open(os.path.join(R, NAME + ".pid")).read().split()[0]), 0)
        except (OSError, ValueError):
            break
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            continue
        key = (st.st_mtime_ns, st.st_size)
        if seen.get(p) == key:
            continue
        try:
            s = open(p).read()
            json.loads(s)
        except (OSError, ValueError):
            continue
        seen[p] = key
        b = os.path.basename(p)[:-5]
        k = int(b.split("_")[-1]) if b != "live_tap" else 0
        f.write(f"{st.st_mtime:.3f}\t{k}\t{s.replace(chr(10), ' ')}\n")
    f.flush()
    if f.tell() > 40_000_000:
        f.close(); os.replace(LOG, LOG + ".old"); f = open(LOG, "a")
    time.sleep(0.1)
