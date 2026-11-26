"""(11-26, 사용자: "실시간 학습하는 차량 보여주는 거(/live)에서 그 차량에서 실시간으로 디코딩해서 보여주고") /live's state decoder: the
training's tapped car writes what a decoder reads every decision (f1sim/learn/live_tap.py `dec_in`: the features the memory
read, the memory after the decision, the car as the decision found it); here the run's NEWEST decoder (the lab watcher trains
one per exam checkpoint, every 25 updates: so up to ~25 updates older than the live policy) runs on it in numpy
(dec_replay.NPDecoder, the 'feat+mem' set, as the console) and /api/live sends `dec` in `dec_in`'s place:
  walls, boxes: [[[x0, y0, x1, y1, ...], [level of each segment, 0..LEVELS]], ...] the decoded edges as polylines in the
               world (f1sim/learn/dec_shapes.py: where the probability crosses one half, each segment as strong as the
               decoder is sure there, fading out over the grid's last metre -- the edges the console draws); the page
               dashes them on the screen (polylines are ~5x smaller than the console's dashes over Tailscale)
  pose: [x, y, yaw] where the decoder puts our car (null when this track is not the map the decoder was trained on)
  v, r: decoded speed [m/s] and yaw rate [rad/s]; tv, tr: the true ones at the decision
  opp: {"p", "x", "y", "yaw", "fut": [[x, y] 0.5 s, [x, y] 1.0 s]} the other car (P(there) > 0.5 and a decoder trained
       with one, exam4) or null
  u: the decoder's checkpoint update (not drawn)
No decoder for the run (none trained yet, or for other input sizes) -> no `dec`: the page draws nothing extra.
numpy only, like dec_replay (no torch in the dash server); one decision is ~4 M multiply-adds, a few ms."""
import base64
import glob
import json
import math
import os
import re
import sys
import threading
import time

import numpy as np

import dec_replay as DR

HERE = os.path.dirname(os.path.abspath(__file__))
_F1 = os.path.normpath(os.path.join(HERE, "../../../../f1sim"))
if _F1 not in sys.path:
    sys.path.insert(0, _F1)
from f1sim.learn import dec_shapes as DS                         # noqa: E402  (numpy only; f1sim's __init__ is lazy)

RUNS = "/home/shchon11/f1sim_runs"
SET = "feat+mem"
LEVELS = 6                     # the segments' strengths, rounded to this many steps (the page strokes a run of one level at once)
NAME = re.compile(r"^(exam\d+)_([A-Za-z0-9]+)_u(\d+)_.*_dec\.npz$")
_lock = threading.RLock()
_memo = {}


def _cached(key, ttl, fn):
    """fn() kept for `ttl` s under `key` (the decoder listing); the last few keys only"""
    now = time.time()
    with _lock:
        hit = _memo.get(key)
        if hit is not None and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        while len(_memo) > 32:
            _memo.pop(next(iter(_memo)))
        _memo[key] = (now, val)
    return val


def decoder_files(run):
    """[(u, priority, path)] every decoder of `run` on this PC: the console's copy beside the checkpoint
    (~/f1sim_runs/<run>/plan_uN_dec.npz -- the watcher's pick: the exam4 record's when there is one) first, then the exam
    records' (out/decoder/<exam>_<run>_uN_..._dec.npz, exam4 before exam3)"""
    out = []
    for p in glob.glob(os.path.join(RUNS, run, "plan_u*_dec.npz")):
        m = re.fullmatch(r"plan_u(\d+)_dec\.npz", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), 3, p))
    for p in glob.glob(os.path.join(DR.DEC, f"exam*_{run}_u*_dec.npz")):
        m = NAME.match(os.path.basename(p))
        if m and m.group(2) == run:
            out.append((int(m.group(3)), 2 if m.group(1) == "exam4" else 1, p))
    return sorted(out)


def newest(run):
    """the path of the run's newest decoder (highest update, then the console's copy, then exam4), or None"""
    if not re.fullmatch(r"[A-Za-z0-9_]+", run or ""):
        return None
    fs = _cached(("files", run), 30.0, lambda: decoder_files(run))
    return fs[-1][2] if fs else None


def decoder(path):
    """the 'feat+mem' set of a decoder file, kept while the file is unchanged"""
    return DR._cached(DR._dec_cache, ("live", path, os.path.getmtime(path)), lambda: DR.NPDecoder(path, only=(SET,)))


def _track(z, k):
    """track k of a file with the tracks' distance fields (live_tracks.npz, an exam record): (edt (H, W), res, origin)"""
    off, H, W = int(z["t_off"][k]), int(z["t_H"][k]), int(z["t_W"][k])
    r = np.asarray(z["t_res"]).reshape(-1)
    org = np.asarray(z["t_origin"]).reshape(-1, 2)
    return z["edt"][off:off + H * W].reshape(H, W), float(r[k] if r.size > k else r[0]), org[k] if len(org) > k else org[0]


def _wall(g, res, org, x, y):
    c = np.floor((x - org[0]) / res).astype(np.int64); r = np.floor((y - org[1]) / res).astype(np.int64)
    ok = (c >= 0) & (c < g.shape[1]) & (r >= 0) & (r < g.shape[0])
    w = np.ones(x.shape, bool)
    w[ok] = g[r[ok], c[ok]] <= 0
    return w


def same_map(d, run, tid):
    """whether track `tid` of the live run is the map the decoder was trained on (its pose is in that map's frame): the
    lanes of the two agree (IoU >= 0.85 on a 0.1 m lattice; the 17 hose variants of korea26scan are 0.96-0.98 against
    the bag map). Without the decoder's record here: the two extents within 1.5 m of each other on every side."""
    def check():
        lz = np.load(os.path.join(RUNS, run, "live_tracks.npz"))
        g, res, org = _track(lz, int(tid))
        ext = (org[0], org[0] + g.shape[1] * res, org[1], org[1] + g.shape[0] * res)
        rec = d.record[:-4] if d.record.endswith(".npz") else d.record
        rp = DR.record_path(rec) if rec and re.fullmatch(r"[A-Za-z0-9_]+", rec) else None
        if rp is None:
            return bool(max(abs(float(a) - float(b)) for a, b in zip(d.pose_ext, ext)) <= 1.5)
        rz = np.load(rp, allow_pickle=False)
        g0, res0, org0 = _track(rz, 0)
        xs, ys = np.meshgrid(np.arange(org0[0], org0[0] + g0.shape[1] * res0, 0.1),
                             np.arange(org0[1], org0[1] + g0.shape[0] * res0, 0.1))
        a, b = ~_wall(g0, res0, org0, xs, ys), ~_wall(g, res, org, xs, ys)
        return bool((a & b).sum() >= 0.85 * max(1, (a | b).sum()))
    return _cached(("map", run, int(tid), d.path), 3600.0, check)


def _lines(prob, grid, pose):
    """a decoded grid's edges -> [[[x0, y0, ...] world, [level per segment]], ...]; the weak ends trimmed, all-weak dropped"""
    segs, st, keys = DS.edges(prob, grid)
    out = []
    for pts, w in DS.chains(segs, st, keys):
        lv = np.round(0.5 * (w[1:] + w[:-1]) * LEVELS).astype(int)
        nz = np.nonzero(lv > 0)[0]
        if not len(nz):
            continue
        a, b = int(nz[0]), int(nz[-1]) + 1
        out.append([np.round(DS.to_world(pts[a:b + 1], pose).reshape(-1).astype(np.float64), 2).tolist(), lv[a:b].tolist()])
    return out


def decode(run, now):
    """the `dec` dict for one tapped decision (`now` = live_tap.json's "now"), or None"""
    din = now.get("dec_in")
    path = newest(run) if din else None
    if not path:
        return None
    d = decoder(path)
    f = np.frombuffer(base64.b64decode(din["f"]), "<f4")
    m = np.frombuffer(base64.b64decode(din["m"]), "<f4")
    st = [float(v) for v in din["st"]]
    if SET not in d.sets or f.size != d.feat_dim or m.size != d.mem_dim or len(st) != 5:
        return None
    o = d.run(f[None].astype(np.float32), m[None].astype(np.float32), SET)
    sig = lambda v: 1.0 / (1.0 + np.exp(-np.clip(np.asarray(v, np.float64), -30, 30)))
    gr = [float(v) for v in d.grid[:5]]
    nx, ny = int(round((gr[1] - gr[0]) / gr[4])), int(round((gr[3] - gr[2]) / gr[4]))
    pose = st[:3]
    ego = o["ego"][0] * d.ego_scale
    out = {"u": int(re.search(r"_u(\d+)_", os.path.basename(path) + "_").group(1)),
           "walls": _lines(sig(o["wall"][0]).reshape(nx, ny), gr, pose), "boxes": _lines(sig(o["box"][0]).reshape(nx, ny), gr, pose),
           "lv": LEVELS, "pose": None,
           "v": round(float(ego[0]), 3), "r": round(float(ego[2]), 4), "tv": round(st[3], 3), "tr": round(st[4], 4), "opp": None}
    try:
        if same_map(d, run, now.get("tid", 0)):
            x0, x1, y0, y1 = (float(v) for v in d.pose_ext)
            p = o["pose"][0]
            out["pose"] = [round(x0 + float(p[0]) * (x1 - x0), 3), round(y0 + float(p[1]) * (y1 - y0), 3),
                           round(math.atan2(float(p[3]), float(p[2])), 4)]
    except (OSError, KeyError, ValueError, IndexError):
        pass
    if d.has_opp:
        q = o["opp"][0]
        po = float(sig(q[0]))
        if po > 0.5:
            S = d.opp_scale
            loc = DS.to_world(np.array([[q[1] * S, q[2] * S], [q[6] * S, q[7] * S], [q[8] * S, q[9] * S]]), pose)
            out["opp"] = {"p": round(po, 3), "x": round(float(loc[0, 0]), 3), "y": round(float(loc[0, 1]), 3),
                          "yaw": round(pose[2] + math.atan2(float(q[4]), float(q[3])), 4),
                          "fut": np.round(loc[1:].astype(np.float64), 3).tolist()}
    return out


_body = {}


def live_body(run, path):
    """live_tap.json's bytes for /api/live, `dec_in` swapped for `dec` (kept per file version: the page polls every 0.1 s)"""
    stt = os.stat(path)
    key = (path, stt.st_mtime_ns, stt.st_size)
    with _lock:
        if key in _body:
            return _body[key]
    raw = open(path, "rb").read()
    body = raw
    if b'"dec_in"' in raw:
        try:
            j = json.loads(raw)
            now = j.get("now") or {}
            dec = None
            try:
                dec = decode(run, now)
            except Exception:                    # a decoder that does not fit must not take the live view down
                dec = None
            now.pop("dec_in", None)
            if dec is not None:
                now["dec"] = dec
            body = json.dumps(j, separators=(",", ":"), ensure_ascii=False).encode()
        except ValueError:
            body = raw
    with _lock:
        for k in [k for k in _body if k[0] == path]:
            _body.pop(k)
        while len(_body) > 16:
            _body.pop(next(iter(_body)))
        _body[key] = body
    return body
