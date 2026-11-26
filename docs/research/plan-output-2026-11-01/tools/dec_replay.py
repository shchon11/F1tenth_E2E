"""(11-26) the site's exam-replay decoder view (/dec, dec_page.html): the decoders state_decoder_train.py trained on an exam record
(out/decoder/<record>_dec.npz, synced by lab_decoder_watch.sh) run here in numpy on one session's policy state
(<record>_decin.npz: feat and mem of the decoder's held-out test sessions), next to that session's truth from the record.
numpy only (no torch in the dash server): one session is ~900 decisions x ~4 M multiply-adds, a fraction of a second."""
import base64
import glob
import io
import json
import math
import os
import re
import threading

import numpy as np

R = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01"
OUT = R + "/out"
DEC = OUT + "/decoder"
NAME = re.compile(r"^(exam\d+)_([A-Za-z0-9]+)_u(\d+)_.*$")
LF, LR = 0.15875, 0.17145           # the exam's car (f1sim.params.VehicleParams), ideal execution = kinematic bicycle
WB = LF + LR


def ok_rec(rec):
    return bool(re.fullmatch(r"[A-Za-z0-9_]+", rec or "")) and os.path.isfile(f"{DEC}/{rec}_dec.npz") and \
        os.path.isfile(f"{DEC}/{rec}_decin.npz")


def records():
    """[{rec, run, u, exam}] every record with a decoder here, newest checkpoint first within a run"""
    out = []
    for p in glob.glob(f"{DEC}/*_dec.npz"):
        rec = os.path.basename(p)[:-8]
        m = NAME.match(rec)
        if not m or not os.path.isfile(f"{DEC}/{rec}_decin.npz"):
            continue
        out.append({"rec": rec, "exam": m.group(1), "run": m.group(2), "u": int(m.group(3)), "mtime": os.path.getmtime(p)})
    runs_t = {}
    for r in out:
        runs_t[r["run"]] = max(runs_t.get(r["run"], 0), r["mtime"])
    out.sort(key=lambda r: (-runs_t[r["run"]], r["run"], -r["u"], r["exam"]))
    return out


def record_path(rec):
    for p in (f"{OUT}/exam/{rec}.npz", f"{DEC}/{rec}.npz"):
        if os.path.isfile(p):
            return p
    return None


class NPDecoder:
    """the decoder sets of one record, in numpy"""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.pose_ext = z["pose_ext"].astype(np.float64)
        self.ego_scale = z["ego_scale"].astype(np.float32)
        self.grid = z["grid"].astype(np.float64)
        self.towers = json.loads(str(z["towers"]))
        self.opp_scale = float(z["opp_scale"])
        self.box_grow = float(z["box_grow"])
        self.has_opp = bool(z["has_opp"])
        self.sets = {}
        for k in z.files:
            if "|" in k:
                s, n = k.split("|", 1)
                self.sets.setdefault(s, {})[n] = z[k].astype(np.float32)

    def run(self, feat, mem, which):
        w = self.sets[which]
        x = feat if which == "feat" else (mem if which == "mem" else np.concatenate([feat, mem], 1))
        x = (x.astype(np.float32) - w["mu"]) / w["sd"]
        out = {}
        for t, outs in self.towers:
            h = np.maximum(x @ w[f"towers.{t}.0.weight"].T + w[f"towers.{t}.0.bias"], 0.0)
            h = np.maximum(h @ w[f"towers.{t}.2.weight"].T + w[f"towers.{t}.2.bias"], 0.0)
            for o in outs:
                out[o] = h @ w[f"heads.{o}.weight"].T + w[f"heads.{o}.bias"]
        return out


_dec_cache, _in_cache = {}, {}
_lock = threading.RLock()                    # the dash server answers on threads


def _cached(cache, key, fn, keep=4):
    with _lock:
        if key not in cache:
            while len(cache) >= keep:
                cache.pop(next(iter(cache)))
            cache[key] = fn()
        return cache[key]


def decoder(rec):
    p = f"{DEC}/{rec}_dec.npz"
    return _cached(_dec_cache, (rec, os.path.getmtime(p)), lambda: NPDecoder(p))


def decin(rec):
    p = f"{DEC}/{rec}_decin.npz"

    def load():
        z = np.load(p, allow_pickle=False)
        return {"feat": z["feat"], "mem": z["mem"], "sessions": z["sessions"].astype(int)}
    return _cached(_in_cache, (rec, os.path.getmtime(p)), load, keep=2)


def track_png(z):
    """the record's walls as a PNG data URL (lane 246 / half a car width from a wall 160 / wall 70, as /api/live_track)"""
    from PIL import Image
    off, H, W = int(z["t_off"][0]), int(z["t_H"][0]), int(z["t_W"][0])
    res = float(np.asarray(z["t_res"]).reshape(-1)[0]); org = np.asarray(z["t_origin"]).reshape(-1, 2)[0]
    g = z["edt"][off:off + H * W].reshape(H, W)
    car = np.asarray(z["car"], float).tolist() if "car" in z.files else [0.58, 0.31]
    img = np.where(g <= 0, 70, np.where(g <= 0.5 * car[1], 160, 246)).astype(np.uint8)[::-1]
    buf = io.BytesIO(); Image.fromarray(img, "L").save(buf, "PNG")
    return {"png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "x0": float(org[0]), "y0": float(org[1]),
            "res": res, "W": W, "H": H, "car": car}


def meta(rec):
    """the record's track, box shapes, the decoder's grid and the test sessions it can replay"""
    z = np.load(record_path(rec), allow_pickle=False)
    d = decoder(rec)
    ses = decin(rec)["sessions"]
    sid = z["sid"]; on = z["on"] if "on" in z.files else np.zeros(sid.shape[:2], bool)
    rows = []
    for s in ses:
        r = {"s": int(s), "box": bool((sid[0, s] >= 0).any()), "opp": bool(on[:, s].any())}
        if "c_pace" in z.files:
            r["pace"] = float(np.asarray(z["c_pace"]).reshape(-1)[s])
        rows.append(r)
    out = {"rec": rec, "T": int(z["rs"].shape[0]), "dt": float(z["dt"]), "track": track_png(z),
           "fp": {k[3:]: np.asarray(z[k]).round(3).tolist() for k in z.files if k.startswith("fp_")},
           "grid": d.grid.tolist(), "sessions": rows, "sets": list(d.sets.keys()), "checkpoint": str(z["name"]),
           "has_opp": d.has_opp}
    return out


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def session(rec, s, which):
    """one test session: truth per decision and the decoder's answer (`which` input set)"""
    z = np.load(record_path(rec), allow_pickle=False)
    d = decoder(rec)
    di = decin(rec)
    k = np.nonzero(di["sessions"] == s)[0]
    if not len(k) or which not in d.sets:
        raise ValueError("no such session / input set")
    k = int(k[0])
    feat, mem = di["feat"][:, k].astype(np.float32), di["mem"][:, k].astype(np.float32)
    T = feat.shape[0]
    st = z["st"][:T, s].astype(np.float64); vy = z["vy"][:T, s].astype(np.float64)
    vx = st[:, 3]; okv = np.abs(vx) > 0.5
    tan_d = np.where(okv, WB / LR * vy / np.where(okv, vx, 1.0), 0.0)
    r_true = np.where(okv, vx * np.cos(np.arctan(np.where(okv, vy / np.where(okv, vx, 1.0), 0.0))) * tan_d / WB, np.nan)
    o = d.run(feat, mem, which)
    x0, x1, y0, y1 = d.pose_ext
    p = o["pose"]
    pose = np.stack([x0 + p[:, 0] * (x1 - x0), y0 + p[:, 1] * (y1 - y0), np.arctan2(p[:, 3], p[:, 2])], 1)
    ego = o["ego"] * d.ego_scale
    sig = lambda v: 1.0 / (1.0 + np.exp(-np.clip(v, -30, 30)))
    q = o["opp"]; S = d.opp_scale
    opp_dec = np.stack([sig(q[:, 0]), q[:, 1] * S, q[:, 2] * S, np.arctan2(q[:, 4], q[:, 3]), q[:, 5] * S,
                        q[:, 6] * S, q[:, 7] * S, q[:, 8] * S, q[:, 9] * S], 1)
    nx, ny = int(round((d.grid[1] - d.grid[0]) / d.grid[4])), int(round((d.grid[3] - d.grid[2]) / d.grid[4]))
    wall = np.round(sig(o["wall"]) * 255).astype(np.uint8).reshape(T, nx, ny)
    box = np.round(sig(o["box"]) * 255).astype(np.uint8).reshape(T, nx, ny)
    # boxes: one entry per slot and stretch it stays put [sid, x, y, yaw, t0, t1)
    sid = z["sid"][:T + 1, s]; pz = z["pose"][:T + 1, s]
    boxes = []
    for j in range(sid.shape[1]):
        t = 0
        while t < len(sid):
            if sid[t, j] < 0:
                t += 1; continue
            t0 = t
            while t + 1 < len(sid) and sid[t + 1, j] == sid[t0, j] and np.abs(pz[t + 1, j, :2] - pz[t0, j, :2]).sum() < 0.05:
                t += 1
            boxes.append([int(sid[t0, j]), round(float(pz[t0, j, 0]), 3), round(float(pz[t0, j, 1]), 3), round(float(pz[t0, j, 2]), 3), t0, t + 1])
            t += 1
    opp = None
    if "op" in z.files and z["on"][:T, s].any():
        op = z["op"][:T, s]
        opp = np.round(np.concatenate([z["on"][:T, s, None].astype(float), op], 1), 3).tolist()
    rnd = lambda a, n=3: np.where(np.isfinite(a), np.round(a, n), None).tolist()
    return {"T": T, "dt": float(z["dt"]), "s": int(s), "set": which, "grid": [float(v) for v in d.grid] + [nx, ny],
            "car": rnd(np.stack([st[:, 0], st[:, 1], st[:, 2], vx, r_true], 1)),
            "dec_pose": rnd(pose), "dec_ego": rnd(ego), "opp": opp, "dec_opp": rnd(opp_dec),
            "boxes": boxes, "wall": b64(wall), "box": b64(box)}


_ses_cache = {}


def session_json(rec, s, which):
    """`session` as JSON text, the last few kept (one is ~5 MB before the server gzips it)"""
    key = (rec, int(s), which, os.path.getmtime(f"{DEC}/{rec}_dec.npz"))
    return _cached(_ses_cache, key, lambda: json.dumps(session(rec, int(s), which), separators=(",", ":")), keep=6)
