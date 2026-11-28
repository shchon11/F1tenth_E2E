"""상자 배치: 저장된 배치(시험 세트 은행의 한 배치) 또는 무작위 배치(시드), 그리고 상자 모양 목록.

모양 목록은 학습 생성기(`procedural_obstacles.build_catalogue`)의 것과 같은 순서다. 그래서 은행 파일의 모양 번호(sid)와
세션 생성기의 번호가 같고, 화면에 그리는 크기가 충돌·LiDAR가 쓰는 크기(선언된 외곽)와 같다.

무작위 배치: 기준 주행선(없으면 매끄럽게 한 중심선)을 상자 수만큼 구간으로 나눠 구간마다 하나, 구간 가운데 60 % 안의
무작위 위치, 옆으로 N(0, 0.15 m)를 ±0.4 m에서 자름(시험 세트 생성기 `exam_line_bank.py`와 같은 규칙). 벽에서 0.25 m보다
가깝거나 출발선 근처(앞 4 m, 뒤 1.5 m)면 다시 뽑는다. **지나갈 수 있는지(최소 시간 경로 판정)는 검사하지 않는다** — 화면에
그렇게 적는다.
"""
from __future__ import annotations

import functools
import math
import os
from typing import Dict, List, Optional

import numpy as np


@functools.lru_cache(maxsize=1)
def catalogue() -> List[dict]:
    from .. import props as _props
    from ..procedural_obstacles import build_catalogue, catalogue_k_pad
    real: List[int] = []
    shapes, _, small = build_catalogue(catalogue_k_pad(), None, real)
    out = []
    for i, sh in enumerate(shapes):
        dims = {k: float(v) for k, v in sh.dims}
        fp = np.asarray(_props.build(sh.style, seed=0, **dims).envelope.footprint, float)
        out.append({"sid": i, "style": sh.style, "dims": dims,
                    "w": float(fp[:, 0].max() - fp[:, 0].min()), "d": float(fp[:, 1].max() - fp[:, 1].min()),
                    "cx": float(0.5 * (fp[:, 0].max() + fp[:, 0].min())), "cy": float(0.5 * (fp[:, 1].max() + fp[:, 1].min())),
                    "h": float(sh.z1), "mass": float(sh.mass), "real": i in real, "small": i in small})
    return out


STYLE_LABEL = {"cardboard_box": "골판지 상자", "wooden_crate": "나무 상자", "steel_drum": "철제 드럼",
               "barrier_block": "방호 블록", "crate_stack_low": "낮은 상자 더미", "marker_post": "표지 기둥"}


def describe(pieces: List[dict]) -> List[dict]:
    """[{sid, x, y, yaw}] -> 그리기·목록용으로 크기와 이름을 붙인다."""
    cat = catalogue()
    out = []
    for i, p in enumerate(pieces):
        c = cat[int(p["sid"])]
        out.append({"i": i, "sid": int(p["sid"]), "x": float(p["x"]), "y": float(p["y"]), "yaw": float(p["yaw"]),
                    "w": c["w"], "d": c["d"], "h": c["h"], "cx": c["cx"], "cy": c["cy"], "style": c["style"],
                    "label": f"{STYLE_LABEL.get(c['style'], c['style'])} {round(c['w'] * 100)}×{round(c['d'] * 100)} cm"})
    return out


# ------------------------------------------------------------------ 저장된 배치(은행)
@functools.lru_cache(maxsize=8)
def _bank(bank_dir: str) -> Optional[dict]:
    if not bank_dir or not os.path.isdir(bank_dir):
        return None
    import torch
    fs = sorted(f for f in os.listdir(bank_dir) if f.endswith(".pt"))
    if not fs:
        return None
    b = torch.load(os.path.join(bank_dir, fs[0]), map_location="cpu", weights_only=False)
    key, e = next(iter(b["per_track"].items()))
    return {"key": key, "sid": e["sid"].numpy().astype(int), "pose": e["pose"].numpy().astype(float),
            "length": float(e.get("length", 0.0)), "file": os.path.join(bank_dir, fs[0])}


def bank_count(bank_dir: str) -> int:
    b = _bank(bank_dir)
    return 0 if b is None else int(b["sid"].shape[0])


def bank_layout(bank_dir: str, index: int, count: Optional[int] = None) -> List[dict]:
    b = _bank(bank_dir)
    if b is None:
        raise ValueError(f"저장된 배치 파일을 읽지 못했습니다: {bank_dir}")
    k = int(index) % b["sid"].shape[0]
    live = np.nonzero(b["sid"][k] >= 0)[0]
    pieces = [{"sid": int(b["sid"][k, j]), "x": float(b["pose"][k, j, 0]), "y": float(b["pose"][k, j, 1]),
               "yaw": float(b["pose"][k, j, 2])} for j in live]
    if count is not None:
        pieces = pieces[:max(0, int(count))]
    return pieces


# ------------------------------------------------------------------ 무작위 배치
def reference_line(spec: str, track) -> np.ndarray:
    """(N, 2) 닫힌 기준선. `spec`: npz 경로(xy) 또는 'centerline'(0.5 m로 매끄럽게 한 중심선)."""
    if spec and spec != "centerline" and os.path.isfile(spec):
        return np.asarray(np.load(spec)["xy"], float)
    cl = np.asarray(track.centerline, float)
    try:
        from ..layout_line import smooth_line
        return np.asarray(smooth_line(cl), float)
    except Exception:
        return cl


def _arc(xy: np.ndarray):
    seg = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]), float(seg.sum())


def _at(xy, s_cum, L, s):
    s = s % L
    j = int(np.searchsorted(s_cum, s, side="right") - 1) % len(xy)
    a, b = xy[j], xy[(j + 1) % len(xy)]
    f = (s - s_cum[j]) / max(s_cum[j + 1] - s_cum[j], 1e-9)
    t = b - a
    t = t / max(np.linalg.norm(t), 1e-9)
    return a + f * (b - a), t


def free_distance(track) -> "callable":
    """빈 바닥 칸에서 가장 가까운 막힌 칸까지 거리[m]를 돌려주는 함수(지도 밖은 0)."""
    from scipy import ndimage
    occ = np.asarray(track.occupancy, bool)
    res = float(track.resolution)
    edt = ndimage.distance_transform_edt(~occ).astype(np.float32) * res
    ox, oy = float(track.origin[0]), float(track.origin[1])

    def f(x: float, y: float) -> float:
        j, i = int(math.floor((x - ox) / res)), int(math.floor((y - oy) / res))
        if 0 <= i < edt.shape[0] and 0 <= j < edt.shape[1]:
            return float(edt[i, j])
        return 0.0
    return f


def random_layout(track, refline: str, n: int, seed: int, start_xy=None, wall_clear: float = 0.25) -> List[dict]:
    n = int(n)
    if n <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    xy = reference_line(refline, track)
    s_cum, L = _arc(xy)
    # 기준선의 s를 코스 중심선 출발선(s=0)에 맞춘다: 출발선에 가장 가까운 기준선 점이 s0
    cl = np.asarray(track.centerline, float)
    p0 = cl[0] if start_xy is None else np.asarray(start_xy, float)
    s0 = float(s_cum[int(np.argmin(np.linalg.norm(xy - p0, axis=1)))])
    dist = free_distance(track)
    cat = [c for c in catalogue() if c["real"]] or [c for c in catalogue() if c["style"] == "cardboard_box"]
    out: List[dict] = []
    for k in range(n):
        for _try in range(200):
            u = rng.uniform(0.2, 0.8)
            s_rel = (k + u) / n * L
            if s_rel < 4.0 or s_rel > L - 1.5:
                continue
            p, t = _at(xy, s_cum, L, s0 + s_rel)
            nrm = np.array([-t[1], t[0]])
            lat = float(np.clip(rng.normal(0.0, 0.15), -0.4, 0.4))
            q = p + lat * nrm
            c = cat[int(rng.integers(len(cat)))]
            half = 0.5 * math.hypot(c["w"], c["d"])
            if dist(q[0], q[1]) < wall_clear + half:
                continue
            if any(math.hypot(q[0] - o["x"], q[1] - o["y"]) < 1.2 for o in out):
                continue
            yaw = float(math.atan2(t[1], t[0]) + rng.uniform(-0.6, 0.6))
            out.append({"sid": int(c["sid"]), "x": float(q[0]), "y": float(q[1]), "yaw": yaw})
            break
    return out


def centerline_s(track, pieces: List[dict]) -> List[float]:
    """상자마다 가장 가까운 중심선 점의 s [m]."""
    cl = np.asarray(track.centerline, float)
    seg = np.linalg.norm(np.diff(np.vstack([cl, cl[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    return [float(s[int(np.argmin(np.linalg.norm(cl - np.array([p["x"], p["y"]]), axis=1)))]) for p in pieces]


def start_line_s(track, pieces: List[dict], behind: float = 2.5, ahead: float = 3.5, step: float = 0.25) -> float:
    """출발선 위치(중심선 s): 0에서 가장 가까운, 상자가 [-behind, +ahead] m 안에 없는 곳.
    시험 세트의 상자는 바퀴 전체에 퍼져 있어 모든 배치에 맞는 고정 출발선이 없다(64배치 192개를 확인) -> 배치마다 고른다."""
    cl = np.asarray(track.centerline, float)
    L = float(np.linalg.norm(np.diff(np.vstack([cl, cl[:1]]), axis=0), axis=1).sum())
    if not pieces:
        return 0.0
    bs = np.asarray(centerline_s(track, pieces))
    best = None
    for k in range(int(L / step)):
        for x in ((k * step) % L, (-k * step) % L):
            d = (bs - x + L / 2) % L - L / 2
            if not ((d > -behind) & (d < ahead)).any():
                return float(x)
            n = int(((d > -behind) & (d < ahead)).sum())
            if best is None or n < best[1]:
                best = (x, n)
    return float(best[0])


def resolve(course: dict, choice: dict, track=None) -> Dict[str, object]:
    """세션 구성의 상자 선택 -> {pieces, source, label}.
    choice: {source: saved|random|none, index, seed, count}"""
    lay = course.get("layouts") or {}
    src = str(choice.get("source", "saved"))
    count = int(choice.get("count", 3))
    if count <= 0 or src == "none":
        return {"pieces": [], "source": "none", "label": "상자 없음"}
    if src == "saved":
        saved = lay.get("saved")
        if not saved:
            raise ValueError("이 코스에는 저장된 상자 배치가 없습니다. 무작위 배치를 고르세요.")
        idx = int(choice.get("index", 0))
        pieces = bank_layout(saved["bank"], idx, count)
        n = bank_count(saved["bank"])
        return {"pieces": pieces, "source": "saved", "index": idx % max(n, 1),
                "label": f"{saved.get('name', '저장된 배치')} {idx % max(n, 1) + 1} / {n}"}
    if src == "random":
        rnd = lay.get("random")
        if not rnd:
            raise ValueError(lay.get("none_reason") or "이 코스에는 무작위 상자 배치 규칙이 없습니다.")
        if track is None:
            from .. import maps
            track = maps.load(maps.smooth_name(course["track"]))
        seed = int(choice.get("seed", 1))
        return {"pieces": random_layout(track, rnd.get("refline", "centerline"), count, seed), "source": "random",
                "seed": seed, "label": f"무작위 {seed}", "unchecked": True}
    raise ValueError(f"모르는 상자 배치 방식 {src!r}")
