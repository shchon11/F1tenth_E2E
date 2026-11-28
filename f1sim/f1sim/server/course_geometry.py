"""코스를 화면에 그릴 기하(M1): 덕트 호스 = 둥근 관의 중심선, 호스가 아닌 벽, 달릴 수 있는 바닥, 미니맵 윤곽, 구간 경계.

물리·LiDAR와 같은 기하를 쓴다(PM 결정 8). 세션이 쓰는 지도는 `+duct`(호스를 제자리에 다시 깖) + `+smooth`(5 cm 격자
계단 대신 매끄러운 곡선을 1.25 cm 격자로 다시 그림, `Track.with_smooth_walls`)이고, 차가 닿고 LiDAR가 맞는 면은 **차로의
경계**다. 여기서는 그 경계를 따라 바깥(막힌 쪽)으로 반지름만큼 물러난 곳을 관의 중심으로 둔다. 그러면 관의 차로 쪽 면이 곧
물리의 벽이다(그린 것 = 닿는 것).

- 관 지름: 코스 파일의 `duct_diameter`(기본 0.30 m). 경계 뒤의 막힌 두께가 지름보다 얇으면(두 차로 사이 칸막이 호스)
  그 두께를 지름으로 쓴다.
- 경계 뒤가 호스 칸(`track.duct`)이 아니고 벽 칸(`track.tall`)이면 관 대신 얇은 벽판으로 보낸다.
- 지도 밖 경계(차로가 아닌 빈 칸 주머니)는 그리지 않는다.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def _lane_mask(track) -> np.ndarray:
    """차가 달릴 수 있는 칸: 중심선이 지나는 빈 칸 연결 영역."""
    from scipy import ndimage
    occ = np.asarray(track.occupancy, bool)
    res = float(track.resolution)
    lab, _ = ndimage.label(~occ)
    cl = np.asarray(track.centerline, float)
    j = np.clip(((cl[:, 0] - track.origin[0]) / res).astype(int), 0, occ.shape[1] - 1)
    i = np.clip(((cl[:, 1] - track.origin[1]) / res).astype(int), 0, occ.shape[0] - 1)
    ids = np.unique(lab[i, j])
    ids = ids[ids > 0]
    return np.isin(lab, ids)


def _resample(xy: np.ndarray, step: float, closed: bool) -> np.ndarray:
    pts = np.vstack([xy, xy[:1]]) if closed else xy
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    n = max(4, int(round(L / step)))
    q = np.linspace(0.0, L, n, endpoint=not closed)
    return np.stack([np.interp(q, s, pts[:, 0]), np.interp(q, s, pts[:, 1])], 1)


def _smooth(xy: np.ndarray, w: int, closed: bool) -> np.ndarray:
    if w <= 1:
        return xy
    k = np.ones(w) / w
    if closed:
        pad = np.vstack([xy[-w:], xy, xy[:w]])
        out = np.stack([np.convolve(pad[:, c], k, mode="same") for c in range(2)], 1)
        return out[w:-w]
    return xy


def _gauss(xy: np.ndarray, sigma_pts: float, closed: bool) -> np.ndarray:
    """닫힌 고리를 가우스로 매끄럽게(1.25 cm 격자 윤곽의 계단 잔물결을 없앤다). 반지름 0.3 m 곡선에서 1 mm 안쪽으로만 움직인다."""
    if sigma_pts <= 0 or not closed:
        return xy
    from scipy.ndimage import gaussian_filter1d
    return np.stack([gaussian_filter1d(xy[:, c], sigma_pts, mode="wrap") for c in range(2)], 1)


def _normals(xy: np.ndarray, closed: bool, k: int = 3) -> np.ndarray:
    nxt = np.roll(xy, -k, 0) if closed else np.vstack([xy[k:], np.repeat(xy[-1:], k, 0)])
    prv = np.roll(xy, k, 0) if closed else np.vstack([np.repeat(xy[:1], k, 0), xy[:-k]])
    t = nxt - prv
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return np.stack([-t[:, 1], t[:, 0]], 1)


def _sample(mask: np.ndarray, track, xy: np.ndarray) -> np.ndarray:
    res = float(track.resolution)
    j = np.floor((xy[:, 0] - track.origin[0]) / res).astype(int)
    i = np.floor((xy[:, 1] - track.origin[1]) / res).astype(int)
    ok = (i >= 0) & (i < mask.shape[0]) & (j >= 0) & (j < mask.shape[1])
    out = np.zeros(len(xy), bool)
    out[ok] = mask[i[ok], j[ok]]
    return out


def _runs(flag: np.ndarray, closed: bool) -> List[tuple]:
    """flag가 참인 연속 구간 [(a, b)] (b 포함 안 함). 닫힌 고리 전체면 [(0, n, True)]."""
    n = len(flag)
    if flag.all():
        return [(0, n, True)]
    if not flag.any():
        return []
    out = []
    start = None
    if closed:
        k0 = int(np.argmin(flag))          # 거짓 위치에서 시작해 한 바퀴
        order = [(k0 + i) % n for i in range(n)]
    else:
        order = list(range(n))
    for idx in order + [None]:
        on = idx is not None and flag[idx]
        if on and start is None:
            start = idx
            prev = idx
        elif on:
            prev = idx
        elif start is not None:
            out.append((start, prev, False))
            start = None
    return out


def _take(arr: np.ndarray, a: int, b: int, n: int) -> np.ndarray:
    if b >= a:
        return arr[a:b + 1]
    return np.concatenate([arr[a:], arr[:b + 1]], 0)


def build(track, duct_diameter: float = 0.30, step: float = 0.03, n_sectors: int = 3) -> Dict[str, object]:
    from skimage.measure import find_contours
    res = float(track.resolution)
    lane = _lane_mask(track)
    occ = np.asarray(track.occupancy, bool)
    duct = np.asarray(track.duct, bool) if track.duct is not None else occ
    free = ~occ
    hoses, walls, outline = [], [], []
    R0 = 0.5 * float(duct_diameter)
    max_probe = int(math.ceil((2 * R0 + 0.05) / 0.01))
    for c in find_contours(np.pad(lane.astype(np.float32), 1), 0.5):
        xy = np.stack([track.origin[0] + (c[:, 1] - 0.5) * res, track.origin[1] + (c[:, 0] - 0.5) * res], 1)
        closed = bool(np.linalg.norm(xy[0] - xy[-1]) < 2 * res)
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        if seg < 1.0:
            continue
        xy = _gauss(_resample(xy[:-1] if closed else xy, step, closed), 0.05 / step, closed)
        nrm = _normals(xy, closed)
        # 법선을 막힌 쪽으로: 2 cm 앞이 차로면 뒤집는다
        probe = xy + 0.02 * nrm
        flip = _sample(lane, track, probe)
        nrm[flip] *= -1.0
        # 경계 뒤 막힌 두께(다른 빈 칸을 만날 때까지, 최대 지름 + 5 cm)
        thick = np.full(len(xy), 2 * R0 + 0.05)
        alive = np.ones(len(xy), bool)
        for k in range(2, max_probe + 1):
            q = xy + (0.01 * k) * nrm
            hit = _sample(free, track, q) & alive
            thick[hit] = 0.01 * k
            alive &= ~hit
            if not alive.any():
                break
        is_hose = _sample(duct, track, xy + 0.02 * nrm) | ~_sample(occ, track, xy + 0.02 * nrm)
        r = np.where(thick < 2 * R0, 0.5 * thick, R0)
        r = np.clip(r, 0.04, R0)
        r = _gauss(np.stack([r, r], 1), 0.15 / step, closed)[:, 0] if closed else r   # 반지름도 천천히 바뀌게
        centre = xy + nrm * r[:, None]
        # 관 중심선을 한 번 더 매끄럽게(칸막이 끝처럼 반지름보다 좁게 꺾이는 곳에서 관이 접히지 않게) 하고, 반지름은
        # 중심에서 가장 가까운 경계까지의 거리로 다시 잰다: 관의 차로 쪽 면이 경계(물리의 벽)에 닿고 넘지 않는다
        if closed:
            from scipy.spatial import cKDTree
            centre = _gauss(centre, 0.10 / step, closed)
            d, _ = cKDTree(xy).query(centre)
            r = np.clip(np.minimum(r, d), 0.04, R0)
        outline.append({"xy": xy[::3].astype(np.float32), "closed": closed})
        for a, b, whole in _runs(is_hose, closed):
            if whole:
                hoses.append({"xy": centre.astype(np.float32), "r": r.astype(np.float32), "closed": True})
            else:
                seg_c = _take(centre, a, b, len(xy))
                if len(seg_c) >= 3:
                    hoses.append({"xy": seg_c.astype(np.float32), "r": _take(r, a, b, len(xy)).astype(np.float32),
                                  "closed": False})
        for a, b, whole in _runs(~is_hose, closed):
            pts = xy if whole else _take(xy, a, b, len(xy))
            if len(pts) >= 3:
                walls.append({"xy": pts[::2].astype(np.float32), "closed": bool(whole)})
    # 바닥 마스크(5 cm 칸, u1: 1 = 달릴 수 있는 면) -- 바닥 셰이더와 미니맵이 쓴다
    f = max(1, int(round(0.05 / res)))
    H, W = lane.shape
    Hc, Wc = H // f, W // f
    lane_c = lane[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f).mean((1, 3)) > 0.5
    cl = np.asarray(track.centerline, float)
    seg = np.linalg.norm(np.diff(np.vstack([cl, cl[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    sectors = []
    for k in range(1, int(n_sectors)):
        sk = L * k / n_sectors
        j = int(np.searchsorted(s, sk, side="right") - 1) % len(cl)
        a_, b_ = cl[j], cl[(j + 1) % len(cl)]
        t = (b_ - a_) / max(np.linalg.norm(b_ - a_), 1e-9)
        p = a_ + (sk - s[j]) * t
        sectors.append({"k": k, "s": sk, "x": float(p[0]), "y": float(p[1]), "yaw": float(math.atan2(t[1], t[0]))})
    xs = np.concatenate([o["xy"][:, 0] for o in outline]) if outline else cl[:, 0]
    ys = np.concatenate([o["xy"][:, 1] for o in outline]) if outline else cl[:, 1]
    return {
        "hoses": hoses, "walls": walls, "outline": outline,
        "lane": {"data": lane_c.astype(np.uint8), "origin": [float(track.origin[0]), float(track.origin[1])],
                 "cell": float(res * f), "shape": [int(Hc), int(Wc)]},
        "sectors": sectors, "length_m": L,
        "content_bounds": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        "duct_diameter": float(duct_diameter),
    }
