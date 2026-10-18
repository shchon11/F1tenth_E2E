"""What a driver other than our policy is doing, drawn the way that method is drawn (2026-10-18).

The user, watching a baseline drive in the console: "내 정책 말고 다른 방법들 선택할 때 우리정책
planner 추론은 도대체 왜하는데? 각 방법에 맞는 시각화 방식을 적용해놔. 뭐 waypoint를 찍는다던지
방법에 맞는 trajectory를 출력하던지". The console drew our policy's plan over a car that
Follow-the-Gap was steering: a path nobody was driving. Each externally driven car now gets the
picture its own method produces:

* a planner (raceline / ForzaETH spliner / predictive spliner / lane switch): the path it follows --
  the reference line, or the evasion spline or lane it moved onto -- with its speed profile, its
  waypoints, and the point its pure-pursuit tracker is aiming at;
* Follow-the-Gap: the scan it searched after its safety border, the gap it chose at its free-space
  radius, and the best point it steers at (the marker the ForzaETH node publishes);
* the Disparity Extender: the scan with its disparities extended -- the walls it actually avoids --,
  the farthest point it aims at and the free road ahead its speed is read from;
* TinyLidarNet / End2Race: the beams the network reads.

A sensor driver plans no path, so the trajectory drawn for it is the one its command implies: the
(steer, speed) it commanded, rolled out on a kinematic bicycle for one second, and the legend says
so. Nothing here runs a driver again with its state: a stateless method (FTG, the Disparity
Extender) is asked `explain` on the newest scan, which is the decision it acts on next; a network
(End2Race's GRU has memory) is never called twice -- the command it last issued is what is drawn.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

#: [m] of path drawn ahead of a planner, and the points along it (0.25 m apart)
PLAN_HORIZON_M = 12.0
PLAN_POINTS = 49
#: a waypoint dot every this many path points (0.5 m)
WAYPOINT_EVERY = 2
#: the command arc: [s] of travel at the commanded speed, its shortest length [m], its points
ARC_TIME_S = 1.0
ARC_MIN_M = 1.5
ARC_POINTS = 25

# Colours (RGBA). Hue on the path ribbon is speed, as it is for our policy's plan; these stay off it.
TARGET = (0.30, 0.95, 1.00, 1.00)          # the point the method steers at
TARGET_LINE = (0.92, 0.96, 1.00, 0.85)     # from the car (its scanner / its lookahead origin) to it
WAYPOINT = (0.95, 0.95, 0.98, 0.80)        # a planner's waypoints
REF_LINE = (0.62, 0.64, 0.70, 0.70)        # the reference line under an offset path
SCAN_VIEW = (1.00, 0.66, 0.22, 0.70)       # the scan as the method processed it
GAP = (0.35, 1.00, 0.50, 0.95)             # FTG's chosen gap
FORWARD = (0.70, 0.72, 0.78, 0.70)         # the Disparity Extender's free road ahead
NET_BEAM = (0.55, 0.82, 1.00, 0.85)        # a network's input beams

STATE_KO = {"racing": "레이싱 라인 주행", "trailing": "앞차 뒤따르는 중", "overtaking": "추월 중 (회피 스플라인)",
            "changing": "차선 변경 중"}


def external_rows(env) -> Dict[int, Tuple[int, int]]:
    """env row -> (slot, index among that slot's rows), for every car an external driver drives."""
    out: Dict[int, Tuple[int, int]] = {}
    for slot, rows in (getattr(env, "external_slot_rows", None) or {}).items():
        if int(slot) not in (getattr(env, "external_drivers", None) or {}):
            continue
        for j, r in enumerate(rows.tolist()):
            out[int(r)] = (int(slot), j)
    return out


def is_planner(drv) -> bool:
    """A planner ego (`learn.common.PlannerEgoDriver`) is asked through the env, not a scan."""
    return hasattr(drv, "command_env")


def kind_title(drv) -> Tuple[str, str]:
    from ..opponent_slots import kind_of
    kind = str(getattr(drv, "kind", None) or getattr(drv, "KIND", "?"))
    try:
        return kind, kind_of(kind).label
    except ValueError:
        return kind, kind


def command_arc(pose, steer: float, speed: float, wheelbase: float) -> np.ndarray:
    """(ARC_POINTS, 3) world [x, y, v]: the commanded (steer, speed) rolled out on a kinematic
    bicycle from the car's pose for `ARC_TIME_S` (at least `ARC_MIN_M`), at the commanded speed."""
    x, y, yaw = (float(v) for v in pose)
    s = np.linspace(0.0, max(ARC_MIN_M, abs(float(speed)) * ARC_TIME_S), ARC_POINTS)
    k = math.tan(float(steer)) / max(float(wheelbase), 1e-3)
    if abs(k) < 1e-6:
        px, py = x + s * math.cos(yaw), y + s * math.sin(yaw)
    else:
        th = yaw + k * s
        px = x + (np.sin(th) - math.sin(yaw)) / k
        py = y - (np.cos(th) - math.cos(yaw)) / k
    return np.stack([px, py, np.full(ARC_POINTS, float(speed))], 1).astype(np.float32)


def scanner_to_world(pose, mount):
    """A function mapping (k, 2) points in the scanner's frame (x forward, y left) to the world, in
    the plane: the scanner sits at (mount_x, mount_y) on the car, turned by mount_yaw."""
    x, y, yaw = (float(v) for v in pose)
    mx, my, myaw = (float(v) for v in mount)
    c, s = math.cos(yaw), math.sin(yaw)
    ox, oy = x + mx * c - my * s, y + mx * s + my * c
    ca, sa = math.cos(yaw + myaw), math.sin(yaw + myaw)

    def f(p) -> np.ndarray:
        p = np.asarray(p, np.float64).reshape(-1, 2)
        return np.stack([ox + p[:, 0] * ca - p[:, 1] * sa, oy + p[:, 0] * sa + p[:, 1] * ca], 1).astype(np.float32)
    return f


# ----------------------------------------------------------------------------------- planners
def planner_tensors(drv, env, row: int) -> list:
    """Device tensors for one planner-driven row, flattened, for the snapshot's single host copy:
    lookahead origin (2), target (2), path (2n), reference line (2n), profile speed (n), state (1)."""
    import torch
    t = drv.teacher
    base = getattr(t, "base", t)
    frenet = base is not t
    dev = env.sim.state.device
    rows = torch.tensor([int(row)], device=dev)
    offset = getattr(t, "last_offset", None) if frenet else None
    ex = base.explain_rows(env.sim.state, env.sim.P, env.sim.tid, rows, offset=offset,
                           horizon_m=PLAN_HORIZON_M, n=PLAN_POINTS)
    path = ex["line"]
    if frenet:
        d = t.path_offsets(rows, ex["s"], env.sim.state[rows, 3])
        if d is not None:
            path = path + d[..., None] * ex["normal"]
    last = getattr(t, "last_state", None) if frenet else None
    code = last[rows].float() if last is not None else torch.full((1,), -1.0, device=dev)
    return [ex["origin"].reshape(-1).float(), ex["target"].reshape(-1).float(), path.reshape(-1).float(),
            ex["line"].reshape(-1).float(), ex["v"].reshape(-1).float(), code.reshape(-1)]


def planner_floats(n: int = PLAN_POINTS) -> int:
    return 2 + 2 + 2 * n + 2 * n + n + 1


def planner_overlay(drv, flat: np.ndarray, n: int = PLAN_POINTS):
    flat = np.asarray(flat, np.float32)
    o, tgt = flat[0:2], flat[2:4]
    i = 4
    path = flat[i:i + 2 * n].reshape(n, 2); i += 2 * n
    line = flat[i:i + 2 * n].reshape(n, 2); i += 2 * n
    v = flat[i:i + n]; i += n
    code = int(round(float(flat[i])))
    kind, title = kind_title(drv)
    names = getattr(drv.teacher, "STATE_NAMES", None)
    state = STATE_KO.get(names[code], names[code]) if names and 0 <= code < len(names) else "레이싱 라인 주행"
    lines: List[dict] = []
    if float(np.abs(path - line).max()) > 0.02:
        lines.append({"xy": line, "rgba": REF_LINE, "w": 1.5})
    lines.append({"xy": np.stack([o, tgt]), "rgba": TARGET_LINE, "w": 2.0})
    points = [{"xy": path[::WAYPOINT_EVERY], "rgba": WAYPOINT, "px": 5.0},
              {"xy": tgt[None], "rgba": TARGET, "px": 13.0}]
    legend = (f"{title} · {state} · 색 띠 = 따라가는 경로와 속도 프로파일 · 작은 점 = 경로 웨이포인트(0.5 m) · "
              f"큰 점 = 경로 추종기가 겨누는 점" + (" · 회색 = 레이싱 라인" if len(lines) > 1 else ""))
    drv_fr = {"kind": kind, "title": title, "legend": legend, "lines": lines, "points": points}
    return np.concatenate([path, v[:, None]], 1).astype(np.float32), drv_fr


# ----------------------------------------------------------------------------------- sensor drivers
def sensor_overlay(drv, scan_m, speed: float, cmd, pose, mount, range_max: float, wheelbase: float):
    """(path (k, 3) world [x, y, v], overlay dict) for a scan-driven baseline on the focus car."""
    kind, title = kind_title(drv)
    rng = np.nan_to_num(np.asarray(scan_m, np.float32), nan=range_max, posinf=range_max, neginf=0.0)
    a = drv.adapt(np.clip(rng, 0.0, range_max)[None])[0]          # exactly what the driver reads
    w = scanner_to_world(pose, mount)
    origin = np.zeros((1, 2))
    lines: List[dict] = []
    points: List[dict] = []
    if kind == "ftg" and hasattr(drv, "explain"):
        ex = drv.explain(a, speed)
        steer, v = ex["steer"], ex["speed"]
        lines += [{"xy": w(ex["scan_xy"]), "rgba": SCAN_VIEW, "w": 1.5},
                  {"xy": w(ex["gap_xy"]), "rgba": GAP, "w": 3.0},
                  {"xy": w(np.vstack([origin, ex["target"][None]])), "rgba": TARGET_LINE, "w": 2.0}]
        points.append({"xy": w(ex["target"][None]), "rgba": TARGET, "px": 13.0})
        legend = (f"{title} · 주황 = 안전 버블을 씌운 스캔 · 초록 호 = 고른 빈 틈(반경 {ex['radius']:.1f} m) · "
                  f"큰 점 = 목표점 · 색 띠 = 명령(조향·속도)을 1초 굴린 궤적")
    elif kind == "disparity" and hasattr(drv, "explain"):
        ex = drv.explain(a, speed)
        steer, v = ex["steer"], ex["speed"]
        lines += [{"xy": w(ex["scan_xy"]), "rgba": SCAN_VIEW, "w": 1.5},
                  {"xy": w(np.array([[0.0, 0.0], [ex["dist"], 0.0]])), "rgba": FORWARD, "w": 2.0},
                  {"xy": w(np.vstack([origin, ex["target"][None]])), "rgba": TARGET_LINE, "w": 2.0}]
        points.append({"xy": w(ex["target"][None]), "rgba": TARGET, "px": 13.0})
        legend = (f"{title} · 주황 = 디스패리티를 넓힌 스캔 · 큰 점 = 가장 먼 목표점 · 회색 = 속도를 정하는 "
                  f"앞쪽 빈 거리 {ex['dist']:.1f} m" + (" · 옆 벽 때문에 직진" if ex["side_block"] else "")
                  + " · 색 띠 = 명령을 1초 굴린 궤적")
    else:
        steer, v = float(cmd[0]), float(cmd[1])
        out = drv.input_beams(a)
        bearing, r = np.asarray(out[0], np.float64), np.asarray(out[1], np.float64)
        seen = np.asarray(out[2], bool) if len(out) > 2 else np.ones(len(r), bool)
        pts = np.stack([r * np.cos(bearing), r * np.sin(bearing)], 1)[seen]
        points.append({"xy": w(pts), "rgba": NET_BEAM, "px": 4.0})
        legend = (f"{title} · 점 = 망이 읽는 빔 {len(r)}개"
                  + (f"(시야 밖 {int((~seen).sum())}개는 채움값이라 안 그림)" if not seen.all() else "")
                  + " · 색 띠 = 망 출력(조향·속도)을 1초 굴린 궤적. 이 방법은 경로를 계획하지 않습니다")
    drv_fr = {"kind": kind, "title": title, "legend": legend, "lines": lines, "points": points}
    return command_arc(pose, steer, v, wheelbase), drv_fr


class DriverViz:
    """The worker's side: which rows an external driver drives, and their overlay for the focus car."""

    def __init__(self, env):
        self.env = env
        self.rows = external_rows(env)
        v = env.cfg.vehicle
        self.wheelbase = float(v.lf + v.lr)
        self.range_max = float(env.cfg.lidar.range_max)

    def driver(self, row: int):
        hit = self.rows.get(int(row))
        return None if hit is None else self.env.external_drivers[hit[0]]

    def tensors(self, row: int) -> list:
        """What the snapshot copies to the host for this row (empty: not an external row)."""
        drv = self.driver(row)
        if drv is None:
            return []
        if is_planner(drv):
            return planner_tensors(drv, self.env, row)
        return [self.env.last_result.odom[int(row), 3:4].float()]

    def decode(self, row: int, flat: np.ndarray, *, pose, mount, scan_m, cmd):
        drv = self.driver(row)
        if is_planner(drv):
            return planner_overlay(drv, flat)
        return sensor_overlay(drv, scan_m, float(flat[0]), cmd, pose, mount, self.range_max, self.wheelbase)
