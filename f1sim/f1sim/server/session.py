"""시뮬 세션 하나: 기존 f1sim 코어(Simulator) + 드라이버 + 랩 계시. 렌더·UI·학습 개념 없음.

스파이크 범위: 코스 1개, 차 N대(기본 1), 드라이버 = Follow-the-Gap(저장소의 ForzaETH 이식본, GPU 불필요).
기존 파일은 고치지 않는다. Config는 코드에서 만든 기본값에 스파이크용 세 가지만 바꾼다:
  - device cpu, compile 끔(대신 `compile_cpu=True`면 물리 루프만 inductor CPU로 컴파일)
  - terminate_on_collision False: 닿아도 세션이 끝나지 않는다(연속 접촉 모델 그대로)
  - rand.enabled False: 명목 차량(무작위화 없음)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .. import maps
from ..params import Config
from ..sim import Simulator
from . import SCHEMA_VERSION


@dataclass
class LapClock:
    """랩 계시. 출발선(중심선 s=0)을 앞으로 지날 때 스텝 안에서 선형 보간한 시각을 쓴다."""
    started_at: Optional[float] = None     # 지금 랩이 시작된 시뮬 시각(첫 통과 전에는 None = 출발 전 구간)
    laps: List[float] = field(default_factory=list)
    best: Optional[float] = None
    contacts_this_lap: int = 0
    clean_streak: int = 0                  # 무충돌 연속 바퀴(예선 채점의 둘째 항목)
    best_streak: int = 0

    def on_cross(self, t_cross: float) -> Optional[dict]:
        ev = None
        if self.started_at is not None:
            lt = t_cross - self.started_at
            self.laps.append(lt)
            clean = self.contacts_this_lap == 0
            self.clean_streak = self.clean_streak + 1 if clean else 0
            self.best_streak = max(self.best_streak, self.clean_streak)
            prev_best = self.best
            if self.best is None or lt < self.best:
                self.best = lt
            ev = {"lap": len(self.laps), "time": lt, "clean": clean, "best": self.best,
                  "is_best": prev_best is None or lt < prev_best, "clean_streak": self.clean_streak}
        self.started_at = t_cross
        self.contacts_this_lap = 0
        return ev


class SimSession:
    def __init__(self, track: str = "real/korea26", n_cars: int = 1, seed: int = 0,
                 ftg_max_speed: float = 6.0, compile_cpu: bool = False, threads: int = 1):
        from ..learn.baselines.reactive import FollowTheGap   # numpy만 쓰는 기준선(학습 코드 아님)

        torch.set_num_threads(max(1, int(threads)))
        t0 = time.perf_counter()
        self.track_spec = track
        self.track = maps.load(track)
        cfg = Config()
        cfg.sim.device = "cpu"
        cfg.sim.compile = False
        cfg.sim.terminate_on_collision = False
        cfg.sim.seed = int(seed)
        cfg.rand.enabled = False
        self.cfg = cfg
        self.n = int(n_cars)
        self.sim = Simulator(self.track, cfg, num_envs=self.n, device="cpu", race_size=self.n)
        if compile_cpu:   # 스파이크 측정: 물리 서브스텝 루프만 inductor(C++)로. 첫 컴파일은 수십 초
            self.sim._roll = torch.compile(self.sim._roll_physics, dynamic=False)
        self.drivers = []
        for _ in range(self.n):
            d = FollowTheGap(max_speed=ftg_max_speed)
            d.bind_scanner(n_beams=cfg.lidar.n_beams, fov=cfg.lidar.fov, range_max=cfg.lidar.range_max)
            self.drivers.append(d)
        self.control_dt = 1.0 / cfg.sim.control_rate
        self.gen = 0
        self.focus = 0
        self.build_s = time.perf_counter() - t0
        self.reset()

    # ------------------------------------------------------------------ 출발 자세
    def grid_poses(self) -> torch.Tensor:
        """출발선 1.0 m 뒤부터 차마다 0.9 m 간격, 진행 방향으로 정렬(중심선 순서가 진행 방향)."""
        cl = np.asarray(self.track.centerline, np.float64)
        seg = np.linalg.norm(np.roll(cl, -1, 0) - cl, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        L = cum[-1]
        poses = []
        for i in range(self.n):
            s = (L - 1.0 - 0.9 * i) % L
            j = int(np.searchsorted(cum, s, side="right") - 1) % len(cl)
            a, b = cl[j], cl[(j + 1) % len(cl)]
            f = (s - cum[j]) / max(seg[j], 1e-9)
            p = a + f * (b - a)
            lat = 0.25 * (1 if i % 2 else -1) if self.n > 1 else 0.0
            yaw = math.atan2(b[1] - a[1], b[0] - a[0])
            p = p + lat * np.array([-math.sin(yaw), math.cos(yaw)])
            poses.append([p[0], p[1], yaw])
        return torch.tensor(poses, dtype=torch.float32)

    def reset(self) -> None:
        self.gen += 1
        self.sim.reset(poses=self.grid_poses())
        self.clocks = [LapClock() for _ in range(self.n)]
        self.cmd = np.zeros((self.n, 2), np.float32)
        self.contacts = np.zeros(self.n, np.int64)
        self.contact_now = np.zeros(self.n, bool)
        self.t0 = float(self.sim.t)
        self.last = self.sim.step(torch.zeros(self.n, 2))   # 센서 한 장(정지 상태)
        self.events: List[dict] = []

    # ------------------------------------------------------------------ 한 제어 주기
    def step(self) -> None:
        r = self.last
        rmax = self.cfg.lidar.range_max
        scan = np.nan_to_num(r.scan.numpy(), nan=rmax, posinf=rmax, neginf=0.0).clip(0.0, rmax)
        v = r.state[:, 3].numpy()
        for i, d in enumerate(self.drivers):
            self.cmd[i] = d.command(d.adapt(scan[i:i + 1]), v[i:i + 1])[0]
        lap_prev = r.lap.clone()
        self.last = r = self.sim.step(torch.as_tensor(self.cmd))
        on = r.contact_onsets.any(1).numpy() if r.contact_onsets is not None else r.collision.numpy()
        self.contacts += on.astype(np.int64)
        self.contact_now = r.collision.numpy().astype(bool)
        for i in range(self.n):
            if on[i]:
                self.clocks[i].contacts_this_lap += 1
                self.events.append({"type": "contact", "car": i, "t": float(r.t)})
            if int(r.lap[i]) > int(lap_prev[i]):
                prog = float(r.progress[i])
                frac = float(r.s[i]) / prog if prog > 1e-6 else 0.0
                t_cross = float(r.t) - min(max(frac, 0.0), 1.0) * self.control_dt
                ev = self.clocks[i].on_cross(t_cross)
                if ev is not None:
                    ev.update(type="lap", car=i, t=float(r.t))
                    self.events.append(ev)

    # ------------------------------------------------------------------ 내보내기
    @property
    def t(self) -> float:
        return float(self.last.t) - self.t0

    def cars(self) -> List[dict]:
        r = self.last
        st = r.state.numpy()
        att = r.attitude.numpy()
        out = []
        for i in range(self.n):
            c = self.clocks[i]
            out.append({
                "id": i, "x": float(st[i, 0]), "y": float(st[i, 1]), "yaw": float(st[i, 2]),
                "v": float(st[i, 3]), "vy": float(st[i, 4]), "wz": float(st[i, 5]), "steer": float(st[i, 6]),
                "roll": float(att[i, 0]), "pitch": float(att[i, 1]),
                "cmd_steer": float(self.cmd[i, 0]), "cmd_v": float(self.cmd[i, 1]),
                "lap": len(c.laps), "s": float(r.s[i]),
                "lap_t": (float(r.t) - c.started_at) if c.started_at is not None else None,
                "last_lap": c.laps[-1] if c.laps else None, "best_lap": c.best,
                "clean_streak": c.clean_streak, "best_streak": c.best_streak,
                "contacts": int(self.contacts[i]), "contact": bool(self.contact_now[i]),
                "driver": "ftg",
            })
        return out

    def scans(self, which: Optional[List[int]] = None) -> List[dict]:
        r = self.last
        lp = self.cfg.lidar
        rng = r.scan.numpy()
        hit = r.scan_type.numpy()
        st = r.state.numpy()
        out = []
        for i in (range(self.n) if which is None else which):
            x, y, yaw = (float(v) for v in st[i, :3])
            sx = x + lp.mount_x * math.cos(yaw) - lp.mount_y * math.sin(yaw)
            sy = y + lp.mount_x * math.sin(yaw) + lp.mount_y * math.cos(yaw)
            rr = np.where(np.isfinite(rng[i]), rng[i], 0.0).astype(np.float32)   # 0 = 반환 없음
            out.append({"car": i, "pose": [sx, sy, yaw], "ranges": rr, "hit": hit[i].astype(np.uint8)})
        return out

    def track_message(self) -> dict:
        from ..viewer.geometry import build_track_geometry   # torch 없는 배열 생성(콘솔 워커와 같은 함수)
        g = build_track_geometry(self.track)

        def mesh(t):
            if t is None:
                return None
            pos, nrm, col, idx = t
            return {"pos": pos, "nrm": nrm, "idx": idx}

        cl = np.asarray(self.track.centerline, np.float32)
        seg = np.linalg.norm(np.roll(cl, -1, 0) - cl, axis=1)
        lp, vp = self.cfg.lidar, self.cfg.vehicle
        a, b = cl[0], cl[1]
        return {
            "k": "track", "v": SCHEMA_VERSION, "gen": self.gen, "name": self.track.name, "spec": self.track_spec,
            "length_m": float(seg.sum()), "duct_height": float(g["duct_height"]),
            "meshes": {"ducts": mesh(g["ducts"]), "walls": mesh(g["walls"]), "floor": mesh(g["floor"])},
            "centerline": cl, "bounds": list(g["content_bounds"] or g["bounds"]),
            "start": {"x": float(a[0]), "y": float(a[1]), "yaw": float(math.atan2(b[1] - a[1], b[0] - a[0]))},
            "presentation": g["presentation"],
            "lidar": {"n": int(lp.n_beams), "fov": float(lp.fov), "angle_min": -float(lp.fov) / 2.0,
                      "range_max": float(lp.range_max), "mount": [float(lp.mount_x), float(lp.mount_y), float(lp.mount_z)],
                      "rate": float(lp.rate)},
            "car": {"length": float(vp.length), "width": float(vp.width), "wheelbase": float(vp.lf + vp.lr)},
            "control_rate": float(self.cfg.sim.control_rate),
        }
