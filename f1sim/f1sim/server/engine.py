"""시뮬 엔진 어댑터(M1, 설계서 §4.2-1a): 옛 콘솔의 실행기 `viewer/sim_worker.SimWorker`를 **고치지 않고 감싸** 세션 하나를
만든다. 학습된 드라이버(체크포인트) · 기준선(Follow-the-Gap) 모두 같은 길(같은 env, 같은 상자, 같은 접촉 회계)로 달린다.

감싸면서 이 모듈이 더하는 것(모두 인스턴스 속성 바꿔 끼우기 · import만, 기존 파일 수정 없음)
1. 물리 프로필: `learn.watch.viewer_config`를 감싸 프로필 값(`f1sim.profiles.loader.overrides`)을 Config에 얹는다.
   세션을 만든 뒤 시뮬레이터 매개변수(`sim.P`)를 다시 읽어 프로필 값과 같은지 확인한다(`applied_check`).
2. 출발: 출발선 1 m 뒤, 차로 가운데, 정지. `env.sim.sample_spawn`을 이 세션의 인스턴스에서만 바꿔 끼운다.
3. 상자: 학습 생성기 슬롯에 고정 배치를 쥐여 둔다(`prop_layout.LayoutHold`, 콘솔과 같은 방법). 치우는 시각이 되면
   `learn.state_rl.PropRemoval`과 같은 방식으로 슬롯을 비운다(빔·접촉·관측이 모두 그 텐서를 읽는다).
4. 실행 방식: 실차 방식 = MAP 추종기(`plan_follower=map`), 이상 실행 = 계획대로 정확히(`plan_follower=ideal`).

벽은 매끄러운 판이다: 실행기가 지도 이름에 `+smooth`를 붙인다(`maps.smooth_name`, 11-17). 이 모듈은 그 사실을
세션 정보(`facts.walls`)로 확인해 적는다.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

CONTACT_COLS = ("kind", "x", "y", "severity", "held_s", "held_kind", "held_x", "held_y", "held_severity",
                "onsets_wall", "onsets_prop", "onsets_car", "hard_wall", "hard_prop", "hard_car")


class _Ctl:
    """SimWorker가 콘솔에 보내던 제어 메시지(단계·기록)를 받아 서버 로그·앱 단계 표시로 넘긴다."""

    def __init__(self, sink: Callable[[dict], None]):
        self.sink = sink
        self.log: List[str] = []

    def send(self, m: dict) -> None:
        kind = m.get("kind")
        if kind == "log":
            self.log.append(str(m.get("text", "")))
        self.sink(m)

    def poll(self) -> bool:
        return False


class _Frames:
    def send(self, m) -> None:
        pass


def _apply_profile(cfg, prof: Optional[dict]):
    if not prof:
        return cfg
    from ..profiles.loader import overrides
    d, _ = overrides(prof)
    for g, kv in d.items():
        if g == "rand":
            cfg.rand.ranges.update(kv["ranges"])
            continue
        grp = getattr(cfg, g)
        for k, v in kv.items():
            setattr(grp, k, v)
    return cfg


class Engine:
    """세션 하나: 만들기, 한 제어 주기 진행, 상태 읽기, 상자 치우기, 처음부터."""

    def __init__(self, *, ckpt: str, track: str, device: str, exec_mode: str = "real", ego: Optional[dict] = None,
                 profile: Optional[str] = None, pieces: Optional[List[dict]] = None, start_behind_m: float = 1.0,
                 start_s: float = 0.0, stage: Optional[Callable[[str, str], None]] = None):
        import torch
        from ..learn import watch as _watch
        from ..viewer import sim_worker as W
        from ..viewer.console import protocol as CP
        self.torch = torch
        self.exec_mode = exec_mode
        self.pieces = list(pieces or [])
        self.start_behind_m = float(start_behind_m)
        self.stage_cb = stage or (lambda name, note: None)
        self.profile_name = profile
        self.prof = None
        if profile:
            from ..profiles.loader import load_profile
            self.prof = load_profile(profile)

        def sink(m):
            if m.get("kind") == "stage":
                self.stage_cb(str(m.get("stage", "")), str(m.get("note", "")))
        self.ctl = _Ctl(sink)
        self.worker = W.SimWorker(self.ctl, _Frames())
        self.worker.gen = 1
        ego = dict(ego or {})
        cfg = CP.SessionConfig(
            run=ckpt, map_name=track, device=device, randomize=False, collision_mode="soft",
            plan_follower=("ideal" if exec_mode == "ideal" else "map"), mu_mode="random",
            procedural=bool(self.pieces), layout_seed=(0 if self.pieces else None), episode_s=7200.0,
            ego_kind=ego.get("kind", "policy"), ego_options=dict(ego.get("options") or {}),
            ego_tuned=bool(ego.get("tuned", False)), max_render_cars=1)
        self.cfg = cfg
        orig = _watch.viewer_config
        prof = self.prof
        _watch.viewer_config = lambda compile_enabled, randomize=True: _apply_profile(orig(compile_enabled, randomize), prof)
        t0 = time.perf_counter()
        try:
            self.worker._build_hold(True)        # 이 프로세스에 다른 세션이 없음 -> CUDA 그래프 캡처 허용
            self.session = self.worker.build_session(cfg, 1)
        finally:
            self.worker._build_hold(False)
            _watch.viewer_config = orig
        self.build_s = time.perf_counter() - t0
        s = self.session
        self.env = env = s["env"]
        self.sim = sim = env.sim
        self.device = str(s["device"])
        self.fastpath = s.get("fastpath") is not None
        self.track = s["track"]
        # 출발 자세: 출발선(중심선 s = start_s) 1 m 뒤, 가운데, 정지
        L = float(sim.track.length[0])
        self.L = L
        e = env.ecfg
        e.spawn_lateral_std = 0.0
        e.spawn_yaw_std = 0.0
        e.spawn_speed_max = 0.0
        e.stagger_first_episode = False
        orig_spawn = sim.sample_spawn
        self.start_s = float(start_s) % L
        s_start = (self.start_s - self.start_behind_m) % L

        def spawn_at_start(n, lateral_std=0.0, yaw_std=0.0, s=None, **kw):
            kw.pop("lat", None)
            return orig_spawn(n, 0.0, 0.0, s=torch.full((n,), s_start, device=sim.device), lat=torch.zeros(n, device=sim.device), **kw)
        sim.sample_spawn = spawn_at_start
        self.hold = s.get("layout_hold")
        if self.pieces and self.hold is not None:
            self.hold.set_pieces([{"sid": int(p["sid"]), "x": float(p["x"]), "y": float(p["y"]), "yaw": float(p["yaw"])}
                                  for p in self.pieces])
        self.boxes_on = bool(self.pieces)
        self.reset()

    # ------------------------------------------------------------------ 확인
    def applied_check(self) -> dict:
        """프로필 값이 시뮬레이터에 실제로 들어갔는지: 프로필이 Config에 넣는 값마다 `sim.P`를 다시 읽어 비교."""
        P_ = self.sim.P
        out = {"profile": self.profile_name or "params.py 기본값", "checked": 0, "same": 0, "differs": [], "not_in_P": []}
        if not self.prof:
            for k in ("mu", "B_f", "B_r", "h", "cmd_delay", "servo_tau"):
                if k in P_:
                    out.setdefault("values", {})[k] = float(P_[k][0])
            return out
        from ..profiles.loader import overrides
        d, skipped = overrides(self.prof)
        for g, kv in d.items():
            if g == "rand":
                continue
            for k, v in kv.items():
                if k not in P_:
                    out["not_in_P"].append(f"{g}.{k}")
                    continue
                got = float(P_[k][0])
                out["checked"] += 1
                if isinstance(v, bool) or abs(got - float(v)) <= 1e-5 * max(1.0, abs(float(v))):
                    out["same"] += 1
                else:
                    out["differs"].append({"key": f"{g}.{k}", "profile": float(v), "sim": got})
        out["skipped"] = skipped
        return out

    def facts(self) -> dict:
        s = self.session
        name = getattr(self.track, "name", "?")
        res = float(self.track.resolution)
        return {"track": name, "resolution_m": res, "walls": "smooth" if res < 0.02 else "grid",
                "device": self.device, "cuda_graphs": self.fastpath, "build_s": round(self.build_s, 2),
                "mode": s.get("mode"), "kind": s.get("state_kind"), "ego": s.get("ego_kind"),
                "policy_idle": bool(s.get("policy_idle")), "exec": self.exec_mode,
                "info": s.get("info_line", ""), "log": list(self.ctl.log)[-12:]}

    # ------------------------------------------------------------------ 진행
    def reset(self) -> None:
        self.worker._reset_episode(self.session)
        self.boxes_on = bool(self.pieces)
        self.t0 = float(self.env.last_result.t)
        self.prev_counts = np.zeros(6, np.float64)
        self.last = self.read()

    def step(self) -> dict:
        self.worker._step_once(self.session)
        self.last = self.read()
        return self.last

    def read(self) -> dict:
        """한 번의 장치->호스트 복사로 채점·기록에 필요한 것(매 제어 주기)."""
        torch = self.torch
        env, sim = self.env, self.sim
        r = env.last_result
        cf = sim.contact_frame()[0]
        pack = torch.cat([r.state[0, :7].float(), r.lap[0:1].float(), r.s[0:1].float(), r.progress[0:1].float(),
                          cf.float(), env.last_cmd[0, :2].float(), sim.ax[0:1].float(), sim.ay[0:1].float()])
        v = pack.cpu().numpy().astype(np.float64)
        st = v[:7]
        c = v[10:10 + len(CONTACT_COLS)]
        k = 10 + len(CONTACT_COLS)
        counts = c[9:15]
        new = np.clip(counts - self.prev_counts, 0, None) if hasattr(self, "prev_counts") else np.zeros(6)
        self.prev_counts = counts.copy()
        return {"t": float(r.t) - self.t0 if hasattr(self, "t0") else 0.0,
                "x": st[0], "y": st[1], "yaw": st[2], "v": st[3], "vy": st[4], "wz": st[5], "steer": st[6],
                "lap_sim": int(round(v[7])), "s": v[8], "progress": v[9],
                "c_kind": int(round(c[0])), "c_x": c[1], "c_y": c[2],
                # 세기: 이번 주기의 가장 센 접촉과 머무는 기록(0.3 s) 중 큰 것 -- 벽 충격은 다음 주기로 넘겨 세는 경우가 있다
                "c_sev": max(c[3], c[8]),
                "counts": counts, "new": new,
                "cmd_steer": v[k], "cmd_v": v[k + 1], "ax": v[k + 2], "ay": v[k + 3]}

    def remove_boxes(self) -> None:
        po = getattr(self.env, "procedural", None)
        if po is None:
            self.boxes_on = False
            return
        with self.torch.no_grad():
            po.p_sid[:] = -1
            po.p_zhi[:] = po.p_zlo - 1.0
            po.p_mass[:] = 0.0
            po.p_vel[:] = 0.0
        self.boxes_on = False

    def box_poses(self) -> List[list]:
        """지금 상자 자세(밀리면 움직인다): [[i, x, y, yaw], ...] -- 살아 있는 슬롯만."""
        po = getattr(self.env, "procedural", None)
        if po is None or not self.boxes_on:
            return []
        sid = po.p_sid[0].cpu().numpy()
        pose = po.p_poses[0].cpu().numpy()
        out = []
        for j in range(min(len(self.pieces), len(sid))):
            if int(sid[j]) >= 0:
                out.append([j, float(pose[j, 0]), float(pose[j, 1]), float(pose[j, 2])])
        return out

    def snapshot(self) -> dict:
        """앱에 그릴 것: LiDAR, 궤적. 실행기의 `_snapshot`(한 번 복사)을 그대로 쓴다."""
        fr = self.worker._snapshot(self.session)
        return fr

    def close(self) -> None:
        try:
            self.worker._release_session(self.session, joined=True)
        except Exception:
            pass
