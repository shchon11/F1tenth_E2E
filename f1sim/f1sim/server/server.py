"""`python -m f1sim.server` : TENTH 시뮬 서버.

M1(기본): 레이스 화면 서버. 앱이 자료 목록을 받고(catalog), 구성(spec)으로 예선 세션을 만들고(session.start),
달리고(cmd start/pause/resume/reset/speed/end), 끝나면 대회 채점 결과(result)를 받는다. 채점은 시뮬 시간 기준.
  python -m f1sim.server --device cuda --port 47800
  python -m f1sim.server --device cpu --spec '{"driver": "ftg"}' --autostart      # 앱 없이 한 세션(시험용)

스파이크(--spike): 2026-11-28 기술 검증 그대로(Korea 2026 + Follow-the-Gap, 세션 구성 없음).

준비가 끝나면 표준 출력에 `TENTH_SERVER_READY {json}` 한 줄을 쓴다(실행기가 이 줄을 기다린다).
연산 장치: `--device cuda`는 CUDA_VISIBLE_DEVICES로 고른 카드(실행기는 5060 = 1)에서 CUDA 그래프로 돈다(실시간 1.0배 안팎).
`--device cpu`는 실시간의 약 0.4배(11-28 측정). 배속은 상태 프레임의 `rt`로 앱에 정직하게 보낸다.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback

import numpy as np

from . import protocol as P
from . import transports, wire

REPLAYS = os.path.join(os.environ.get("TENTH_DATA", "/data/tenth"), "replays")


class Recorder:
    """리플레이 기록: 서버가 보낸 메시지를 u32 길이 접두 틀 그대로 이어 쓴다. 첫 메시지는 hello."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.f = open(path, "wb")

    def write(self, payload: bytes) -> None:
        self.f.write(wire.frame(payload))

    def close(self) -> None:
        self.f.close()


class RaceServer:
    """M1 레이스 서버: 세션 하나를 들고, 앱 명령을 처리하고, 실시간(× 배속)에 맞춰 진행한다."""

    def __init__(self, tr, device: str, record_dir: str = REPLAYS, record: bool = True):
        from .library import Library
        self.tr = tr
        self.device = device
        self.lib = Library()
        self.catalog = self.lib.catalog()
        self.sess = None
        self.status = "idle"
        self.gen = 0
        self.seq = 0
        self.speed = 1.0
        self.rt = 0.0
        self.rec = None
        self.record_dir = record_dir
        self.record = record
        self.preview_cache = {}
        self.last_result = None
        self.hello = wire.packb(P.hello("tenth-sim", 40.0, "tcp", pid=os.getpid(), device=device, m1=True))
        self.last_frame_wall = 0.0
        self.anchor = None            # (wall, sim) 배속 기준점
        self.win = (time.perf_counter(), 0.0)
        self.step_ms = []

    # ------------------------------------------------------------------ 보내기
    def send(self, msg: dict, cid=None, reliable: bool = True) -> None:
        b = wire.packb(msg)
        if reliable:
            self.tr.send_reliable(b, cid)
        else:
            self.tr.send_state(b)
        if self.rec is not None and cid is None:
            self.rec.write(b)

    def stage(self, name: str, note: str) -> None:
        self.send(P.msg(P.K_STAGE, gen=self.gen, stage=name, note=note))
        self.pump(only_ping=True)

    def error(self, text: str, cid=None) -> None:
        print("TENTH_ERROR " + text, flush=True)
        self.send(P.msg(P.K_ERROR, note=text), cid)

    # ------------------------------------------------------------------ 요청
    def preview(self, m: dict) -> dict:
        """준비·메인 화면의 장면: 코스 기하(관·바닥·윤곽) + 상자 배치 + 출발 자세. 세션을 만들지 않는다(1초 안)."""
        import math as _m
        from . import course_geometry as CG
        from . import layouts as LY
        from .quali import place_gate_s
        from .. import maps
        course = self.lib.get("course", m.get("course") or self.lib.courses[0]["id"])
        key = course["id"]
        if key not in self.preview_cache:
            tr = maps.load(maps.smooth_name(course["track"]))
            g = CG.build(tr, duct_diameter=float((course.get("duct_diameter") or {}).get("v", 0.30)), step=0.04)
            self.preview_cache[key] = (tr, g)
        tr, g = self.preview_cache[key]
        b = dict(m.get("boxes") or {})
        try:
            lay = LY.resolve(course, b, tr) if int(b.get("count", 0)) > 0 else {"pieces": [], "label": "상자 없음"}
            pieces, note = lay["pieces"], lay.get("label", "")
        except ValueError as exc:
            pieces, note = [], str(exc)
        box_s = LY.centerline_s(tr, pieces) if pieces else []
        s0 = None
        for behind, ahead in ((2.5, 3.5), (1.5, 2.0), (1.0, 1.0)):
            s0 = place_gate_s(tr, 0.0, box_s, behind, ahead, 60.0)
            if s0 is not None:
                break
        cl = np.asarray(tr.centerline, float)
        seg = np.linalg.norm(np.diff(np.vstack([cl, cl[:1]]), axis=0), axis=1)
        sc = np.concatenate([[0.0], np.cumsum(seg)])
        sv = ((s0 or 0.0) - 1.0) % sc[-1]
        j = int(np.searchsorted(sc, sv, side="right") - 1) % len(cl)
        a_, b_ = cl[j], cl[(j + 1) % len(cl)]
        p_ = a_ + (sv - sc[j]) / max(seg[j], 1e-9) * (b_ - a_)
        start = {"x": float(p_[0]), "y": float(p_[1]), "yaw": float(_m.atan2(b_[1] - a_[1], b_[0] - a_[0]))}
        return P.msg(P.K_PREVIEW, course=key, outline=g["outline"], bounds=g["content_bounds"], boxes=LY.describe(pieces),
                     layout=note, length_m=g["length_m"], unchecked=bool(b.get("source") == "random"),
                     hoses=g["hoses"], walls_poly=g["walls"], lane=g["lane"], duct_height=float(tr.duct_height),
                     start=start, sectors=[])

    def start_session(self, spec: dict, autostart: bool) -> None:
        from .quali import QualiSession, SpecError
        if self.sess is not None:
            self.sess.close()
            self.sess = None
        self.gen += 1
        self.status = "building"
        self.send_state(force=True)
        t0 = time.perf_counter()
        try:
            self.sess = QualiSession(self.lib, spec, self.device, stage=self.stage)
        except (SpecError, ValueError) as exc:
            self.status = "idle"
            self.error(str(exc))
            return
        except Exception as exc:     # 세션을 못 만들면 사실 한 문장 + 서버 로그에 자세히
            traceback.print_exc()
            self.status = "idle"
            self.error(f"세션을 만들지 못했습니다: {type(exc).__name__}: {exc}")
            return
        build_s = time.perf_counter() - t0
        if self.record:
            if self.rec is not None:
                self.rec.close()
            self.rec = Recorder(os.path.join(self.record_dir, self.sess.id + ".tenthrec"))
            self.rec.write(self.hello)
        sm = self.sess.session_message()
        sm["build_s"] = round(build_s, 2)
        self.send(P.msg(P.K_SESSION, gen=self.gen, **sm))
        self.send(P.msg(P.K_TRACK, gen=self.gen, **self.sess.track_message()))
        print("TENTH_SESSION " + json.dumps({"id": self.sess.id, "build_s": round(build_s, 2), "facts": sm["facts"],
                                              "physics_check": {k: v for k, v in sm["physics_check"].items() if k != "skipped"}},
                                             ensure_ascii=False), flush=True)
        self.status = "ready"
        self.last_result = None
        if autostart:
            self.run()
        self.send_state(force=True)

    def run(self) -> None:
        if self.sess is None or self.sess.done:
            return
        self.status = "running"
        self.anchor = (time.perf_counter(), self.sess.t)
        self.win = (time.perf_counter(), self.sess.t)

    def finish(self) -> None:
        s = self.sess
        if s is None:
            return
        self.status = "finished"
        measured = {"rt_mean": round(s.t / max(sum(s.wall_steps), 1e-9), 3) if s.wall_steps else None,
                    "step_ms_median": round(1000 * float(np.median(s.wall_steps)), 1) if s.wall_steps else None,
                    "step_ms_mean": round(1000 * float(np.mean(s.wall_steps)), 1) if s.wall_steps else None,
                    "device": self.device, "speed_setting": self.speed}
        res = s.result(measured)
        try:
            paths = s.save(res)
        except OSError as exc:
            paths = {"error": str(exc)}
        res["files"] = paths
        try:
            if "json" in paths:
                with open(paths["json"], "w", encoding="utf-8") as f:
                    from .quali import _json_default
                    json.dump(res, f, ensure_ascii=False, indent=1, default=_json_default)
        except OSError:
            pass
        if self.rec is not None:
            res["files"]["replay"] = self.rec.path
        self.last_result = res
        self.send(P.msg(P.K_RESULT, gen=self.gen, **res))
        print("TENTH_RESULT " + json.dumps({"id": res["id"], "score": res["score"], "n_laps": res["n_laps"],
                                             "contacts": res["contacts_counted"], "files": res["files"],
                                             "measured": measured}, ensure_ascii=False), flush=True)
        if self.rec is not None:
            self.rec.close()
            self.rec = None

    def handle(self, cid, m: dict) -> None:
        k = m["k"]
        if k == P.K_HELLO:
            return
        if k == P.K_REQ:
            if m["op"] == "catalog":
                self.send(P.msg(P.K_CATALOG, **self.catalog, device=self.device), cid)
            elif m["op"] == "preview":
                try:
                    self.send(self.preview(m), cid)
                except Exception as exc:
                    traceback.print_exc()
                    self.error(f"미리 보기를 만들지 못했습니다: {exc}", cid)
            return
        if k == P.K_SESSION:
            self.send(wire_ack(m, "session"), cid)
            self.start_session(dict(m.get("spec") or {}), bool(m.get("autostart", False)))
            return
        if k == P.K_CMD:
            op = m["op"]
            ok = True
            if op in ("start", "resume"):
                if self.status in ("ready", "paused"):
                    self.run()
                else:
                    ok = False
            elif op == "pause":
                if self.status == "running":
                    self.status = "paused"
            elif op == "reset":
                if self.sess is not None:
                    self.gen += 1
                    self.sess.restart()
                    self.status = "ready"
                    self.last_result = None
                    if self.rec is None and self.record:
                        self.rec = Recorder(os.path.join(self.record_dir, self.sess.id + f"-g{self.gen}.tenthrec"))
                        self.rec.write(self.hello)
            elif op == "speed":
                self.speed = max(0.0, float(m.get("x", 1.0)))
                if self.status == "running":
                    self.anchor = (time.perf_counter(), self.sess.t)
            elif op in ("end", "stop"):
                if self.sess is not None and self.status in ("running", "paused", "ready"):
                    self.sess.done = True
                    self.finish()
            self.send(P.ack(m.get("seq", 0), op, ok), cid)
            self.send_state(force=True)

    def pump(self, only_ping: bool = False) -> None:
        for cid in self.tr.joined():
            self.tr.send_reliable(self.hello, cid)
            self.send(P.msg(P.K_CATALOG, **self.catalog, device=self.device), cid)
            if self.sess is not None and not only_ping:
                sm = self.sess.session_message()
                self.send(P.msg(P.K_SESSION, gen=self.gen, **sm), cid)
                self.send(P.msg(P.K_TRACK, gen=self.gen, **self.sess.track_message()), cid)
                if self.last_result is not None:
                    self.send(P.msg(P.K_RESULT, gen=self.gen, **self.last_result), cid)
        if only_ping:
            return
        for cid, m in self.tr.poll():
            err = P.check_client(m)
            if err:
                self.error(err, cid)
                continue
            try:
                self.handle(cid, m)
            except Exception as exc:
                traceback.print_exc()
                self.error(f"명령을 처리하지 못했습니다: {type(exc).__name__}: {exc}", cid)

    # ------------------------------------------------------------------ 상태
    def send_state(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_frame_wall < 1.0 / 40.0:
            return
        self.last_frame_wall = now
        self.seq += 1
        s = self.sess
        if s is None:
            self.send(P.state_frame(gen=self.gen, seq=self.seq, t=0.0, rt=0.0, status=self.status, cars=[], scans=[]),
                      reliable=False)
            return
        snap = s.engine.snapshot()
        plan = s.plan_message(snap)
        fr = P.state_frame(gen=self.gen, seq=self.seq, t=s.t, rt=self.rt if self.status == "running" else 0.0,
                           status=self.status, cars=[s.car_state(snap)], scans=[s.scan_message(snap)])
        fr.update(t_left=s.t_left(), duration_s=s.r["duration_s"], remove_at_s=s.r["remove_at_s"],
                  boxes_on=bool(s.engine.boxes_on), boxes=s.engine.box_poses(), speed=self.speed,
                  plan=plan if plan is not None else None, device=self.device)
        self.send(fr, reliable=False)

    def tick(self) -> None:
        """진행 중이면 배속에 맞춰 몇 주기를 돌리고, 상태 프레임을 보낸다."""
        s = self.sess
        if self.status != "running" or s is None:
            self.send_state()
            return
        now = time.perf_counter()
        w0, s0 = self.anchor
        target = s.t + 0.025 if self.speed <= 0 else s0 + (now - w0) * self.speed
        budget_end = now + 0.030
        n = 0
        while s.t < target - 1e-9 and not s.done:
            a = time.perf_counter()
            for ev in s.step():
                ev = dict(ev)
                kind = ev.pop("type")
                self.send(P.event(kind, gen=self.gen, **ev))
            self.step_ms.append((time.perf_counter() - a) * 1e3)
            n += 1
            if time.perf_counter() > budget_end:
                break
        if self.speed > 0 and s.t < target - 0.25:          # 따라잡지 않는다(폭주 방지): 지금부터 다시 센다
            self.anchor = (time.perf_counter(), s.t)
        if now - self.win[0] >= 1.0:
            self.rt = (s.t - self.win[1]) / (now - self.win[0])
            self.win = (now, s.t)
        self.send_state(force=n > 0)
        if s.done:
            self.finish()


def wire_ack(m: dict, op: str) -> dict:
    return P.ack(m.get("seq", 0), op, True)


def main_m1(a) -> int:
    tr = transports.make("tcp", a.host, a.port)
    tr.start()
    srv = RaceServer(tr, a.device, record=not a.no_record)
    gpu = ""
    if a.device.startswith("cuda"):
        import torch
        gpu = torch.cuda.get_device_name(0)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    print("TENTH_SERVER_READY " + json.dumps({"mode": "m1", "transport": "tcp", "host": a.host, "port": a.port,
                                              "pid": os.getpid(), "device": a.device, "gpu": gpu,
                                              "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                                              "courses": len(srv.catalog["courses"]), "drivers": len(srv.catalog["drivers"])},
                                             ensure_ascii=False), flush=True)
    if a.spec:
        srv.start_session(json.loads(a.spec), a.autostart)
    t_end = time.perf_counter() + a.duration if a.duration > 0 else None
    t_stats = time.perf_counter()
    while not stop["flag"]:
        if t_end is not None and time.perf_counter() >= t_end:
            break
        srv.pump()
        try:
            srv.tick()
        except Exception as exc:          # 한 주기의 오류가 서버를 끝내지 않게: 세션을 멈추고 알린다
            traceback.print_exc()
            srv.status = "paused"
            srv.error(f"진행 중 오류로 멈췄습니다: {type(exc).__name__}: {exc}")
        if a.stats_every > 0 and time.perf_counter() - t_stats >= a.stats_every and srv.sess is not None:
            t_stats = time.perf_counter()
            sm = sorted(srv.step_ms) or [0.0]
            print("TENTH_STATS " + json.dumps({"t_sim": round(srv.sess.t, 2), "rt": round(srv.rt, 3), "status": srv.status,
                                               "speed": srv.speed, "step_ms_med": round(sm[len(sm) // 2], 2),
                                               "step_ms_mean": round(float(np.mean(sm)), 2),
                                               "clients": len(tr.clients), "dropped": tr.dropped()}), flush=True)
            srv.step_ms.clear()
        if srv.status != "running":
            time.sleep(0.01)
        else:
            time.sleep(0.0005)
        if a.exit_after_result and srv.status == "finished":
            time.sleep(1.0)
            break
    if srv.sess is not None:
        srv.sess.close()
    tr.close()
    print("TENTH_SERVER_BYE", flush=True)
    return 0


def main_spike(a) -> int:
    from .session import SimSession
    t_boot = time.perf_counter()
    sess = SimSession(a.track, n_cars=a.cars, ftg_max_speed=a.ftg_speed, compile_cpu=a.compile, threads=a.threads)
    if a.compile:
        sess.step(); sess.reset()
    t0 = time.perf_counter()
    track_payload = wire.packb(sess.track_message())
    geom_ms = (time.perf_counter() - t0) * 1e3
    tr = transports.make(a.transport, a.host, a.port)
    tr.start()
    rec = Recorder(a.record) if a.record else None
    hello = wire.packb(P.hello("tenth-sim", 1.0 / sess.control_dt, a.transport, track=a.track, cars=a.cars,
                               pid=os.getpid()))
    if rec:
        rec.write(hello); rec.write(track_payload)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    print("TENTH_SERVER_READY " + json.dumps({
        "mode": "spike", "transport": a.transport, "host": a.host, "port": a.port, "pid": os.getpid(),
        "boot_s": round(time.perf_counter() - t_boot, 2), "track_bytes": len(track_payload),
        "geometry_ms": round(geom_ms, 1)}), flush=True)
    status = "running" if a.autostart else "ready"
    seq = 0
    dt = sess.control_dt
    t_next = time.perf_counter()
    t_end = time.perf_counter() + a.duration if a.duration > 0 else None
    win_wall, win_sim, step_ms = time.perf_counter(), sess.t, []
    rt = 1.0
    t_stats = time.perf_counter()
    while not stop["flag"]:
        now = time.perf_counter()
        if t_end is not None and now >= t_end:
            break
        for cid in tr.joined():
            tr.send_reliable(hello, cid)
            tr.send_reliable(track_payload, cid)
        for cid, m in tr.poll():
            err = P.check_client(m)
            if err:
                tr.send_reliable(wire.packb(P.msg(P.K_ERROR, note=err)), cid)
                continue
            if m["k"] == P.K_CMD:
                op = m["op"]
                if op in ("start", "resume"):
                    status = "running"
                elif op == "pause":
                    status = "paused"
                elif op in ("stop", "end"):
                    status = "stopped"
                elif op == "reset":
                    sess.reset()
                    status = "ready"
                tr.send_reliable(wire.packb(P.ack(m.get("seq", 0), op)), cid)
        if status == "running":
            a0 = time.perf_counter()
            sess.step()
            step_ms.append((time.perf_counter() - a0) * 1e3)
            for ev in sess.events:
                p = wire.packb(P.event(ev.pop("type"), gen=sess.gen, **ev))
                tr.send_reliable(p)
                if rec:
                    rec.write(p)
            sess.events.clear()
        if now - win_wall >= 1.0:
            rt = (sess.t - win_sim) / (now - win_wall) if status == "running" else 0.0
            win_wall, win_sim = now, sess.t
        seq += 1
        frame = wire.packb(P.state_frame(gen=sess.gen, seq=seq, t=sess.t, rt=rt, status=status,
                                         cars=sess.cars(), scans=sess.scans([sess.focus])))
        tr.send_state(frame)
        if rec:
            rec.write(frame)
        if a.stats_every > 0 and now - t_stats >= a.stats_every:
            t_stats = now
            sm = sorted(step_ms) if step_ms else [0.0]
            print("TENTH_STATS " + json.dumps({
                "t_sim": round(sess.t, 2), "rt": round(rt, 3), "status": status,
                "step_ms_med": round(sm[len(sm) // 2], 2), "frame_bytes": len(frame),
                "frames_sent": tr.frames_sent, "dropped": tr.dropped(), "clients": len(tr.clients),
                "laps": [c["lap"] for c in sess.cars()], "best": [c["best_lap"] for c in sess.cars()]}), flush=True)
            step_ms.clear()
        t_next += dt
        sl = t_next - time.perf_counter()
        if sl > 0:
            time.sleep(sl)
        else:
            t_next = time.perf_counter()
    tr.close()
    if rec:
        rec.close()
    print("TENTH_SERVER_BYE", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m f1sim.server", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spike", action="store_true", help="2026-11-28 스파이크 서버(Korea 2026 + FTG)")
    ap.add_argument("--device", default="cuda", help="M1: cuda | cpu")
    ap.add_argument("--spec", default="", help="M1: 시작하자마자 만들 세션 구성(JSON)")
    ap.add_argument("--exit-after-result", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--track", default="real/korea26")
    ap.add_argument("--cars", type=int, default=1)
    ap.add_argument("--transport", default="tcp", choices=("tcp", "ws", "udp", "shm"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47800)
    ap.add_argument("--autostart", action="store_true", help="앱의 start 명령 없이 바로 달린다")
    ap.add_argument("--ftg-speed", type=float, default=6.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--duration", type=float, default=0.0, help="이 시간(초, 벽시계) 뒤 스스로 끝남. 0 = 무한")
    ap.add_argument("--record", default="")
    ap.add_argument("--stats-every", type=float, default=5.0)
    a = ap.parse_args(argv)
    if a.spike:
        return main_spike(a)
    import torch
    torch.set_num_threads(max(1, int(a.threads)))
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("TENTH_WARN CUDA를 쓸 수 없어 CPU로 돕니다", flush=True)
        a.device = "cpu"
    return main_m1(a)


if __name__ == "__main__":
    sys.exit(main())
