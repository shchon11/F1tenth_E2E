"""TENTH 세션 프로토콜 v1 (스파이크). 모든 메시지는 사전 하나이고 `k`(종류)와 `v`(스키마 버전)를 가진다.

서버 -> 앱
  hello  : 서버 이름, 스키마 버전, 제어 주기, 지원 명령
  track  : 코스 지오메트리(덕트 관·벽·바닥 메시 = 기존 viewer.geometry 배열 그대로), 중심선, 출발선, LiDAR·차 치수
  state  : 매 제어 주기(40 Hz) 한 장. 차마다 자세·속도·조향·랩 계시, 보는 차의 LiDAR 거리 배열. **버려도 되는** 흐름
  event  : 랩 완료·접촉 같은 사건. 버리지 않는다
  ack    : 명령 확인(seq 그대로)
  pong   : ping에 즉시 답(왕복 지연 측정)
앱 -> 서버
  hello  : 앱 이름, 스키마 버전
  cmd    : {op: start|stop|pause|resume|reset|end|speed, seq, x(배속, speed일 때: 0 = 가능한 한 빨리)}
  ping   : {t: 앱 시각}

M1 추가(레이스 화면: 준비 -> 주행 -> 결과)
  앱 -> 서버  req     : {op: catalog}                       -> catalog(코스·드라이버·규칙·물리 프로필·실행 방식)
                        {op: preview, course, boxes}       -> preview(미니맵 윤곽 + 상자 배치 미리 보기)
              session : {op: start, spec{...}, autostart}  -> stage(만드는 단계) ... session(표기·상자·규칙) + track + state
  서버 -> 앱  stage   : {stage, note}  세션을 만드는 동안(수 초)
              session : 세션 정보(사람용 이름, 상자, 치우는 시각, 채점 항목, 물리 프로필 적용 확인)
              result  : 세션이 끝나면 대회 채점 결과(바퀴 표, 접촉 목록, 저장 경로)
              error   : 문장 하나(한국어)

세대(gen): 리셋·새 세션마다 1 증가. 앱은 지난 gen의 state를 버린다(기존 콘솔 protocol.py와 같은 규칙).
좌표: 시뮬 세계 좌표(x, y [m], yaw [rad], z 위). Godot 쪽에서 (x, z, -y)로 바꾼다.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from . import SCHEMA_VERSION

K_HELLO, K_TRACK, K_STATE, K_EVENT, K_ACK, K_PONG, K_CMD, K_PING, K_ERROR = (
    "hello", "track", "state", "event", "ack", "pong", "cmd", "ping", "error")
K_REQ, K_SESSION, K_CATALOG, K_PREVIEW, K_STAGE, K_RESULT = ("req", "session", "catalog", "preview", "stage", "result")
OPS = ("start", "stop", "pause", "resume", "reset", "end", "speed")
REQ_OPS = ("catalog", "preview")
SESSION_OPS = ("start",)
STATUS = ("idle", "building", "ready", "running", "paused", "finished", "stopped")


def msg(message_kind: str, **fields: Any) -> Dict[str, Any]:
    d = {"k": message_kind, "v": SCHEMA_VERSION}
    d.update(fields)
    return d


def hello(server: str, control_rate: float, transport: str, **extra) -> Dict[str, Any]:
    return msg(K_HELLO, server=server, control_rate=float(control_rate), transport=transport,
               ops=list(OPS), ts=time.time(), **extra)


def ack(seq: int, op: str, ok: bool = True, note: str = "") -> Dict[str, Any]:
    return msg(K_ACK, seq=int(seq), op=op, ok=bool(ok), note=note)


def pong(t_client: float) -> Dict[str, Any]:
    return msg(K_PONG, t=t_client, ts=time.time())


def event(event_type: str, **fields: Any) -> Dict[str, Any]:
    return msg(K_EVENT, type=event_type, ts=time.time(), **fields)


def check_client(m: Any) -> Optional[str]:
    """앱이 보낸 메시지의 형식 오류를 문장으로(없으면 None)."""
    if not isinstance(m, dict):
        return "메시지가 사전이 아닙니다"
    if m.get("v") != SCHEMA_VERSION:
        return f"스키마 버전이 다릅니다: 앱 {m.get('v')!r}, 서버 {SCHEMA_VERSION}"
    k = m.get("k")
    if k not in (K_HELLO, K_CMD, K_PING, K_REQ, K_SESSION):
        return f"모르는 메시지 종류 {k!r}"
    if k == K_CMD and m.get("op") not in OPS:
        return f"모르는 명령 {m.get('op')!r} (가능: {', '.join(OPS)})"
    if k == K_REQ and m.get("op") not in REQ_OPS:
        return f"모르는 요청 {m.get('op')!r} (가능: {', '.join(REQ_OPS)})"
    if k == K_SESSION and m.get("op") not in SESSION_OPS:
        return f"모르는 세션 명령 {m.get('op')!r}"
    if k == K_SESSION and not isinstance(m.get("spec", {}), dict):
        return "세션 구성(spec)은 사전이어야 합니다"
    return None


def state_frame(*, gen: int, seq: int, t: float, rt: float, status: str, cars: List[dict],
                scans: List[dict]) -> Dict[str, Any]:
    return msg(K_STATE, gen=int(gen), seq=int(seq), t=float(t), ts=time.time(), rt=float(rt),
               status=status, cars=cars, scans=scans)
