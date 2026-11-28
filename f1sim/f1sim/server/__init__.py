"""TENTH 시뮬 서버: 기존 f1sim 물리 코어를 소켓 너머의 프런트엔드(Godot 앱)에 내보내는 다리.

2026-11-28 기술 검증 스파이크. 이 패키지는 기존 모듈을 고치지 않고 import만 한다.

- `wire`       : 언어 중립 직렬화(MessagePack 부분집합 + 길이 접두 틀). 큰 배열은 원시 little-endian 바이트.
- `protocol`   : 스키마 버전, 메시지 종류, 메시지 만들기.
- `session`    : Simulator 1대 + 드라이버(Follow-the-Gap) + 랩 계시. 실시간 보조 루프.
- `transports` : TCP / WebSocket / UDP / 공유 메모리(/dev/shm) 네 가지 전송. 같은 프레임 바이트를 보낸다.
- `server`     : `python -m f1sim.server` 진입점(세션 + 전송 하나).
- `bench`      : 합성 프레임(차 2대 × LiDAR 1081점 × 40 Hz)으로 전송 방식을 비교하는 서버.

설계 문서: docs/pm/sim-arch-2026-11-28/ARCHITECTURE.md
"""

SCHEMA_VERSION = 1
