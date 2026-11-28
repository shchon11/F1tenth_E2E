"""전송 네 가지. 모두 같은 프레임 바이트(MessagePack 본문)를 나른다.

공통 규칙
- `send_reliable(payload, client)`: hello·track·event·ack·pong. 버리지 않는다(TCP·WS). UDP·공유 메모리는 한계가 있다.
- `send_state(payload)`: 상태 흐름. 앱마다 **가장 최근 한 장만** 들고 있다가 보낸다(drop-oldest). 느린 앱이
  서버를 막지 않고, 쌓인 옛 프레임이 지연을 늘리지 않는다(기존 콘솔 protocol.LatestSlot과 같은 규칙).
- `poll()`: 받은 메시지 [(client_id, dict)]. `joined()`: 새로 붙은 앱 id(서버가 hello·track을 보낸다).

TCP  : u32 길이 접두 틀. TCP_NODELAY. 원격 그대로 됨.
WS   : websockets 동기 서버, 이진 메시지 1개 = 본문 1개. 압축 끔.
UDP  : 데이터그램 1개 = 본문 1개(64 KB 미만만). 큰 메시지(track)는 못 보냄 -> 실제로는 TCP 옆 채널이 필요.
SHM  : /dev/shm 파일 하나에 seqlock(홀수 = 쓰는 중) + 최신 본문. 같은 PC 한 방향. 명령 채널 없음.
"""
from __future__ import annotations

import collections
import mmap
import os
import queue
import socket
import struct
import threading
from typing import Dict, List, Optional, Tuple

from . import wire


class _Outbox:
    """앱 하나의 보낼 거리: 순서 보장 대기열(신뢰) + 최신 상태 한 칸."""

    def __init__(self):
        self.cv = threading.Condition()
        self.reliable: collections.deque = collections.deque()
        self.state: Optional[bytes] = None
        self.dropped = 0
        self.closed = False

    def put_reliable(self, b: bytes) -> None:
        with self.cv:
            self.reliable.append(b)
            self.cv.notify()

    def put_state(self, b: bytes) -> None:
        with self.cv:
            if self.state is not None:
                self.dropped += 1
            self.state = b
            self.cv.notify()

    def take(self, timeout: float = 0.5) -> List[bytes]:
        with self.cv:
            if not self.reliable and self.state is None and not self.closed:
                self.cv.wait(timeout)
            out = list(self.reliable)
            self.reliable.clear()
            if self.state is not None:
                out.append(self.state)
                self.state = None
            return out

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify()


class Transport:
    kind = "?"

    def __init__(self):
        self.inbox: "queue.Queue[Tuple[int, dict]]" = queue.Queue()
        self._joined: "queue.Queue[int]" = queue.Queue()
        self.clients: Dict[int, _Outbox] = {}
        self.lock = threading.Lock()
        self.next_id = 1
        self.bytes_sent = 0
        self.frames_sent = 0
        self.stop_ev = threading.Event()

    # 공통 ---------------------------------------------------------------------
    def _register(self) -> Tuple[int, _Outbox]:
        with self.lock:
            cid = self.next_id
            self.next_id += 1
            ob = _Outbox()
            self.clients[cid] = ob
        self._joined.put(cid)
        return cid, ob

    def _drop(self, cid: int) -> None:
        with self.lock:
            ob = self.clients.pop(cid, None)
        if ob:
            ob.close()

    def _deliver(self, cid: int, payload: bytes) -> None:
        try:
            m = wire.unpackb(payload)
        except Exception as e:   # 형식이 틀린 메시지는 오류 사건으로 넘긴다
            m = {"k": "_bad", "err": str(e)}
        if isinstance(m, dict) and m.get("k") == "ping":
            # 왕복 지연은 전송 계층에서 바로 답한다(시뮬 루프 주기에 묶이지 않게)
            from . import protocol as P
            self.send_reliable(wire.packb(P.pong(m.get("t", 0.0))), cid)
            return
        self.inbox.put((cid, m))

    def joined(self) -> List[int]:
        out = []
        while True:
            try:
                out.append(self._joined.get_nowait())
            except queue.Empty:
                return out

    def poll(self) -> List[Tuple[int, dict]]:
        out = []
        while True:
            try:
                out.append(self.inbox.get_nowait())
            except queue.Empty:
                return out

    def send_reliable(self, payload: bytes, client: Optional[int] = None) -> None:
        with self.lock:
            targets = list(self.clients.items()) if client is None else [(client, self.clients.get(client))]
        for _, ob in targets:
            if ob is not None:
                ob.put_reliable(payload)

    def send_state(self, payload: bytes) -> None:
        with self.lock:
            obs = list(self.clients.values())
        for ob in obs:
            ob.put_state(payload)

    def dropped(self) -> int:
        with self.lock:
            return sum(ob.dropped for ob in self.clients.values())

    def start(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self.stop_ev.set()
        with self.lock:
            for ob in self.clients.values():
                ob.close()


# ---------------------------------------------------------------------------- TCP
class TcpTransport(Transport):
    kind = "tcp"

    def __init__(self, host: str = "127.0.0.1", port: int = 47800):
        super().__init__()
        self.host, self.port = host, port

    def start(self) -> None:
        self.srv = socket.create_server((self.host, self.port), reuse_port=False)
        self.srv.settimeout(0.5)
        threading.Thread(target=self._accept, name="tcp-accept", daemon=True).start()

    def _accept(self) -> None:
        while not self.stop_ev.is_set():
            try:
                conn, _ = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            cid, ob = self._register()
            threading.Thread(target=self._reader, args=(cid, conn), daemon=True).start()
            threading.Thread(target=self._writer, args=(cid, conn, ob), daemon=True).start()

    def _reader(self, cid: int, conn: socket.socket) -> None:
        sp = wire.FrameSplitter()
        try:
            while not self.stop_ev.is_set():
                data = conn.recv(65536)
                if not data:
                    break
                for p in sp.feed(data):
                    self._deliver(cid, p)
        except OSError:
            pass
        finally:
            self._drop(cid)

    def _writer(self, cid: int, conn: socket.socket, ob: _Outbox) -> None:
        try:
            while not ob.closed and not self.stop_ev.is_set():
                for p in ob.take():
                    conn.sendall(wire.frame(p))
                    self.bytes_sent += len(p) + 4
                    self.frames_sent += 1
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass
            self._drop(cid)

    def close(self) -> None:
        super().close()
        try:
            self.srv.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------- WebSocket
class WsTransport(Transport):
    kind = "ws"

    def __init__(self, host: str = "127.0.0.1", port: int = 47801):
        super().__init__()
        self.host, self.port = host, port

    def start(self) -> None:
        from websockets.sync.server import serve
        self.srv = serve(self._handler, self.host, self.port, compression=None, max_size=2 ** 26)
        threading.Thread(target=self.srv.serve_forever, name="ws-serve", daemon=True).start()

    def _handler(self, conn) -> None:
        cid, ob = self._register()
        sock = getattr(conn, "socket", None)
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

        def writer():
            try:
                while not ob.closed and not self.stop_ev.is_set():
                    for p in ob.take():
                        conn.send(p)
                        self.bytes_sent += len(p)
                        self.frames_sent += 1
            except Exception:
                pass

        th = threading.Thread(target=writer, daemon=True)
        th.start()
        try:
            for m in conn:
                if isinstance(m, (bytes, bytearray)):
                    self._deliver(cid, bytes(m))
        except Exception:
            pass
        finally:
            self._drop(cid)

    def close(self) -> None:
        super().close()
        try:
            self.srv.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------- UDP
class UdpTransport(Transport):
    kind = "udp"
    MAX = 65000

    def __init__(self, host: str = "127.0.0.1", port: int = 47802):
        super().__init__()
        self.host, self.port = host, port
        self.addr_of: Dict[int, tuple] = {}
        self.id_of: Dict[tuple, int] = {}
        self.too_big = 0

    def start(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(0.5)
        threading.Thread(target=self._reader, name="udp-read", daemon=True).start()

    def _reader(self) -> None:
        while not self.stop_ev.is_set():
            try:
                data, addr = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            cid = self.id_of.get(addr)
            if cid is None:
                cid, ob = self._register()
                self.addr_of[cid] = addr
                self.id_of[addr] = cid
                threading.Thread(target=self._writer, args=(cid, ob), daemon=True).start()
            self._deliver(cid, data)

    def _writer(self, cid: int, ob: _Outbox) -> None:
        addr = self.addr_of[cid]
        while not ob.closed and not self.stop_ev.is_set():
            for p in ob.take():
                if len(p) > self.MAX:
                    self.too_big += 1     # track 같은 큰 메시지는 UDP로 못 보낸다
                    continue
                try:
                    self.sock.sendto(p, addr)
                    self.bytes_sent += len(p)
                    self.frames_sent += 1
                except OSError:
                    pass


# ---------------------------------------------------------------------------- 공유 메모리
class ShmTransport(Transport):
    """/dev/shm/<name>: [0:8] b'TENTHSHM' [8:16] u64 seq (홀수=쓰는 중) [16:20] u32 len [20:24] 0 [24:] 본문.
    한 방향(서버 -> 앱). 앱이 매 프레임 seq를 두 번 읽어 찢긴 읽기를 버린다."""
    kind = "shm"
    HDR = 24

    def __init__(self, name: str = "tenth_bench", capacity: int = 1 << 20):
        super().__init__()
        self.path = f"/dev/shm/{name}"
        self.cap = capacity
        self.seq = 0

    def start(self) -> None:
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(fd, self.HDR + self.cap)
        self.mm = mmap.mmap(fd, self.HDR + self.cap)
        os.close(fd)
        self.mm[0:8] = b"TENTHSHM"
        struct.pack_into("<QI", self.mm, 8, 0, 0)
        self._register()   # 앱이 몇 개 붙었는지 모른다: 가상의 앱 하나

    def send_state(self, payload: bytes) -> None:
        n = len(payload)
        if n > self.cap:
            return
        self.seq += 1
        struct.pack_into("<Q", self.mm, 8, 2 * self.seq - 1)      # 홀수: 쓰는 중
        struct.pack_into("<I", self.mm, 16, n)
        self.mm[self.HDR:self.HDR + n] = payload
        struct.pack_into("<Q", self.mm, 8, 2 * self.seq)          # 짝수: 다 씀
        self.bytes_sent += n
        self.frames_sent += 1

    def send_reliable(self, payload: bytes, client: Optional[int] = None) -> None:
        pass   # 한 방향 최신 칸뿐: 신뢰 메시지 통로가 없다

    def close(self) -> None:
        super().close()
        try:
            self.mm.close()
            os.unlink(self.path)
        except Exception:
            pass


def make(kind: str, host: str, port: int) -> Transport:
    if kind == "tcp":
        return TcpTransport(host, port)
    if kind == "ws":
        return WsTransport(host, port)
    if kind == "udp":
        return UdpTransport(host, port)
    if kind == "shm":
        return ShmTransport(f"tenth_{port}")
    raise ValueError(f"모르는 전송 {kind!r} (tcp | ws | udp | shm)")
