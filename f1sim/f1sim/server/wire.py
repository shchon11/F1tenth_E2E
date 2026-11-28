"""언어 중립 직렬화: MessagePack 부분집합 + 길이 접두 틀(stream framing).

왜 직접 구현하나: 저장소 venv에 `msgpack` 패키지가 없고(2026-11-28 확인), 스파이크는 환경을 바꾸지 않는다.
`msgpack`이 설치돼 있으면 그것을 쓰고, 없으면 아래 순수 파이썬 구현을 쓴다. 둘의 출력은 같은 규격이다.

규약
- 스칼라·문자열·리스트·사전은 표준 MessagePack(big-endian)이다. 실수는 float64(0xcb)로 보낸다.
- 숫자 배열(numpy)은 사전 하나로 보낸다: {"$nd": "<f4|f8|i4|i2|u1|i1|u2|u4>", "shape": [...], "data": <bin>}.
  data는 **little-endian 원시 바이트**다. Godot은 PackedByteArray.to_float32_array()로 복사 한 번에 읽는다.
- 틀(frame): u32 little-endian 길이 + MessagePack 본문. TCP·UDS 같은 바이트 흐름에서만 쓴다.
  WebSocket·UDP·공유 메모리는 메시지 경계가 이미 있으므로 본문만 보낸다.
"""
from __future__ import annotations

import struct
from typing import Any

import numpy as np

try:  # 있으면 C 구현을 쓴다 (규격 동일)
    import msgpack as _msgpack  # type: ignore
except Exception:  # pragma: no cover - 이 PC에는 없다
    _msgpack = None

_DT = {"f4": np.float32, "f8": np.float64, "i4": np.int32, "i2": np.int16, "i1": np.int8,
       "u1": np.uint8, "u2": np.uint16, "u4": np.uint32}
_DT_NAME = {np.dtype(v).newbyteorder("<"): k for k, v in _DT.items()}


def nd(arr) -> dict:
    """numpy 배열 -> 배열 사전. 바이트 순서는 little-endian으로 맞춘다."""
    a = np.asarray(arr)
    dt = a.dtype.newbyteorder("<") if a.dtype.byteorder not in ("|",) else a.dtype
    name = _DT_NAME.get(np.dtype(dt).newbyteorder("<"))
    if name is None:
        a = a.astype(np.float32)
        name = "f4"
        dt = np.dtype("<f4")
    a = np.ascontiguousarray(a, dtype=dt)
    return {"$nd": name, "shape": list(a.shape), "data": a.tobytes()}


def from_nd(d: dict) -> np.ndarray:
    a = np.frombuffer(d["data"], dtype=np.dtype(_DT[d["$nd"]]).newbyteorder("<"))
    return a.reshape(d["shape"])


def _prep(o: Any) -> Any:
    """numpy를 표준 타입으로 바꾼다(배열은 배열 사전)."""
    if isinstance(o, np.ndarray):
        return nd(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {str(k): _prep(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_prep(v) for v in o]
    return o


# ------------------------------------------------------------------ 순수 파이썬 MessagePack (부분집합)
def _pack(o: Any, out: bytearray) -> None:
    if o is None:
        out.append(0xC0)
    elif o is True:
        out.append(0xC3)
    elif o is False:
        out.append(0xC2)
    elif isinstance(o, int):
        if 0 <= o < 128:
            out.append(o)
        elif -32 <= o < 0:
            out.append(o & 0xFF)
        elif 0 <= o < 2 ** 8:
            out += b"\xcc" + struct.pack(">B", o)
        elif 0 <= o < 2 ** 16:
            out += b"\xcd" + struct.pack(">H", o)
        elif 0 <= o < 2 ** 32:
            out += b"\xce" + struct.pack(">I", o)
        elif 0 <= o < 2 ** 64:
            out += b"\xcf" + struct.pack(">Q", o)
        elif -2 ** 7 <= o < 0:
            out += b"\xd0" + struct.pack(">b", o)
        elif -2 ** 15 <= o < 0:
            out += b"\xd1" + struct.pack(">h", o)
        elif -2 ** 31 <= o < 0:
            out += b"\xd2" + struct.pack(">i", o)
        else:
            out += b"\xd3" + struct.pack(">q", o)
    elif isinstance(o, float):
        out += b"\xcb" + struct.pack(">d", o)
    elif isinstance(o, str):
        b = o.encode("utf-8")
        n = len(b)
        if n < 32:
            out.append(0xA0 | n)
        elif n < 2 ** 8:
            out += b"\xd9" + struct.pack(">B", n)
        elif n < 2 ** 16:
            out += b"\xda" + struct.pack(">H", n)
        else:
            out += b"\xdb" + struct.pack(">I", n)
        out += b
    elif isinstance(o, (bytes, bytearray, memoryview)):
        b = bytes(o)
        n = len(b)
        if n < 2 ** 8:
            out += b"\xc4" + struct.pack(">B", n)
        elif n < 2 ** 16:
            out += b"\xc5" + struct.pack(">H", n)
        else:
            out += b"\xc6" + struct.pack(">I", n)
        out += b
    elif isinstance(o, (list, tuple)):
        n = len(o)
        if n < 16:
            out.append(0x90 | n)
        elif n < 2 ** 16:
            out += b"\xdc" + struct.pack(">H", n)
        else:
            out += b"\xdd" + struct.pack(">I", n)
        for v in o:
            _pack(v, out)
    elif isinstance(o, dict):
        n = len(o)
        if n < 16:
            out.append(0x80 | n)
        elif n < 2 ** 16:
            out += b"\xde" + struct.pack(">H", n)
        else:
            out += b"\xdf" + struct.pack(">I", n)
        for k, v in o.items():
            _pack(k, out)
            _pack(v, out)
    else:
        raise TypeError(f"직렬화할 수 없는 타입: {type(o).__name__}")


class _Reader:
    __slots__ = ("b", "i")

    def __init__(self, b: bytes):
        self.b = memoryview(b)
        self.i = 0

    def take(self, n: int) -> bytes:
        j = self.i + n
        if j > len(self.b):
            raise ValueError("MessagePack 본문이 잘렸습니다")
        s = self.b[self.i:j]
        self.i = j
        return bytes(s)

    def obj(self) -> Any:
        t = self.take(1)[0]
        if t < 0x80:
            return t
        if t >= 0xE0:
            return t - 0x100
        if 0x80 <= t <= 0x8F:
            return self.map(t & 0x0F)
        if 0x90 <= t <= 0x9F:
            return [self.obj() for _ in range(t & 0x0F)]
        if 0xA0 <= t <= 0xBF:
            return self.take(t & 0x1F).decode("utf-8")
        u = struct.unpack
        if t == 0xC0:
            return None
        if t == 0xC2:
            return False
        if t == 0xC3:
            return True
        if t == 0xC4:
            return self.take(u(">B", self.take(1))[0])
        if t == 0xC5:
            return self.take(u(">H", self.take(2))[0])
        if t == 0xC6:
            return self.take(u(">I", self.take(4))[0])
        if t == 0xCA:
            return u(">f", self.take(4))[0]
        if t == 0xCB:
            return u(">d", self.take(8))[0]
        fmt = {0xCC: ">B", 0xCD: ">H", 0xCE: ">I", 0xCF: ">Q", 0xD0: ">b", 0xD1: ">h", 0xD2: ">i", 0xD3: ">q"}.get(t)
        if fmt:
            return u(fmt, self.take(struct.calcsize(fmt)))[0]
        if t == 0xD9:
            return self.take(u(">B", self.take(1))[0]).decode("utf-8")
        if t == 0xDA:
            return self.take(u(">H", self.take(2))[0]).decode("utf-8")
        if t == 0xDB:
            return self.take(u(">I", self.take(4))[0]).decode("utf-8")
        if t == 0xDC:
            return [self.obj() for _ in range(u(">H", self.take(2))[0])]
        if t == 0xDD:
            return [self.obj() for _ in range(u(">I", self.take(4))[0])]
        if t == 0xDE:
            return self.map(u(">H", self.take(2))[0])
        if t == 0xDF:
            return self.map(u(">I", self.take(4))[0])
        raise ValueError(f"지원하지 않는 MessagePack 형식 0x{t:02x}")

    def map(self, n: int) -> dict:
        d = {}
        for _ in range(n):
            k = self.obj()
            d[k] = self.obj()
        return d


def packb(o: Any) -> bytes:
    o = _prep(o)
    if _msgpack is not None:
        return _msgpack.packb(o, use_bin_type=True, use_single_float=False)
    out = bytearray()
    _pack(o, out)
    return bytes(out)


def unpackb(b: bytes) -> Any:
    if _msgpack is not None:
        return _msgpack.unpackb(b, raw=False, strict_map_key=False)
    r = _Reader(b)
    o = r.obj()
    if r.i != len(r.b):
        raise ValueError(f"MessagePack 본문 뒤에 {len(r.b) - r.i}바이트가 남았습니다")
    return o


# ------------------------------------------------------------------ 바이트 흐름 틀
_LEN = struct.Struct("<I")
MAX_FRAME = 64 * 1024 * 1024


def frame(payload: bytes) -> bytes:
    return _LEN.pack(len(payload)) + payload


class FrameSplitter:
    """바이트 흐름에서 u32 길이 접두 틀을 잘라 낸다(부분 수신 허용)."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes):
        self.buf += data
        out = []
        while len(self.buf) >= 4:
            n = _LEN.unpack_from(self.buf, 0)[0]
            if n > MAX_FRAME:
                raise ValueError(f"틀 길이 {n}이 상한 {MAX_FRAME}을 넘습니다")
            if len(self.buf) < 4 + n:
                break
            out.append(bytes(self.buf[4:4 + n]))
            del self.buf[:4 + n]
        return out
