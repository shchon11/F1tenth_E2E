## MessagePack 부분집합 (서버 f1sim/server/wire.py와 같은 규격).
## - 표준 MessagePack은 big-endian이고, PackedByteArray.decode_*는 little-endian이라 바이트를 뒤집어 읽는다.
## - {"$nd": "f4", "shape": [...], "data": <bin>} 사전은 Packed*Array로 바꾼다(data는 little-endian 원시 바이트).
class_name MsgPack
extends RefCounted

class _R:
	var b: PackedByteArray
	var i: int = 0
	var err: String = ""

	func _init(bytes: PackedByteArray) -> void:
		b = bytes

	func need(n: int) -> bool:
		if i + n > b.size():
			err = "본문이 잘렸습니다(%d바이트 필요, %d 남음)" % [n, b.size() - i]
			return false
		return true

	func u8() -> int:
		if not need(1):
			return 0
		i += 1
		return b[i - 1]

	func be(n: int) -> PackedByteArray:
		# n바이트를 읽어 little-endian 순서로 뒤집어 돌려준다
		if not need(n):
			return PackedByteArray()
		var s := b.slice(i, i + n)
		s.reverse()
		i += n
		return s

	func uint(n: int) -> int:
		if n == 1:
			return u8()
		var s := be(n)
		if s.is_empty():
			return 0
		match n:
			2: return s.decode_u16(0)
			4: return s.decode_u32(0)
			_: return s.decode_s64(0)   # u64는 GDScript int(s64) 범위만

	func sint(n: int) -> int:
		var s := be(n)
		if s.is_empty():
			return 0
		match n:
			1: return s.decode_s8(0)
			2: return s.decode_s16(0)
			4: return s.decode_s32(0)
			_: return s.decode_s64(0)

	func raw(n: int) -> PackedByteArray:
		if not need(n):
			return PackedByteArray()
		var s := b.slice(i, i + n)
		i += n
		return s

	func text(n: int) -> String:
		return raw(n).get_string_from_utf8()

	func obj() -> Variant:
		var t := u8()
		if err != "":
			return null
		if t < 0x80:
			return t
		if t >= 0xE0:
			return t - 0x100
		if t <= 0x8F:
			return map(t & 0x0F)
		if t <= 0x9F:
			return arr(t & 0x0F)
		if t <= 0xBF:
			return text(t & 0x1F)
		match t:
			0xC0: return null
			0xC2: return false
			0xC3: return true
			0xC4: return raw(uint(1))
			0xC5: return raw(uint(2))
			0xC6: return raw(uint(4))
			0xCA:
				var s := be(4)
				return s.decode_float(0) if not s.is_empty() else 0.0
			0xCB:
				var s := be(8)
				return s.decode_double(0) if not s.is_empty() else 0.0
			0xCC: return uint(1)
			0xCD: return uint(2)
			0xCE: return uint(4)
			0xCF: return uint(8)
			0xD0: return sint(1)
			0xD1: return sint(2)
			0xD2: return sint(4)
			0xD3: return sint(8)
			0xD9: return text(uint(1))
			0xDA: return text(uint(2))
			0xDB: return text(uint(4))
			0xDC: return arr(uint(2))
			0xDD: return arr(uint(4))
			0xDE: return map(uint(2))
			0xDF: return map(uint(4))
		err = "지원하지 않는 형식 0x%02x" % t
		return null

	func arr(n: int) -> Array:
		var a := []
		a.resize(n)
		for k in n:
			a[k] = obj()
		return a

	func map(n: int) -> Variant:
		var d := {}
		for k in n:
			var key = obj()
			d[key] = obj()
		if d.has("$nd"):
			return MsgPack.nd_to_packed(d)
		return d


## 본문 -> Variant. 실패하면 {"_error": "..."}.
static func decode(bytes: PackedByteArray) -> Variant:
	var r := _R.new(bytes)
	var v = r.obj()
	if r.err != "":
		return {"_error": r.err}
	if r.i != bytes.size():
		return {"_error": "본문 뒤에 %d바이트가 남았습니다" % (bytes.size() - r.i)}
	return v


## 배열 사전 -> Packed*Array (모양은 무시하고 평평하게). 모르는 형은 원래 사전 그대로.
static func nd_to_packed(d: Dictionary) -> Variant:
	var data: PackedByteArray = d.get("data", PackedByteArray())
	match d["$nd"]:
		"f4": return data.to_float32_array()
		"f8": return data.to_float64_array()
		"i4": return data.to_int32_array()
		"u1": return data
	return d


## Variant -> 본문 (앱 -> 서버 메시지용: null, bool, int, float, String, Array, Dictionary, PackedByteArray)
static func encode(v: Variant) -> PackedByteArray:
	var out := PackedByteArray()
	_enc(v, out)
	return out


static func _be(n: int, value: int, out: PackedByteArray) -> void:
	var s := PackedByteArray()
	s.resize(8)
	s.encode_s64(0, value)
	for k in range(n - 1, -1, -1):
		out.append(s[k])


static func _enc(v: Variant, out: PackedByteArray) -> void:
	match typeof(v):
		TYPE_NIL:
			out.append(0xC0)
		TYPE_BOOL:
			out.append(0xC3 if v else 0xC2)
		TYPE_INT:
			var x: int = v
			if x >= 0 and x < 128:
				out.append(x)
			elif x < 0 and x >= -32:
				out.append(x & 0xFF)
			elif x >= 0 and x < 0x100:
				out.append(0xCC); _be(1, x, out)
			elif x >= 0 and x < 0x10000:
				out.append(0xCD); _be(2, x, out)
			elif x >= 0 and x < 0x100000000:
				out.append(0xCE); _be(4, x, out)
			elif x >= -0x80000000 and x < 0:
				out.append(0xD2); _be(4, x, out)
			else:
				out.append(0xD3); _be(8, x, out)
		TYPE_FLOAT:
			var s := PackedByteArray()
			s.resize(8)
			s.encode_double(0, v)
			s.reverse()
			out.append(0xCB)
			out.append_array(s)
		TYPE_STRING, TYPE_STRING_NAME:
			var b := String(v).to_utf8_buffer()
			var n := b.size()
			if n < 32:
				out.append(0xA0 | n)
			elif n < 0x100:
				out.append(0xD9); _be(1, n, out)
			elif n < 0x10000:
				out.append(0xDA); _be(2, n, out)
			else:
				out.append(0xDB); _be(4, n, out)
			out.append_array(b)
		TYPE_PACKED_BYTE_ARRAY:
			var n: int = v.size()
			if n < 0x100:
				out.append(0xC4); _be(1, n, out)
			elif n < 0x10000:
				out.append(0xC5); _be(2, n, out)
			else:
				out.append(0xC6); _be(4, n, out)
			out.append_array(v)
		TYPE_ARRAY:
			var n: int = v.size()
			if n < 16:
				out.append(0x90 | n)
			else:
				out.append(0xDC); _be(2, n, out)
			for e in v:
				_enc(e, out)
		TYPE_DICTIONARY:
			var n: int = v.size()
			if n < 16:
				out.append(0x80 | n)
			else:
				out.append(0xDE); _be(2, n, out)
			for k in v:
				_enc(k, out)
				_enc(v[k], out)
		_:
			push_error("MsgPack: 직렬화할 수 없는 타입 %s" % type_string(typeof(v)))
			out.append(0xC0)
