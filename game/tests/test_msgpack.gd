extends TestCase

const FIXTURE := "res://tests/fixtures/state_frame.msgpack"
const EXPECT := "res://tests/fixtures/state_frame.json"


func test_roundtrip_scalars() -> void:
	for v in [null, true, false, 0, 1, 127, 128, 255, 256, 65535, 65536, 4294967295, 4294967296, -1, -32, -33,
			-128, -129, -32768, -32769, -2147483648, -2147483649, 0.5, -1.25e-7, 1234567.875, "", "가", "LiDAR 1081줄"]:
		var b := MsgPack.encode(v)
		var back = MsgPack.decode(b)
		eq(back, v, "왕복 %s" % str(v))


func test_roundtrip_nested() -> void:
	var v := {"k": "cmd", "v": 1, "op": "start", "seq": 3, "args": [1, 2.5, "x", {"a": null}]}
	var back = MsgPack.decode(MsgPack.encode(v))
	eq(back, v, "중첩 사전")


func test_long_containers() -> void:
	var arr := []
	for i in 40:
		arr.append(i * 3)
	eq(MsgPack.decode(MsgPack.encode(arr)), arr, "16개 넘는 배열(array16)")
	var s := "x".repeat(300)
	eq(MsgPack.decode(MsgPack.encode(s)), s, "str16")


func test_truncated_is_error() -> void:
	var b := MsgPack.encode({"k": "state", "seq": 70000})
	var r = MsgPack.decode(b.slice(0, b.size() - 2))
	check(typeof(r) == TYPE_DICTIONARY and r.has("_error"), "잘린 본문은 오류여야 합니다: %s" % str(r))


## 파이썬 서버(f1sim.server.wire)가 만든 실제 상태 프레임을 읽는다 — 두 언어의 규격이 같은지.
func test_python_fixture() -> void:
	var f := FileAccess.open(FIXTURE, FileAccess.READ)
	check(f != null, "고정 자료가 없습니다: game/tools/make_fixtures.py를 돌리세요")
	if f == null:
		return
	var m = MsgPack.decode(f.get_buffer(f.get_length()))
	var want = JSON.parse_string(FileAccess.get_file_as_string(EXPECT))
	eq(m.get("k"), "state", "종류")
	eq(m.get("v"), int(want.v), "스키마 버전")
	eq(m.get("seq"), int(want.seq), "seq")
	eq(m.get("cars").size(), int(want.n_cars), "차 수")
	near(float(m.cars[1].x), float(want.car1_x), 1e-12, "차 1 x(float64)")
	var r: PackedFloat32Array = m.scans[0].ranges
	eq(r.size(), int(want.n_beams), "빔 수")
	near(r[0], float(want.r0_first), 1e-6, "첫 빔(float32 little-endian)")
	near(r[r.size() - 1], float(want.r0_last), 1e-6, "끝 빔")
	var hit: PackedByteArray = m.scans[0].hit
	eq(hit.size(), int(want.n_beams), "맞은 것 배열 길이")
	eq(m.cars[0].get("last_lap"), null if want.car0_last_lap == null else float(want.car0_last_lap), "없는 값은 null")
