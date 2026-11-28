## 시뮬 서버 연결(자동 로드 SimLink). 전송 4종(tcp / ws / udp / shm)을 같은 신호로 감싼다.
## - 상태(state)는 가장 최근 것만 쓴다(같은 프레임 안에 여러 장이 오면 마지막 것). 지난 세대(gen)는 버린다.
## - 지연 측정: 서버가 찍은 송신 시각(ts, 유닉스 초)과 받은 시각의 차. 같은 PC에서만 의미가 있다.
##   왕복(ping -> pong)은 원격에서도 의미가 있다.
extends Node

signal connected(info: Dictionary)
signal disconnected(reason: String)
signal track_received(track: Dictionary)
signal state_received(state: Dictionary)
signal event_received(ev: Dictionary)
signal ack_received(ack: Dictionary)
# M1: 레이스 화면(준비 -> 주행 -> 결과)
signal catalog_received(catalog: Dictionary)
signal preview_received(preview: Dictionary)
signal stage_received(stage: Dictionary)
signal session_received(session: Dictionary)
signal result_received(result: Dictionary)
signal error_received(text: String)

const SCHEMA_VERSION := 1

var transport := "tcp"
var host := "127.0.0.1"
var port := 47800
var status := "idle"          # idle | connecting | connected | closed
var hello: Dictionary = {}
var track: Dictionary = {}
var state: Dictionary = {}    # 가장 최근 상태
var gen := 0
var catalog: Dictionary = {}
var session: Dictionary = {}
var result: Dictionary = {}

# 측정
var stats := {
	"frames": 0, "frames_applied": 0, "bytes": 0, "decode_us": [], "latency_ms": [], "rtt_ms": [],
	"seq_gaps": 0, "torn": 0, "ack_ms": [], "cmd_to_state_ms": [],
}
var _cmd_pending: Dictionary = {}
var _last_seq := -1
var _cmd_seq := 0
var _ping_t := 0.0

var _tcp: StreamPeerTCP
var _ws: WebSocketPeer
var _udp: PacketPeerUDP
var _shm: FileAccess
var _shm_seq := -1
var _buf := PackedByteArray()
var _got_hello := false


func open(p_transport: String, p_host: String, p_port: int) -> void:
	transport = p_transport
	host = p_host
	port = p_port
	status = "connecting"
	_buf.clear()
	match transport:
		"tcp":
			_tcp = StreamPeerTCP.new()
			var err := _tcp.connect_to_host(host, port)
			if err != OK:
				_fail("TCP 연결을 시작하지 못했습니다 (%s)" % error_string(err))
		"ws":
			_ws = WebSocketPeer.new()
			_ws.inbound_buffer_size = 1 << 23
			_ws.outbound_buffer_size = 1 << 20
			_ws.max_queued_packets = 4096
			var err := _ws.connect_to_url("ws://%s:%d" % [host, port])
			if err != OK:
				_fail("WebSocket 연결을 시작하지 못했습니다 (%s)" % error_string(err))
		"udp":
			_udp = PacketPeerUDP.new()
			_udp.set_dest_address(host, port)
			_udp.connect_to_host(host, port)
			_send({"k": "hello", "v": SCHEMA_VERSION, "client": "tenth-godot"})
		"shm":
			_shm_open()
		_:
			_fail("모르는 전송 %s" % transport)


func close() -> void:
	if _tcp: _tcp.disconnect_from_host()
	if _ws: _ws.close()
	if _udp: _udp.close()
	_shm = null
	status = "closed"


func command(op: String) -> int:
	_cmd_seq += 1
	var now := Time.get_unix_time_from_system()
	# 입력 -> 화면 지연: 명령을 보낸 때부터 상태(status 또는 gen)가 바뀐 첫 프레임을 받은 때까지
	_cmd_pending = {"seq": _cmd_seq, "t": now, "status": state.get("status", ""), "gen": gen, "op": op}
	_send({"k": "cmd", "v": SCHEMA_VERSION, "op": op, "seq": _cmd_seq})
	print("TENTH_CMD %s seq=%d" % [op, _cmd_seq])
	return _cmd_seq


## 명령에 값을 붙여 보낸다(예: 배속 {x: 2.0}).
func command_with(op: String, extra: Dictionary) -> int:
	_cmd_seq += 1
	var m := {"k": "cmd", "v": SCHEMA_VERSION, "op": op, "seq": _cmd_seq}
	m.merge(extra)
	_cmd_pending = {"seq": _cmd_seq, "t": Time.get_unix_time_from_system(), "status": state.get("status", ""), "gen": gen, "op": op}
	_send(m)
	print("TENTH_CMD %s seq=%d" % [op, _cmd_seq])
	return _cmd_seq


func request(op: String, fields: Dictionary = {}) -> void:
	var m := {"k": "req", "v": SCHEMA_VERSION, "op": op}
	m.merge(fields)
	_send(m)


func start_session(spec: Dictionary, autostart: bool = true) -> void:
	result = {}
	_send({"k": "session", "v": SCHEMA_VERSION, "op": "start", "spec": spec, "autostart": autostart})
	print("TENTH_SESSION_REQUEST " + JSON.stringify(spec))


func ping() -> void:
	_send({"k": "ping", "v": SCHEMA_VERSION, "t": Time.get_unix_time_from_system()})


func _process(_dt: float) -> void:
	if status == "idle" or status == "closed":
		return
	match transport:
		"tcp": _poll_tcp()
		"ws": _poll_ws()
		"udp": _poll_udp()
		"shm": _poll_shm()
	var now := Time.get_unix_time_from_system()
	if status == "connected" and transport != "shm" and now - _ping_t > 0.5:
		_ping_t = now
		ping()


# ------------------------------------------------------------------ 전송별 수신
func _poll_tcp() -> void:
	_tcp.poll()
	var st := _tcp.get_status()
	if st == StreamPeerTCP.STATUS_CONNECTED:
		if status == "connecting":
			status = "connected"
			_tcp.set_no_delay(true)
			_send({"k": "hello", "v": SCHEMA_VERSION, "client": "tenth-godot"})
		var n := _tcp.get_available_bytes()
		if n > 0:
			var r := _tcp.get_data(n)
			if r[0] == OK:
				_buf.append_array(r[1])
				stats.bytes += n
		while _buf.size() >= 4:
			var ln := _buf.decode_u32(0)
			if _buf.size() < 4 + ln:
				break
			var payload := _buf.slice(4, 4 + ln)
			_buf = _buf.slice(4 + ln)
			_on_payload(payload)
	elif st == StreamPeerTCP.STATUS_ERROR or (st == StreamPeerTCP.STATUS_NONE and status == "connected"):
		_fail("서버와 연결이 끊겼습니다")


func _poll_ws() -> void:
	_ws.poll()
	var st := _ws.get_ready_state()
	if st == WebSocketPeer.STATE_OPEN:
		if status == "connecting":
			status = "connected"
			_send({"k": "hello", "v": SCHEMA_VERSION, "client": "tenth-godot"})
		while _ws.get_available_packet_count() > 0:
			var p := _ws.get_packet()
			stats.bytes += p.size()
			_on_payload(p)
	elif st == WebSocketPeer.STATE_CLOSED and status == "connected":
		_fail("WebSocket이 닫혔습니다 (%d)" % _ws.get_close_code())


func _poll_udp() -> void:
	while _udp.get_available_packet_count() > 0:
		var p := _udp.get_packet()
		stats.bytes += p.size()
		if status == "connecting":
			status = "connected"
		_on_payload(p)


func _shm_open() -> void:
	# 파일이 생겼는지만 본다. 읽을 때마다 다시 연다: FileAccess는 stdio 버퍼를 써서(glibc fseek가 버퍼 안 이동이면
	# 다시 읽지 않는다) 같은 핸들로는 서버가 mmap에 쓴 새 값을 못 본다(스파이크에서 확인).
	if not FileAccess.file_exists("/dev/shm/tenth_%d" % port):
		return
	_shm = FileAccess.open("/dev/shm/tenth_%d" % port, FileAccess.READ)
	status = "connected"
	connected.emit({"server": "shm", "transport": "shm"})


func _poll_shm() -> void:
	if _shm == null:
		_shm_open()
		return
	var f := FileAccess.open("/dev/shm/tenth_%d" % port, FileAccess.READ)
	if f == null:
		return
	f.seek(8)
	var s1 := f.get_64()
	if s1 == _shm_seq or (s1 & 1) == 1:
		return
	var ln := f.get_32()
	f.seek(24)
	var p := f.get_buffer(ln)
	var f2 := FileAccess.open("/dev/shm/tenth_%d" % port, FileAccess.READ)
	f2.seek(8)
	var s2 := f2.get_64()
	if s1 != s2:
		stats.torn += 1
		return
	_shm_seq = s1
	stats.bytes += ln
	_on_payload(p)


# ------------------------------------------------------------------ 공통
func _send(m: Dictionary) -> void:
	var body := MsgPack.encode(m)
	match transport:
		"tcp":
			if _tcp and _tcp.get_status() == StreamPeerTCP.STATUS_CONNECTED:
				var hdr := PackedByteArray()
				hdr.resize(4)
				hdr.encode_u32(0, body.size())
				_tcp.put_data(hdr + body)
		"ws":
			if _ws and _ws.get_ready_state() == WebSocketPeer.STATE_OPEN:
				_ws.send(body, WebSocketPeer.WRITE_MODE_BINARY)
		"udp":
			if _udp:
				_udp.put_packet(body)


func _on_payload(p: PackedByteArray) -> void:
	var t0 := Time.get_ticks_usec()
	var m = MsgPack.decode(p)
	var dec := Time.get_ticks_usec() - t0
	if typeof(m) != TYPE_DICTIONARY or m.has("_error"):
		push_warning("SimLink: 메시지를 읽지 못했습니다: %s" % [m])
		return
	if m.get("v", -1) != SCHEMA_VERSION:
		_fail("스키마 버전이 다릅니다: 서버 %s, 앱 %d" % [m.get("v"), SCHEMA_VERSION])
		return
	match m.get("k", ""):
		"hello":
			hello = m
			_got_hello = true
			connected.emit(m)
		"track":
			track = m
			track_received.emit(m)
		"state":
			stats.frames += 1
			_push(stats.decode_us, dec)
			var lat := (Time.get_unix_time_from_system() - float(m.get("ts", 0.0))) * 1000.0
			_push(stats.latency_ms, lat)
			var sq: int = m.get("seq", 0)
			if _last_seq >= 0 and sq > _last_seq + 1 and m.get("gen", 0) == gen:
				stats.seq_gaps += sq - _last_seq - 1
			_last_seq = sq
			if m.get("gen", 0) < gen:
				return
			gen = m.get("gen", 0)
			state = m
			if not _cmd_pending.is_empty() and (m.get("status", "") != _cmd_pending.status or gen != _cmd_pending.gen):
				var d := (Time.get_unix_time_from_system() - float(_cmd_pending.t)) * 1000.0
				stats.cmd_to_state_ms.append(d)
				print("TENTH_CMD_APPLIED %s %.1f ms -> %s" % [_cmd_pending.op, d, m.get("status", "")])
				_cmd_pending = {}
			state_received.emit(m)
		"event":
			event_received.emit(m)
		"catalog":
			catalog = m
			catalog_received.emit(m)
		"preview":
			preview_received.emit(m)
		"stage":
			stage_received.emit(m)
		"session":
			session = m
			gen = maxi(gen, int(m.get("gen", gen)))
			session_received.emit(m)
		"result":
			result = m
			result_received.emit(m)
		"ack":
			if not _cmd_pending.is_empty() and int(m.get("seq", -1)) == int(_cmd_pending.seq):
				_push(stats.ack_ms, (Time.get_unix_time_from_system() - float(_cmd_pending.t)) * 1000.0)
			ack_received.emit(m)
		"pong":
			_push(stats.rtt_ms, (Time.get_unix_time_from_system() - float(m.get("t", 0.0))) * 1000.0)
		"error":
			push_warning("서버 오류: %s" % m.get("note", ""))
			print("TENTH_SERVER_ERROR ", m.get("note", ""))
			error_received.emit(String(m.get("note", "")))


func _push(arr: Array, v: float) -> void:
	arr.append(v)
	if arr.size() > 4000:
		arr.remove_at(0)


func _fail(reason: String) -> void:
	status = "closed"
	push_warning("SimLink: " + reason)
	disconnected.emit(reason)


static func percentile(arr: Array, q: float) -> float:
	if arr.is_empty():
		return NAN
	var s := arr.duplicate()
	s.sort()
	return float(s[clampi(int(round(q * (s.size() - 1))), 0, s.size() - 1)])
