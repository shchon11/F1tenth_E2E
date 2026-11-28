## 앱 전역(자동 로드 App): 명령줄 인자, 에이전트용 캡처·측정, 대화형 창 처리.
##
## 사용자 인자(`--` 뒤):
##   --server=host:port  --transport=tcp|ws|udp|shm  --autostart   (연결 뒤 start 명령)
##   --camera=chase|top  --capture=<png 경로>  --capture-after=<초>  --capture-size=1920x1200
##   --metrics=<json 경로>  --quit-after=<초>  --offscreen
## M1 흐름 캡처: --capture-plan=setup:3,live:15,live:40,result:3  --capture-dir=<폴더>
##   화면 이름과 그 화면에 들어간 뒤의 초. 계획을 다 찍으면 앱이 스스로 끝난다(--quit-after-plan=<초>, 기본 1.5).
##   --spec=<json> --go-after=<초> 는 main.gd가 읽는다(준비 화면에서 고르고 출발).
## --offscreen: 화면을 가로채지 않는 에이전트 모드. 창은 실행기(tools/tenth.py)가 만든 override.cfg 덕에
##   포커스를 받지 않는 8×8 창으로 생기고, 장면은 content_scale(viewport)로 캡처 해상도 그대로 렌더된다.
extends Node

var args := {}
var offscreen := false
var _t0_ms := 0
var _captured := false
var _fps_samples: Array = []
var _frame_ms: Array = []
var _last_tick := 0
var _inject: Array = []      # [[동작, 초], ...] 남은 입력 주입
var _plan: Array = []        # [[화면, 초, 찍음], ...]
var _entered := {}           # 화면 -> 처음 들어간 시각
var _plan_done_at := -1.0
var _shots := 0


func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	args = parse_args(OS.get_cmdline_user_args())
	offscreen = args.has("offscreen")
	# --inject=race_toggle@4,race_reset@7 : 정해진 시각에 입력 동작을 넣는다(키보드 없이 입력 경로 전체를 시험)
	for item in String(args.get("inject", "")).split(",", false):
		var p := item.split("@")
		_inject.append([p[0], float(p[1])])
	for item in String(args.get("capture-plan", "")).split(",", false):
		var p := item.split(":")
		_plan.append([p[0], float(p[1]), false])
	_t0_ms = Time.get_ticks_msec()
	var size := capture_size()
	if offscreen:
		get_tree().root.content_scale_mode = Window.CONTENT_SCALE_MODE_VIEWPORT
		get_tree().root.content_scale_size = size
		get_tree().root.content_scale_aspect = Window.CONTENT_SCALE_ASPECT_IGNORE
		Engine.max_fps = int(args.get("max-fps", 60))   # 0 = 제한 없음(렌더 여유 측정용, 수직 동기도 꺼야 함)
	else:
		DisplayServer.window_set_title("TENTH")


static func parse_args(raw: PackedStringArray) -> Dictionary:
	var d := {}
	for a in raw:
		if not a.begins_with("--"):
			continue
		var kv := a.substr(2).split("=", true, 1)
		d[kv[0]] = kv[1] if kv.size() > 1 else true
	return d


func arg(name: String, default: Variant = null) -> Variant:
	return args.get(name, default)


func capture_size() -> Vector2i:
	var s := String(args.get("capture-size", "1920x1200")).split("x")
	return Vector2i(int(s[0]), int(s[1]))


func server_endpoint() -> Array:
	var s := String(args.get("server", "127.0.0.1:47800"))
	var i := s.rfind(":")
	return [s.substr(0, i), int(s.substr(i + 1))]


func elapsed_s() -> float:
	return (Time.get_ticks_msec() - _t0_ms) / 1000.0


func _process(_dt: float) -> void:
	var now := Time.get_ticks_usec()
	if _last_tick > 0:
		_frame_ms.append((now - _last_tick) / 1000.0)
		if _frame_ms.size() > 20000:
			_frame_ms.remove_at(0)
	_last_tick = now
	var t := elapsed_s()
	while not _inject.is_empty() and t >= float(_inject[0][1]):
		var name: String = _inject.pop_front()[0]
		for pressed in [true, false]:
			var ev := InputEventAction.new()
			ev.action = name
			ev.pressed = pressed
			Input.parse_input_event(ev)
		print("TENTH_INJECT %s @ %.2f s" % [name, t])
	if args.has("capture") and not _captured and t >= float(args.get("capture-after", 5.0)):
		_captured = true
		_capture(String(args["capture"]))
	for item in _plan:
		if not item[2] and _entered.has(item[0]) and t - float(_entered[item[0]]) >= float(item[1]):
			item[2] = true
			_shots += 1
			var d := String(args.get("capture-dir", "user://captures"))
			_capture_to(d.path_join("%02d_%s_%ds.png" % [_shots, item[0], int(item[1])]))
	if not _plan.is_empty() and _plan_done_at < 0.0 and _plan.all(func(x): return x[2]):
		_plan_done_at = t
	if _plan_done_at > 0.0 and t - _plan_done_at >= float(args.get("quit-after-plan", 1.5)):
		finish()
	if args.has("quit-after") and t >= float(args["quit-after"]):
		finish()


## 화면 이름이 바뀔 때 main.gd가 부른다(캡처 계획의 기준 시각).
func screen_entered(name: String) -> void:
	if not _entered.has(name):
		_entered[name] = elapsed_s()


func _capture_to(path: String) -> void:
	await RenderingServer.frame_post_draw
	var img := get_viewport().get_texture().get_image()
	var abs_path := path if path.is_absolute_path() else ProjectSettings.globalize_path("res://").path_join(path)
	DirAccess.make_dir_recursive_absolute(abs_path.get_base_dir())
	var err := img.save_png(abs_path)
	print("TENTH_CAPTURE %s %dx%d %s" % [abs_path, img.get_width(), img.get_height(), error_string(err)])


func _capture(path: String) -> void:
	await RenderingServer.frame_post_draw
	var img := get_viewport().get_texture().get_image()
	var abs_path := path if path.is_absolute_path() else ProjectSettings.globalize_path("res://").path_join(path)
	DirAccess.make_dir_recursive_absolute(abs_path.get_base_dir())
	var err := img.save_png(abs_path)
	print("TENTH_CAPTURE %s %dx%d %s" % [abs_path, img.get_width(), img.get_height(), error_string(err)])
	if not args.has("quit-after"):
		finish()


## 측정 결과를 쓰고 끝낸다(메트릭 경로가 있으면).
func finish() -> void:
	if args.has("metrics"):
		var fps := 1000.0 / maxf(SimLink.percentile(_frame_ms, 0.5), 0.001)
		var m := {
			"transport": SimLink.transport, "elapsed_s": elapsed_s(),
			"frames_rendered": Engine.get_frames_drawn(), "fps_median": fps,
			"frame_ms_p50": SimLink.percentile(_frame_ms, 0.5), "frame_ms_p95": SimLink.percentile(_frame_ms, 0.95),
			"frame_ms_p99": SimLink.percentile(_frame_ms, 0.99),
			"state_frames": SimLink.stats.frames, "bytes": SimLink.stats.bytes,
			"latency_ms_p50": SimLink.percentile(SimLink.stats.latency_ms, 0.5),
			"latency_ms_p95": SimLink.percentile(SimLink.stats.latency_ms, 0.95),
			"rtt_ms_p50": SimLink.percentile(SimLink.stats.rtt_ms, 0.5),
			"rtt_ms_p95": SimLink.percentile(SimLink.stats.rtt_ms, 0.95),
			"decode_us_p50": SimLink.percentile(SimLink.stats.decode_us, 0.5),
			"seq_gaps": SimLink.stats.seq_gaps, "torn": SimLink.stats.torn,
			"ack_ms": SimLink.stats.ack_ms, "cmd_to_state_ms": SimLink.stats.cmd_to_state_ms,
			"apply_ms_p50": SimLink.percentile(get_meta("apply_ms", []), 0.5) if has_meta("apply_ms") else NAN,
			"adapter": RenderingServer.get_video_adapter_name(),
			"driver": RenderingServer.get_current_rendering_driver_name(),
			"method": RenderingServer.get_current_rendering_method(),
			"godot": Engine.get_version_info().string,
		}
		var path := String(args["metrics"])
		var abs_path := path if path.is_absolute_path() else ProjectSettings.globalize_path("res://").path_join(path)
		DirAccess.make_dir_recursive_absolute(abs_path.get_base_dir())
		var f := FileAccess.open(abs_path, FileAccess.WRITE)
		f.store_string(JSON.stringify(m, "  "))
		f.close()
		print("TENTH_METRICS " + JSON.stringify(m))
	get_tree().quit()


func record_apply_ms(v: float) -> void:
	var a: Array = get_meta("apply_ms", [])
	a.append(v)
	if a.size() > 20000:
		a.remove_at(0)
	set_meta("apply_ms", a)
