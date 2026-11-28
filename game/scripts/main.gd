## 앱의 흐름(M1): 메인 -> 레이스 준비 -> 주행 -> 결과. 화면 전환, 카메라, 서버 메시지 배달만 한다.
## 장면(3D)은 모든 화면의 배경이다(디자인 A안): 메인 = 서 있는 차, 준비·결과 = 코스 조감, 주행 = 추격.
extends Node

@onready var world: Node3D = $World
@onready var menu: Control = $UI/Menu
@onready var setup: Control = $UI/Setup
@onready var live: Control = $UI/Live
@onready var result_screen: Control = $UI/Result

var screen := ""
var last_spec: Dictionary = {}
var track: Dictionary = {}
var _flow_spec: Dictionary = {}
var _flow_applied := false
var _go_at := -1.0
var _run_at := -1.0
var _live_camera := "chase"


func _ready() -> void:
	# 모양은 Look 한 곳에서: 화면 넷의 뿌리에 테마를 건다(자식이 물려받는다)
	get_tree().root.theme = Look.theme()
	for c in [menu, setup, live, result_screen]:
		(c as Control).theme = Look.theme()
	Keys.install()
	SimLink.connected.connect(_on_connected)
	SimLink.disconnected.connect(_on_disconnected)
	SimLink.catalog_received.connect(_on_catalog)
	SimLink.preview_received.connect(_on_preview)
	SimLink.stage_received.connect(_on_stage)
	SimLink.session_received.connect(_on_session)
	SimLink.track_received.connect(_on_track)
	SimLink.state_received.connect(_on_state)
	SimLink.event_received.connect(func(ev): live.on_event(ev))
	SimLink.result_received.connect(_on_result)
	SimLink.error_received.connect(_on_error)
	menu.run_pressed.connect(func(): show_screen("setup"))
	menu.quit_pressed.connect(func(): App.finish())
	setup.go_pressed.connect(go)
	setup.back_pressed.connect(func(): show_screen("main"))
	setup.pick_changed.connect(_on_pick)
	live.camera_cycle.connect(func():
		_live_camera = world.next_camera()
		live.set_camera_label(_live_camera))
	live.layer_toggled.connect(func(n, on): world.set_layer(n, on))
	result_screen.again_pressed.connect(func(): go(last_spec))
	result_screen.change_pressed.connect(func(): show_screen("setup"))
	# 자동 흐름(시험·캡처): --spec=<json> 으로 고르고 --go-after=<초> 뒤 출발(메인에서 달리기를 먼저 누른다)
	if App.args.has("spec"):
		var parsed = JSON.parse_string(String(App.args["spec"]))
		if typeof(parsed) == TYPE_DICTIONARY:
			_flow_spec = parsed
	if App.args.has("camera"):
		_live_camera = String(App.args["camera"])
		live.set_camera_label(_live_camera)
	show_screen("main")
	var ep := App.server_endpoint()
	SimLink.open(String(App.arg("transport", "tcp")), ep[0], ep[1])


func show_screen(name: String) -> void:
	screen = name
	menu.visible = name == "main"
	setup.visible = name == "setup"
	live.visible = name == "live"
	result_screen.visible = name == "result"
	match name:
		"main":
			world.set_camera("show")
			_reload_preview()
			menu.focus_default()
		"setup":
			world.set_camera("overview")
			_reload_preview()
			setup.focus_default()
		"live":
			world.set_camera(_live_camera)
		"result":
			world.set_camera("overview")
	App.screen_entered(name)
	print("TENTH_SCREEN ", name)


## 메뉴·준비로 돌아오면 장면을 세션 코스에서 미리 보기 코스로 되돌린다.
func _reload_preview() -> void:
	if SimLink.status == "connected" and not setup.pick.course.is_empty():
		setup._last_preview_key = ""
		setup._request_preview()


func go(spec: Dictionary) -> void:
	if spec.is_empty():
		return
	last_spec = spec.duplicate(true)
	live.show_banner("준비 중")
	show_screen("live")
	SimLink.start_session(spec, true)


func _on_connected(_info: Dictionary) -> void:
	menu.set_message("")
	setup.set_connected(true)


func _on_disconnected(reason: String) -> void:
	setup.set_connected(false)
	menu.set_message(reason)
	setup.set_message(reason)


func _on_catalog(cat: Dictionary) -> void:
	setup.bind_catalog(cat)
	if not _flow_spec.is_empty() and not _flow_applied:
		_flow_applied = true
		setup.apply_spec(_flow_spec)
		if App.args.has("go-after"):
			_run_at = App.elapsed_s() + float(App.args["go-after"]) * 0.5
			_go_at = App.elapsed_s() + float(App.args["go-after"])


func _on_pick(p: Dictionary) -> void:
	var cat: Dictionary = setup.catalog
	var name_of := func(items: Array, ident: String) -> String:
		for it in items:
			if String(it.id) == ident:
				return String(it.get("car_label", it.get("name", "—")))
		return "—"
	menu.bind_now(name_of.call(cat.get("physics", []), String(p.physics)), name_of.call(cat.get("courses", []), String(p.course)),
		name_of.call(cat.get("drivers", []), String(p.driver)))


func _on_preview(pv: Dictionary) -> void:
	if screen in ["main", "setup"]:
		world.build_preview(pv)
		world.set_camera("show" if screen == "main" else "overview")


func _on_stage(_st: Dictionary) -> void:
	if screen == "live":
		live.show_banner("준비 중")


func _on_session(s: Dictionary) -> void:
	live.bind_session(s)
	if screen != "live":
		show_screen("live")


func _on_track(t: Dictionary) -> void:
	if not t.has("hoses"):
		return
	track = t
	world.build(t)
	world.set_camera(_live_camera)
	live.bind_track(t)


func _on_state(st: Dictionary) -> void:
	if screen != "live":
		return
	world.apply_state(st)
	live.on_state(st)
	var s := String(st.get("status", ""))
	if s == "paused":
		live.show_banner("일시정지")
	elif s in ["running", "finished"]:
		live.show_banner("")


func _on_result(res: Dictionary) -> void:
	result_screen.bind(res, track)
	show_screen("result")


func _on_error(text: String) -> void:
	if screen == "live" and String(SimLink.state.get("status", "")) in ["building", "idle", ""]:
		show_screen("setup")
	if screen == "setup":
		setup.set_message(text)
	elif screen == "main":
		menu.set_message(text)
	else:
		live.show_banner(text)


func _process(_dt: float) -> void:
	if _run_at > 0.0 and App.elapsed_s() >= _run_at:
		_run_at = -1.0
		show_screen("setup")
	if _go_at > 0.0 and App.elapsed_s() >= _go_at:
		_go_at = -1.0
		print("TENTH_FLOW_GO ", JSON.stringify(setup.spec()))
		go(setup.spec())


func _unhandled_input(ev: InputEvent) -> void:
	if screen == "setup" and (ev.is_action_pressed("ui_go") or ev.is_action_pressed("race_toggle")):
		go(setup.spec())
	elif screen == "setup" and ev.is_action_pressed("ui_cancel"):
		show_screen("main")
	elif ev.is_action_pressed("app_quit"):
		App.finish()
