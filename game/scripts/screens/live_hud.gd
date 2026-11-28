## 주행 HUD 바인딩: 상태 프레임·사건을 받아 값만 넣는다. 배치는 scenes/screens/live_hud.tscn, 모양은 Look.
## 기본으로 보이는 것(디자인 A안): 왼쪽 위 바퀴·시간·채점, 오른쪽 위 코스 지도, 아래 가운데 속도.
## 멈추면(Space·Esc) 가로 낱말: 계속 / 다시 / 카메라 / 배속 / 표시 / 끝내기. 표시 = 바퀴 기록·계기·LiDAR 위·LiDAR 점·궤적.
extends Control

signal camera_cycle
signal layer_toggled(name: String, on: bool)

const SPEEDS := [0.5, 1.0, 2.0, 4.0, 0.0]          # 0 = 가능한 한 빨리
const VIEW_ITEMS := [["laps", "바퀴 기록"], ["gauge", "계기"], ["scope", "LiDAR 위"], ["lidar", "LiDAR 점"], ["plan", "궤적"]]

var session: Dictionary = {}
var track: Dictionary = {}
var _state: Dictionary = {}
var _accum := 1.0
var _speed_i := 1
var _laps: Array = []
var _sector_labels: Array = []
var view := {"laps": false, "gauge": false, "scope": false, "lidar": true, "plan": true}
var stage_text := ""
var minimap: Minimap:
	get:
		return %Minimap


func _ready() -> void:
	%Resume.pressed.connect(toggle_pause)
	%Restart.pressed.connect(restart)
	%End.pressed.connect(end_session)
	%SpeedBtn.pressed.connect(func(): cycle_speed(1, true))
	%CamBtn.pressed.connect(func(): camera_cycle.emit())
	var pm: PopupMenu = (%ViewBtn as MenuButton).get_popup()
	pm.add_theme_font_size_override("font_size", 26)
	for i in VIEW_ITEMS.size():
		pm.add_check_item(VIEW_ITEMS[i][1], i)
	pm.id_pressed.connect(func(i: int): set_view(VIEW_ITEMS[i][0], not bool(view[VIEW_ITEMS[i][0]])))
	_apply_view()


# ------------------------------------------------------------------ 세션·코스
func bind_session(s: Dictionary) -> void:
	session = s
	_laps.clear()
	minimap.contacts = []
	_rebuild_laps()
	show_banner("")


func bind_track(t: Dictionary) -> void:
	track = t
	minimap.set_course(t.get("outline", []), t.get("bounds", []))
	minimap.gates = t.get("sectors", [])
	minimap.boxes = t.get("boxes", [])
	for c in %Sectors.get_children():
		c.queue_free()
	_sector_labels.clear()
	var n: int = maxi(1, t.get("sectors", []).size())
	for k in n:
		var v := VBoxContainer.new()
		v.mouse_filter = Control.MOUSE_FILTER_IGNORE
		var name_l := Label.new()
		name_l.theme_type_variation = &"Name"
		name_l.text = "구간 %d" % (k + 1)
		var val := Label.new()
		val.theme_type_variation = &"ValueNum"
		val.text = "—"
		val.custom_minimum_size = Vector2(110, 0)
		v.add_child(name_l)
		v.add_child(val)
		%Sectors.add_child(v)
		_sector_labels.append(val)


func show_banner(text: String) -> void:
	stage_text = text
	%Banner.visible = text != ""
	%Banner.text = text


# ------------------------------------------------------------------ 상태·사건
func on_state(st: Dictionary) -> void:
	_state = st
	var cars: Array = st.get("cars", [])
	if cars.size() > 0:
		var c: Dictionary = cars[0]
		minimap.car = Vector3(float(c.x), float(c.y), float(c.yaw))
		minimap.has_car = true
	minimap.boxes_on = bool(st.get("boxes_on", false))
	var paused := String(st.get("status", "")) == "paused"
	if %PauseMenu.visible != paused:
		%PauseMenu.visible = paused
		if paused:
			%Resume.grab_focus()
	if view.scope:
		var scans: Array = st.get("scans", [])
		if scans.size() > 0:
			(%Scope as LidarScope).set_scan(scans[0], track.get("lidar", {}))


func on_event(ev: Dictionary) -> void:
	match String(ev.get("type", "")):
		"lap":
			_laps.append(ev)
			_rebuild_laps()
		"contact":
			if bool(ev.get("counted", true)):
				minimap.contacts.append({"x": ev.get("x", 0.0), "y": ev.get("y", 0.0), "hard": ev.get("hard", false)})
				minimap.queue_redraw()


func _rebuild_laps() -> void:
	var g: GridContainer = %LapRows
	for c in g.get_children():
		c.queue_free()
	var best := INF
	for e in _laps:
		best = minf(best, float(e.time))
	for e in _laps.slice(-10):
		var bestrow := absf(float(e.time) - best) < 1e-6
		for txt in [str(int(e.lap)), Fmt.lap(e.time), "" if bool(e.clean) else "접촉 %d" % int(e.contacts)]:
			var l := Label.new()
			l.theme_type_variation = &"ValueNum"
			l.text = txt
			l.add_theme_color_override("font_color", Look.WHITE if bestrow else Look.DIM)
			g.add_child(l)


func _process(dt: float) -> void:
	_accum += dt
	if _accum < 0.1:          # 숫자는 초당 10번
		return
	_accum = 0.0
	var st := _state
	var status := String(st.get("status", ""))
	var rt := float(st.get("rt", 0.0))
	%Rate.text = ("배속 %.2f×" % rt) if status == "running" else ""
	%SpeedBtn.text = "배속 " + ("최대" if SPEEDS[_speed_i] == 0.0 else (str(SPEEDS[_speed_i]).trim_suffix(".0") + "×"))
	var tl = st.get("t_left")
	%TimeLeft.text = _clock(tl) if tl != null else "—"
	var rm = st.get("remove_at_s")
	if session.get("boxes", []).is_empty():
		%BoxState.text = "없음"
	elif bool(st.get("boxes_on", false)):
		%BoxState.text = ("%s에 치움" % _clock(rm)) if rm != null else "있음"
	else:
		%BoxState.text = "치움"
	var cars: Array = st.get("cars", [])
	if cars.is_empty():
		return
	var c: Dictionary = cars[0]
	# 채점 항목 = 무충돌 연속 바퀴의 최댓값. 지금 이어가는 연속은 아래 줄
	%Streak.text = str(int(c.get("best_streak", 0)))
	%StreakNow.text = str(int(c.get("clean_streak", 0)))
	%Best.text = Fmt.lap(c.get("best_lap"))
	var timing: bool = c.get("lap_t") != null
	%Lap.text = str(int(c.get("lap", 0)) + 1) if timing else "—"
	%LapTime.text = Fmt.lap(c.get("lap_t")) if timing else "—"
	var last = c.get("last_lap")
	%Last.text = Fmt.lap(last)
	%Last.add_theme_color_override("font_color", Look.lap_color(last, c.get("best_lap")))
	var cur: Array = c.get("cur_sectors", [])
	var bests: Array = c.get("best_sectors", [])
	for k in _sector_labels.size():
		var l: Label = _sector_labels[k]
		if k < cur.size():
			l.text = Fmt.lap(cur[k])
			var b = bests[k] if k < bests.size() else null
			l.add_theme_color_override("font_color", Look.WHITE if b == null or float(cur[k]) <= float(b) + 1e-6 else Look.DIM)
		elif k == cur.size() and timing:
			l.text = Fmt.lap(c.get("sector_t"))
			l.add_theme_color_override("font_color", Look.WHITE)
		else:
			l.text = "—"
			l.add_theme_color_override("font_color", Look.DIM)
	%Speed.text = "%.2f" % absf(float(c.get("v", 0.0)))
	%Sub.text = "m/s   요구 %.2f" % float(c.get("cmd_v", 0.0))
	if view.gauge:
		%Steer.text = "%s° / %s°" % [Fmt.deg(float(c.get("steer", 0.0))), Fmt.deg(float(c.get("cmd_steer", 0.0)))]
		%Accel.text = Fmt.signed(float(c.get("ax", 0.0)), 1)
		%Lat.text = Fmt.signed(float(c.get("ay", 0.0)), 1)
	minimap.queue_redraw()


func _clock(t: Variant) -> String:
	if t == null:
		return "—"
	var x := maxf(0.0, float(t))
	return "%d:%04.1f" % [int(x) / 60, fmod(x, 60.0)]


# ------------------------------------------------------------------ 조작
func toggle_pause() -> void:
	var s := String(_state.get("status", ""))
	if s == "running":
		SimLink.command("pause")
	elif s in ["paused", "ready"]:
		SimLink.command("resume")


func restart() -> void:
	_laps.clear()
	_rebuild_laps()
	minimap.contacts = []
	SimLink.command("reset")
	SimLink.command("start")


func end_session() -> void:
	SimLink.command("end")


func cycle_speed(d: int, wrap: bool = false) -> void:
	if wrap:
		_speed_i = (_speed_i + d + SPEEDS.size()) % SPEEDS.size()
	else:
		_speed_i = clampi(_speed_i + d, 0, SPEEDS.size() - 1)
	SimLink.command_with("speed", {"x": SPEEDS[_speed_i]})


func set_camera_label(mode: String) -> void:
	%CamBtn.text = "카메라 " + {"chase": "추격", "top": "위", "free": "자유"}.get(mode, mode)


func set_view(name: String, on: bool) -> void:
	view[name] = on
	_apply_view()
	if name in ["lidar", "plan"]:
		layer_toggled.emit(name, on)


func _apply_view() -> void:
	%LapsPanel.visible = bool(view.laps)
	%GaugePanel.visible = bool(view.gauge)
	%ScopePanel.visible = bool(view.scope)
	var pm: PopupMenu = (%ViewBtn as MenuButton).get_popup()
	for i in VIEW_ITEMS.size():
		pm.set_item_checked(i, bool(view[VIEW_ITEMS[i][0]]))


func _unhandled_input(ev: InputEvent) -> void:
	if not visible:
		return
	if ev.is_action_pressed("race_toggle") or ev.is_action_pressed("race_end"):
		toggle_pause()
	elif ev.is_action_pressed("race_reset"):
		restart()
	elif ev.is_action_pressed("camera_next"):
		camera_cycle.emit()
	elif ev.is_action_pressed("speed_up"):
		cycle_speed(1)
	elif ev.is_action_pressed("speed_down"):
		cycle_speed(-1)
	elif ev.is_action_pressed("view_laps"):
		set_view("laps", not view.laps)
	elif ev.is_action_pressed("view_gauge"):
		set_view("gauge", not view.gauge)
	elif ev.is_action_pressed("view_scope"):
		set_view("scope", not view.scope)
	elif ev.is_action_pressed("lidar_toggle"):
		set_view("lidar", not view.lidar)
	elif ev.is_action_pressed("view_plan"):
		set_view("plan", not view.plan)
