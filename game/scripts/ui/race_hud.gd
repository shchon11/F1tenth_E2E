## 레이스 HUD(스파이크): 세션 띠(위) + 계측 한 줄 판(아래 가운데) + 키 안내 띠(맨 아래).
## 디자인: 02_direction.md(피트 월), 03_ia.md §1, 05_components.md(SessionStrip, Plate, DigitReadout, TimeValue, HintBar).
## 모든 색은 Tokens(tokens.css에서 생성), 글자 단계는 테마 형 변형(ReadoutXL, Caps ...).
extends Control

var speed: DigitReadout
var lbl := {}
var dot: ColorRect
var _accum := 0.0
var _car: Dictionary = {}
var _state: Dictionary = {}
var title_venue := "Korea 2026"
var title_driver := "Follow-the-Gap"


func _ready() -> void:
	set_anchors_preset(PRESET_FULL_RECT)
	mouse_filter = MOUSE_FILTER_IGNORE
	_build_strip()
	_build_plate()
	_build_hints()


func _label(text: String, variation: StringName, color: Color = Color(0, 0, 0, 0)) -> Label:
	var l := Label.new()
	l.text = text
	l.theme_type_variation = variation
	if color.a > 0.0:
		l.add_theme_color_override("font_color", color)
	l.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	return l


func _keycap(k: String) -> PanelContainer:
	var p := PanelContainer.new()
	p.theme_type_variation = &"KeyCap"
	p.add_child(_label(k, &"Micro", Tokens.C_TEXT_2))
	return p


func _build_strip() -> void:
	var strip := PanelContainer.new()
	strip.theme_type_variation = &"Strip"
	strip.set_anchors_preset(PRESET_TOP_WIDE)
	strip.custom_minimum_size = Vector2(0, 44)
	add_child(strip)
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 14)
	strip.add_child(row)
	var chip := PanelContainer.new()
	chip.theme_type_variation = &"ChipAccent"
	chip.add_child(_label("1/10", &"Caps", Tokens.C_TEXT_ON_ACCENT))
	chip.size_flags_vertical = SIZE_SHRINK_CENTER
	row.add_child(chip)
	row.add_child(_label("레이스", &"BodyS", Tokens.C_TEXT_3))
	row.add_child(_label("/", &"BodyS", Tokens.C_TEXT_4))
	row.add_child(_label("자유 주행", &"TitleS"))
	row.add_child(_label("/", &"BodyS", Tokens.C_TEXT_4))
	lbl.venue = _label(title_venue, &"BodyS", Tokens.C_TEXT_2)
	row.add_child(lbl.venue)
	row.add_child(_label("·", &"BodyS", Tokens.C_TEXT_4))
	lbl.driver = _label(title_driver, &"BodyS", Tokens.C_TEXT_2)
	row.add_child(lbl.driver)
	var spacer := Control.new()
	spacer.size_flags_horizontal = SIZE_EXPAND_FILL
	row.add_child(spacer)
	dot = ColorRect.new()
	dot.custom_minimum_size = Vector2(8, 8)
	dot.size_flags_vertical = SIZE_SHRINK_CENTER
	dot.color = Tokens.C_TEXT_4
	row.add_child(dot)
	lbl.status = _label("연결 중", &"TitleS")
	row.add_child(lbl.status)
	lbl.rt = _label("배속 —", &"DataS", Tokens.C_TEXT_3)
	row.add_child(lbl.rt)
	var spacer2 := Control.new()
	spacer2.custom_minimum_size = Vector2(24, 0)
	row.add_child(spacer2)


func _cell(parent: Container, caption: String) -> VBoxContainer:
	var v := VBoxContainer.new()
	v.add_theme_constant_override("separation", 2)
	v.add_child(_label(caption, &"Caps"))
	parent.add_child(v)
	return v


func _sep(parent: Container) -> void:
	var s := ColorRect.new()
	s.color = Tokens.C_LINE_STRONG
	s.custom_minimum_size = Vector2(1, 64)
	s.size_flags_vertical = SIZE_SHRINK_CENTER
	parent.add_child(s)


func _build_plate() -> void:
	var plate := PanelContainer.new()
	plate.theme_type_variation = &"Plate"
	plate.anchor_left = 0.5
	plate.anchor_right = 0.5
	plate.anchor_top = 1.0
	plate.anchor_bottom = 1.0
	plate.grow_horizontal = GROW_DIRECTION_BOTH
	plate.grow_vertical = GROW_DIRECTION_BEGIN
	plate.offset_bottom = -36 - 24
	add_child(plate)
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 28)
	plate.add_child(row)
	# 속도: 고정폭 큰 숫자 + 단위
	var sp := HBoxContainer.new()
	sp.add_theme_constant_override("separation", 8)
	speed = DigitReadout.new()
	speed.type_variation = &"ReadoutL"
	speed.font_size = 72
	speed.color = Tokens.C_TEXT
	sp.add_child(speed)
	var unit := VBoxContainer.new()
	unit.alignment = BoxContainer.ALIGNMENT_END
	unit.add_child(_label("m/s", &"Data", Tokens.C_TEXT_2))
	lbl.cmd = _label("요구 —", &"DataS", Tokens.C_ACCENT)
	unit.add_child(lbl.cmd)
	sp.add_child(unit)
	row.add_child(sp)
	_sep(row)
	var c := _cell(row, "바퀴")
	lbl.lap = _label("—", &"ReadoutM")
	c.add_child(lbl.lap)
	c = _cell(row, "이번 바퀴")
	lbl.lap_t = _label("—", &"ReadoutM")
	c.add_child(lbl.lap_t)
	c = _cell(row, "직전")
	lbl.last = _label("—", &"ReadoutM")
	c.add_child(lbl.last)
	c = _cell(row, "세션 최고")
	lbl.best = _label("—", &"ReadoutM", Tokens.C_TIME_BEST)
	c.add_child(lbl.best)
	_sep(row)
	c = _cell(row, "무충돌 연속")
	lbl.streak = _label("—", &"ReadoutM")
	c.add_child(lbl.streak)
	c = _cell(row, "접촉")
	lbl.contacts = _label("0", &"ReadoutM")
	c.add_child(lbl.contacts)
	c = _cell(row, "조향")
	lbl.steer = _label("—", &"Data", Tokens.C_TEXT_2)
	c.add_child(lbl.steer)


func _build_hints() -> void:
	var bar := PanelContainer.new()
	bar.theme_type_variation = &"HintBar"
	bar.set_anchors_preset(PRESET_BOTTOM_WIDE)
	bar.grow_vertical = GROW_DIRECTION_BEGIN
	bar.custom_minimum_size = Vector2(0, 36)
	add_child(bar)
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 10)
	bar.add_child(row)
	for action in ["race_toggle", "race_reset", "camera_next", "lidar_toggle", "app_quit"]:
		var spec: Array = Keys.ACTIONS[action]
		row.add_child(_keycap(spec[2]))
		row.add_child(_label(spec[1], &"BodyS", Tokens.C_TEXT_2))
		var gap := Control.new()
		gap.custom_minimum_size = Vector2(10, 0)
		row.add_child(gap)
	var spacer := Control.new()
	spacer.size_flags_horizontal = SIZE_EXPAND_FILL
	row.add_child(spacer)
	lbl.diag = _label("시뮬 서버 · —", &"DataS", Tokens.C_TEXT_4)
	row.add_child(lbl.diag)


func set_state(state: Dictionary, focus: int = 0) -> void:
	_state = state
	var cars: Array = state.get("cars", [])
	if focus < cars.size():
		_car = cars[focus]


func _process(dt: float) -> void:
	# 숫자는 초당 10번만 바꾼다(02_direction.md §7). 점 깜빡임만 매 프레임.
	var st := String(_state.get("status", ""))
	if st == "running":
		dot.color = Tokens.C_ACCENT if fmod(App.elapsed_s(), 2.0) < 1.4 else Tokens.C_ACCENT_PRESS
	_accum += dt
	if _accum < 0.1:
		return
	_accum = 0.0
	var names := {"running": "주행 중", "ready": "출발 대기", "paused": "일시정지", "stopped": "멈춤"}
	lbl.status.text = names.get(st, "연결 중" if SimLink.status != "connected" else "대기")
	if st != "running":
		dot.color = Tokens.C_WARN if st == "paused" else Tokens.C_TEXT_4
	var rt := float(_state.get("rt", 0.0))
	lbl.rt.text = "배속 %.2f×" % rt if st == "running" else "배속 —"
	lbl.rt.add_theme_color_override("font_color", Tokens.C_WARN if st == "running" and rt < 0.95 else Tokens.C_TEXT_3)
	if _car.is_empty():
		return
	speed.text = Fmt.speed(float(_car.get("v", 0.0)))
	lbl.cmd.text = "요구 %s" % Fmt.speed(float(_car.get("cmd_v", 0.0)))
	lbl.lap.text = str(int(_car.get("lap", 0)) + 1) if _car.get("lap_t") != null else "출발 전"
	lbl.lap_t.text = Fmt.lap(_car.get("lap_t"))
	lbl.last.text = Fmt.lap(_car.get("last_lap"))
	var last = _car.get("last_lap")
	var best = _car.get("best_lap")
	var col := Tokens.C_TEXT
	if last != null and best != null:
		col = Tokens.C_TIME_BEST if absf(float(last) - float(best)) < 1e-6 else Tokens.C_TIME_SLOWER
	lbl.last.add_theme_color_override("font_color", col)
	lbl.best.text = Fmt.lap(best)
	lbl.streak.text = str(int(_car.get("clean_streak", 0)))
	var nc := int(_car.get("contacts", 0))
	lbl.contacts.text = str(nc)
	lbl.contacts.add_theme_color_override("font_color", Tokens.C_DANGER if bool(_car.get("contact", false)) else (Tokens.C_TEXT if nc == 0 else Tokens.C_WARN))
	lbl.steer.text = "%s° · 요구 %s°" % [Fmt.deg(float(_car.get("steer", 0.0))), Fmt.deg(float(_car.get("cmd_steer", 0.0)))]
	var lat := SimLink.percentile(SimLink.stats.latency_ms.slice(-40), 0.5)
	lbl.diag.text = "시뮬 서버 %s · %s · 지연 %.1f ms · 화면 %d fps" % [
		SimLink.transport.to_upper(), "연결됨" if SimLink.status == "connected" else SimLink.status,
		lat, Engine.get_frames_per_second()]
