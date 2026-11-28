## 결과 화면 바인딩: 서버의 result 메시지(대회 채점)를 값으로 넣는다. 배치는 scenes/screens/result.tscn, 모양은 Look.
extends Control

signal again_pressed
signal change_pressed

var result: Dictionary = {}


func _ready() -> void:
	%Again.pressed.connect(func(): again_pressed.emit())
	%Change.pressed.connect(func(): change_pressed.emit())
	%DetailBtn.toggled.connect(func(on: bool): %Detail.visible = on)


func _label(text: String, variation: StringName, color: Variant = null) -> Label:
	var l := Label.new()
	l.text = text
	l.theme_type_variation = variation
	if color != null:
		l.add_theme_color_override("font_color", color)
	return l


func _clear(c: Node) -> void:
	for x in c.get_children():
		x.queue_free()


func bind(res: Dictionary, track: Dictionary = {}) -> void:
	result = res
	var sc: Dictionary = res.get("score", {})
	%Streak.text = str(int(sc.get("best_clean_streak", 0)))
	%Best.text = Fmt.lap(sc.get("best_lap"))
	%NLaps.text = str(int(res.get("n_laps", 0)))
	%NContacts.text = str(int(res.get("contacts_counted", 0)))
	%Title.text = "결과" if bool(res.get("finished", true)) else "결과 · 중간에 끝냄"
	# 표기: 무엇으로 어떻게 달렸나
	var lb: Dictionary = res.get("labels", {})
	_clear(%Facts)
	for pair in [["코스", lb.get("course", "—")], ["드라이버", lb.get("driver", "—")],
			["실행", lb.get("exec_label", "—")], ["차", lb.get("car", lb.get("physics", "—"))],
			["규칙", lb.get("rules_line", "—")], ["배치", lb.get("layout", "—")],
			["계시", "시뮬 시간"]]:
		%Facts.add_child(_label(String(pair[0]), &"Small"))
		%Facts.add_child(_label(String(pair[1]), &"Cell"))
	# 바퀴별
	_clear(%LapTable)
	var laps: Array = res.get("laps", [])
	var nsec := 0
	for r in laps:
		nsec = maxi(nsec, r.get("sectors", []).size())
	%LapTable.columns = 3 + nsec
	%LapTable.add_child(_label("바퀴", &"Small"))
	%LapTable.add_child(_label("시간", &"Small"))
	for k in nsec:
		%LapTable.add_child(_label("구간 %d" % (k + 1), &"Small"))
	%LapTable.add_child(_label("", &"Small"))
	var best = sc.get("best_lap")
	var best_sec: Array = []
	for k in nsec:
		var b := INF
		for r in laps:
			var s: Array = r.get("sectors", [])
			if k < s.size():
				b = minf(b, float(s[k]))
		best_sec.append(b)
	var streak_end := int(res.get("best_streak_end_lap", 0))
	var streak_n := int(sc.get("best_clean_streak", 0))
	for r in laps:
		var n := int(r.n)
		var in_streak := streak_n > 0 and n > streak_end - streak_n and n <= streak_end
		%LapTable.add_child(_label(str(n), &"Cell", Look.WHITE if in_streak else Look.DIM))
		var is_best: bool = best != null and absf(float(r.time) - float(best)) < 1e-9
		%LapTable.add_child(_label(Fmt.lap(r.time), &"Cell", Look.WHITE if is_best else Look.DIM))
		var s: Array = r.get("sectors", [])
		for k in nsec:
			var v = s[k] if k < s.size() else null
			var col := Look.DIM
			if v != null and absf(float(v) - float(best_sec[k])) < 1e-9:
				col = Look.WHITE
			%LapTable.add_child(_label(Fmt.lap(v), &"Cell", col))
		if bool(r.clean):
			%LapTable.add_child(_label("무충돌", &"Cell", Look.DIM))
		else:
			%LapTable.add_child(_label("접촉 %d" % int(r.contacts), &"Cell", Look.WHITE))
	# 접촉 목록
	_clear(%ContactTable)
	var contacts: Array = res.get("contacts", []).filter(func(c): return bool(c.get("counted", true)))
	for h in ["시각", "종류", "세기", "바퀴", "구간"]:
		%ContactTable.add_child(_label(h, &"Small"))
	for c in contacts:
		var hard := bool(c.get("hard", false))
		var col := Look.WHITE if hard else Look.DIM
		%ContactTable.add_child(_label("%.1f초" % float(c.t), &"Cell", col))
		%ContactTable.add_child(_label(String(c.get("kind_label", c.get("kind", ""))), &"Cell", col))
		%ContactTable.add_child(_label("%s %.2f m/s" % [String(c.get("severity_label", "")), float(c.get("severity", 0.0))], &"Cell", col))
		%ContactTable.add_child(_label(str(int(c.get("lap", 0))) if int(c.get("lap", 0)) > 0 else "출발", &"Cell", col))
		%ContactTable.add_child(_label(str(int(c.get("sector", 0))), &"Cell", col))
	if contacts.is_empty():
		%ContactTable.add_child(_label("없음", &"Cell"))
	# 미니맵: 접촉 위치
	var mm: Minimap = %Minimap
	if not track.is_empty():
		mm.set_course(track.get("outline", []), track.get("bounds", []))
		mm.gates = track.get("sectors", [])
	mm.boxes = res.get("boxes", [])
	mm.boxes_on = true
	mm.has_car = false
	mm.contacts = contacts
	mm.queue_redraw()
	# 상세(원래 식별자·파일)
	var d: Dictionary = res.get("detail", {})
	var drv: Dictionary = d.get("driver", {})
	var files: Dictionary = res.get("files", {})
	var m: Dictionary = res.get("measured", {})
	var pc: Dictionary = res.get("physics_check", {})
	%Detail.text = "결과 %s\n기록 %s\n다시 보기 %s\n드라이버 %s update %s · %s\n코스 %s · 물리 프로필 %s (적용 확인 %s/%s)\n배속 평균 %s× · 장치 %s" % [
		files.get("json", "—"), files.get("steps", "—"), files.get("replay", "—"),
		drv.get("run", "—"), str(drv.get("update", "—")), drv.get("path", "—"),
		d.get("course_track", "—"), str(d.get("profile", "params.py")), str(pc.get("same", "—")), str(pc.get("checked", "—")),
		str(m.get("rt_mean", "—")), str(m.get("device", "—"))]
