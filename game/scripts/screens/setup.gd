## 레이스 준비 화면의 바인딩: 서버 목록(catalog)으로 값 줄(이름표 ◀ 값 ▶)을 채우고, 고른 것으로 세션 구성(spec)을 만든다.
## 배치는 scenes/screens/setup.tscn(+ scenes/ui/value_row.tscn), 모양은 Look. 여기는 값만 다룬다.
extends Control

signal go_pressed(spec: Dictionary)
signal back_pressed
signal pick_changed(pick: Dictionary)

const ROW := preload("res://scenes/ui/value_row.tscn")
const ROW_KEYS := [["course", "코스"], ["driver", "드라이버"], ["rules", "규칙"], ["exec", "실행"], ["physics", "차"],
	["duration", "시간"], ["count", "상자"], ["remove", "치움"], ["layout", "배치"]]

var catalog: Dictionary = {}
var pick := {
	"course": "", "driver": "", "exec": "real", "physics": "", "rules": "",
	"duration_s": 90.0, "remove_at_s": 50.0,
	"boxes": {"source": "saved", "index": 0, "seed": 1, "count": 3},
}
var rows := {}
var connected := false
var _last_preview_key := ""


func _ready() -> void:
	%Go.pressed.connect(_on_go)
	%Back.pressed.connect(func(): back_pressed.emit())
	_build_rows()


func _build_rows() -> void:
	if not rows.is_empty():
		return
	for kv in ROW_KEYS:
		var r: ValueRow = ROW.instantiate()
		%Rows.add_child(r)
		r.setup(kv[0], kv[1])
		r.stepped.connect(_on_step)
		r.pressed.connect(func(): pass)
		rows[kv[0]] = r


func set_message(text: String) -> void:
	%Message.text = text


func focus_default() -> void:
	%Go.grab_focus()


# ------------------------------------------------------------------ 목록
func bind_catalog(cat: Dictionary) -> void:
	_build_rows()
	catalog = cat
	var drivers: Array = cat.get("drivers", [])
	if pick.course == "" and cat.get("courses", []).size() > 0:
		pick.course = cat.courses[0].id
	if pick.driver == "":
		for d in drivers:
			if bool(d.get("available", true)):
				pick.driver = d.id
				break
	if pick.physics == "":
		pick.physics = _default_id(cat.get("physics", []))
	if pick.rules == "":
		pick.rules = _default_id(_rules_list())
		_apply_rule_defaults(_find(_rules_list(), pick.rules))
	_refresh()


func _rules_list() -> Array:
	return catalog.get("rules", []).filter(func(r): return bool(r.get("available", true)) and r.get("format") == "qualifier")


func _default_id(items: Array) -> String:
	for it in items:
		if bool(it.get("default", false)):
			return String(it.id)
	return String(items[0].id) if items.size() > 0 else ""


func _find(items: Array, ident: String) -> Dictionary:
	for it in items:
		if String(it.id) == ident:
			return it
	return {}


func _apply_rule_defaults(r: Dictionary) -> void:
	if r.is_empty():
		return
	pick.duration_s = float(r.duration_s.get("v", 90))
	pick.boxes.count = int(r.boxes.get("count", 0))
	pick.remove_at_s = float(r.boxes.get("remove_at_s", 0))


# ------------------------------------------------------------------ 줄 값
func _course() -> Dictionary:
	return _find(catalog.get("courses", []), String(pick.course))


func _rule() -> Dictionary:
	return _find(_rules_list(), String(pick.rules))


func _learned() -> bool:
	return String(_find(catalog.get("drivers", []), String(pick.driver)).get("kind", "learned")) == "learned"


func _remove_choices() -> Array:
	var out: Array = []
	for v in _rule().get("boxes", {}).get("remove_choices", []):
		if float(v) > 0.0 and float(v) < float(pick.duration_s):
			out.append(int(v))
	out.append(-1)
	return out


func _box_counts() -> Array:
	var c := _course()
	if int(c.get("saved_layouts", 0)) == 0 and not bool(c.get("random_layouts", false)):
		return [0]
	return _rule().get("boxes", {}).get("choices", [0]).map(func(v): return int(v))


func _cycle(list: Array, cur: Variant, d: int) -> Variant:
	if list.is_empty():
		return cur
	var i := list.find(cur)
	if i < 0:
		i = 0
	else:
		i = (i + d + list.size()) % list.size()
	return list[i]


func _ids(items: Array, only_available: bool = false) -> Array:
	var out: Array = []
	for it in items:
		if not only_available or bool(it.get("available", true)):
			out.append(String(it.id))
	return out


func _on_step(key: String, d: int) -> void:
	match key:
		"course":
			pick.course = _cycle(_ids(catalog.get("courses", [])), String(pick.course), d)
			pick.boxes.index = 0
		"driver":
			pick.driver = _cycle(_ids(catalog.get("drivers", []), true), String(pick.driver), d)
		"rules":
			pick.rules = _cycle(_ids(_rules_list()), String(pick.rules), d)
			_apply_rule_defaults(_rule())
		"exec":
			pick.exec = _cycle(_ids(catalog.get("execs", [])), String(pick.exec), d)
		"physics":
			pick.physics = _cycle(_ids(catalog.get("physics", [])), String(pick.physics), d)
		"duration":
			var durs: Array = _rule().get("duration_s", {}).get("choices", [pick.duration_s])
			pick.duration_s = float(_cycle(durs.map(func(v): return int(v)), int(pick.duration_s), d))
		"count":
			pick.boxes.count = int(_cycle(_box_counts().map(func(v): return int(v)), int(pick.boxes.count), d))
		"remove":
			var cur := int(pick.remove_at_s) if pick.remove_at_s != null and float(pick.remove_at_s) > 0.0 else -1
			var v := int(_cycle(_remove_choices(), cur, d))
			pick.remove_at_s = float(v) if v > 0 else null
		"layout":
			var c := _course()
			var n := int(c.get("saved_layouts", 0))
			var kinds: Array = []
			if n > 0:
				kinds.append("saved")
			if bool(c.get("random_layouts", false)):
				kinds.append("random")
			# 값 순서: 저장된 배치 1..n, 그다음 무작위 번호(앞뒤로 계속)
			if pick.boxes.source == "saved" and n > 0:
				var i := int(pick.boxes.index) + d
				if i >= n and kinds.has("random"):
					pick.boxes.source = "random"
				elif i < 0 and kinds.has("random"):
					pick.boxes.source = "random"
				else:
					pick.boxes.index = (i + n) % n
			elif kinds.has("random"):
				var sd := int(pick.boxes.seed) + d
				if sd < 1 and kinds.has("saved"):
					pick.boxes.source = "saved"
					pick.boxes.index = n - 1
				else:
					pick.boxes.seed = maxi(1, sd)
	_refresh()


func _refresh() -> void:
	if catalog.is_empty():
		return
	var c := _course()
	var n := int(c.get("saved_layouts", 0))
	var can_random := bool(c.get("random_layouts", false))
	if not _box_counts().has(int(pick.boxes.count)):
		pick.boxes.count = 0
	if pick.boxes.source == "saved" and n == 0:
		pick.boxes.source = "random" if can_random else "none"
	if pick.boxes.source != "saved" and not can_random and n > 0:
		pick.boxes.source = "saved"
	var name_of := func(items: Array, ident: String) -> String:
		return String(_find(items, ident).get("name", "—"))
	rows.course.set_value(name_of.call(catalog.get("courses", []), pick.course))
	rows.driver.set_value(name_of.call(catalog.get("drivers", []), pick.driver))
	rows.rules.set_value(name_of.call(_rules_list(), pick.rules), _rules_list().size() > 1)
	var learned := _learned()
	rows.exec.set_value(name_of.call(catalog.get("execs", []), pick.exec) if learned else "직접 명령", learned)
	rows.physics.set_value(String(_find(catalog.get("physics", []), String(pick.physics)).get("car_label", "—")))
	rows.duration.set_value("%d초" % int(pick.duration_s))
	var boxes_on := int(pick.boxes.count) > 0
	rows.count.set_value("%d개" % int(pick.boxes.count) if boxes_on else "없음", _box_counts().size() > 1)
	var rm = pick.remove_at_s
	var rm_ok: bool = rm != null and float(rm) > 0.0 and float(rm) < float(pick.duration_s)
	rows.remove.set_value(("%d초" % int(rm)) if rm_ok else "치우지 않음", boxes_on)
	if not boxes_on:
		rows.layout.set_value("—", false)
	elif pick.boxes.source == "saved":
		rows.layout.set_value("%d / %d" % [int(pick.boxes.index) + 1, n])
	else:
		rows.layout.set_value("무작위 %d" % int(pick.boxes.seed))
	%Go.disabled = pick.course == "" or pick.driver == "" or not connected
	_request_preview()
	pick_changed.emit(pick)


func _request_preview() -> void:
	var key := JSON.stringify([pick.course, pick.boxes])
	if key == _last_preview_key or not connected:
		return
	_last_preview_key = key
	SimLink.request("preview", {"course": pick.course, "boxes": pick.boxes.duplicate()})


func set_connected(on: bool) -> void:
	connected = on
	if on:
		_last_preview_key = ""
	_refresh()


# ------------------------------------------------------------------ 출발
func spec() -> Dictionary:
	var learned := _learned()
	return {"course": pick.course, "driver": pick.driver, "exec": pick.exec if learned else "real",
		"physics": pick.physics, "rules": pick.rules, "duration_s": pick.duration_s,
		"remove_at_s": pick.remove_at_s if int(pick.boxes.count) > 0 else null, "boxes": pick.boxes.duplicate()}


## 자동 흐름(시험·캡처)용: 바깥에서 고른 값을 넣는다. 화면 줄과 같은 길로 반영된다.
func apply_spec(d: Dictionary) -> void:
	for k in ["course", "driver", "exec", "physics", "rules"]:
		if d.has(k):
			pick[k] = String(d[k])
			if k == "rules":
				_apply_rule_defaults(_rule())
	if d.has("duration_s"):
		pick.duration_s = float(d.duration_s)
	if d.has("remove_at_s"):
		pick.remove_at_s = d.remove_at_s
	if d.has("boxes"):
		for k in d.boxes:
			pick.boxes[k] = d.boxes[k]
	_refresh()


func _on_go() -> void:
	if %Go.disabled:
		return
	go_pressed.emit(spec())
