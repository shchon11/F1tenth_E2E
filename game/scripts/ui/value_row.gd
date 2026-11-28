## 값 줄: 이름표 + ◀ 값 ▶. 위아래로 줄을 옮기고(초점), 좌우 키·화살표·휠로 값을 바꾼다. 모양은 테마 형 "Row".
class_name ValueRow
extends Button

signal stepped(key: String, delta: int)

var key := ""
var can_change := true


func setup(p_key: String, name_text: String) -> void:
	key = p_key
	(get_node("H/Name") as Label).text = name_text


func set_value(text: String, p_can_change: bool = true) -> void:
	can_change = p_can_change
	(get_node("H/Value") as Label).text = text
	_arrows()


func value_text() -> String:
	return (get_node("H/Value") as Label).text


func _ready() -> void:
	(get_node("H/Prev") as Button).pressed.connect(func(): _step(-1))
	(get_node("H/Next") as Button).pressed.connect(func(): _step(1))
	focus_entered.connect(_arrows)
	focus_exited.connect(_arrows)
	mouse_entered.connect(_arrows)
	mouse_exited.connect(_arrows)
	_arrows()


func _arrows() -> void:
	var on := can_change and (has_focus() or is_hovered())
	(get_node("H/Prev") as Control).modulate.a = 1.0 if on else 0.0
	(get_node("H/Next") as Control).modulate.a = 1.0 if on else 0.0
	(get_node("H/Name") as Label).add_theme_color_override("font_color", Look.WHITE if has_focus() else Look.DIM)


func _step(d: int) -> void:
	if can_change:
		stepped.emit(key, d)


func _gui_input(ev: InputEvent) -> void:
	if ev.is_action_pressed("ui_left"):
		_step(-1)
		accept_event()
	elif ev.is_action_pressed("ui_right"):
		_step(1)
		accept_event()
	elif ev is InputEventMouseButton and (ev as InputEventMouseButton).pressed:
		var mb := ev as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_WHEEL_UP:
			_step(-1)
			accept_event()
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_step(1)
			accept_event()
