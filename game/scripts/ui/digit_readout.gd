## DigitReadout (05_components.md §3): 계속 바뀌는 큰 숫자를 글자마다 고정폭 칸에 그린다.
## Saira 숫자는 폭이 서로 달라(1이 좁다, tnum 없음) 그냥 Label로 쓰면 숫자가 흔들린다.
## 갱신은 초당 10번(02_direction.md §7). 값이 한계를 넘으면 호박색.
@tool
class_name DigitReadout
extends Control

@export var text := "0.00":
	set(v):
		if v != text:
			text = v
			queue_redraw()
@export var font_size := 104
@export var color := Color(0.949, 0.957, 0.945)
@export var type_variation := &"ReadoutXL"

var _cell := 0.0
var _font: Font


func _ready() -> void:
	_font = get_theme_font(&"font", type_variation)
	_cell = 0.0
	for ch in "0123456789":
		_cell = maxf(_cell, _font.get_string_size(ch, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x)
	custom_minimum_size = Vector2(_measure("88.88"), font_size * 0.86)


func _measure(s: String) -> float:
	var w := 0.0
	for ch in s:
		w += _cell if ch.is_valid_int() else _font.get_string_size(ch, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x
	return w


func _draw() -> void:
	if _font == null:
		return
	var x := size.x - _measure(text)     # 오른쪽 정렬
	var base := font_size * 0.80
	for ch in text:
		if ch.is_valid_int():
			var w := _font.get_string_size(ch, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x
			draw_string(_font, Vector2(x + (_cell - w) * 0.5, base), ch, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, color)
			x += _cell
		else:
			draw_string(_font, Vector2(x, base), ch, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, color)
			x += _font.get_string_size(ch, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x
