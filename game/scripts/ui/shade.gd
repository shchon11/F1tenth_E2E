## 글 뒤의 어두운 막 하나: 왼쪽(또는 위)에서 옅어진다. 색은 Look에서.
@tool
class_name Shade
extends TextureRect

@export var vertical := false
@export var reach := 0.56          # 막이 사라지는 곳(화면 폭/높이 비율)
@export var strength := 1.0
@export var min_alpha := 0.0       # 끝에 남는 막(표가 많은 화면에서 글이 장면에 묻히지 않게)


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	stretch_mode = TextureRect.STRETCH_SCALE
	var g := Gradient.new()
	var c := Look.VEIL
	g.offsets = PackedFloat32Array([0.0, reach * 0.5, reach])
	g.colors = PackedColorArray([Color(c, c.a * strength), Color(c, maxf(c.a * 0.62 * strength, min_alpha)), Color(c, min_alpha)])
	var gt := GradientTexture2D.new()
	gt.gradient = g
	gt.width = 256
	gt.height = 256
	gt.fill_from = Vector2(0, 0)
	gt.fill_to = Vector2(0, 1) if vertical else Vector2(1, 0)
	texture = gt
