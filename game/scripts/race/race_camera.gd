## 카메라 두 가지(02_direction.md §9.3): 추격(차 뒤 1.4 m, 높이 0.6 m, 화각 58°) / 위에서(정사영, 코스 전체).
extends Camera3D

var mode := "chase"
var follow: Node3D
var track_rect := Rect2(0, 0, 10, 10)
var _init := false


func set_mode(m: String) -> void:
	mode = m
	_init = false
	if mode == "top":
		projection = PROJECTION_ORTHOGONAL
		var vp := get_viewport().get_visible_rect().size
		var aspect := vp.x / maxf(vp.y, 1.0)
		size = maxf(track_rect.size.y, track_rect.size.x / aspect) * 1.15
		var c := track_rect.get_center()
		transform = Transform3D(Basis.from_euler(Vector3(-PI / 2.0, 0, 0)), Geo.to_godot(c.x, c.y, 30.0))
		near = 0.1
		far = 100.0
	else:
		projection = PROJECTION_PERSPECTIVE
		fov = 58.0
		near = 0.03
		far = 80.0


func _process(dt: float) -> void:
	if mode != "chase" or follow == null:
		return
	var car := follow.global_transform
	var fwd := car.basis.x.normalized()
	var want := car.origin - fwd * 1.4 + Vector3.UP * 0.6
	var look := car.origin + fwd * 1.6 + Vector3.UP * 0.05
	var a := 1.0 if not _init else 1.0 - exp(-dt / 0.12)
	_init = true
	global_position = global_position.lerp(want, a)
	look_at(look, Vector3.UP)
