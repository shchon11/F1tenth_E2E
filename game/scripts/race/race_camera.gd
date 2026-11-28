## 카메라: 추격(차 뒤 1.4 m, 높이 0.6 m) / 위(정사영, 코스 전체) / 자유(차 둘레를 도는 궤도, 끌어 돌리고 휠로 당김).
## 메뉴용 둘(순환에 안 들어감): 진열(show, 서 있는 차를 앞 옆에서 낮게) / 조감(overview, 코스 전체를 비스듬히 위에서).
extends Camera3D

const MODES := ["chase", "top", "free"]          # 주행 중 순환
const MENU_MODES := ["show", "overview"]
var mode := "chase"
var follow: Node3D
var track_rect := Rect2(0, 0, 10, 10)
var _init := false
var orbit_yaw := 0.9
var orbit_pitch := 0.75
var orbit_dist := 4.5
var _drag := false


func set_mode(m: String) -> void:
	mode = m if (m in MODES or m in MENU_MODES) else "chase"
	_init = false
	if mode == "overview":
		projection = PROJECTION_PERSPECTIVE
		fov = 40.0
		near = 0.1
		far = 200.0
		var c := track_rect.get_center()
		var span := maxf(track_rect.size.x, track_rect.size.y)
		# 화면 오른쪽 넓은 쪽에 코스가 오도록 시점을 왼쪽으로 비켜 둔다
		var eye := Geo.to_godot(c.x - span * 0.10, c.y - span * 0.62, span * 0.78)
		transform = Transform3D(Basis(), eye).looking_at(Geo.to_godot(c.x - span * 0.20, c.y, 0.0), Vector3.UP)
		return
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
		fov = 58.0 if mode == "chase" else 50.0
		near = 0.03
		far = 80.0


func next_mode() -> String:
	set_mode(MODES[(maxi(MODES.find(mode), 0) + 1) % MODES.size()] if mode in MODES else "chase")
	return mode


func _unhandled_input(ev: InputEvent) -> void:
	if mode != "free":
		return
	if ev is InputEventMouseButton:
		var mb := ev as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_LEFT:
			_drag = mb.pressed
		elif mb.button_index == MOUSE_BUTTON_WHEEL_UP:
			orbit_dist = maxf(1.2, orbit_dist * 0.9)
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			orbit_dist = minf(30.0, orbit_dist * 1.1)
	elif ev is InputEventMouseMotion and _drag:
		var mm := ev as InputEventMouseMotion
		orbit_yaw -= mm.relative.x * 0.006
		orbit_pitch = clampf(orbit_pitch + mm.relative.y * 0.004, 0.1, 1.45)


func _process(dt: float) -> void:
	if follow == null or mode == "top" or mode == "overview":
		return
	var car := follow.global_transform
	var want: Vector3
	var look: Vector3
	if mode == "chase":
		var fwd := car.basis.x.normalized()
		want = car.origin - fwd * 1.4 + Vector3.UP * 0.6
		look = car.origin + fwd * 1.6 + Vector3.UP * 0.05
	elif mode == "show":
		var fwd := car.basis.x.normalized()
		var side := car.basis.z.normalized()
		want = car.origin + fwd * 0.95 + side * 0.55 + Vector3.UP * 0.28
		# 차가 화면 오른쪽 가운데에 오도록 바라보는 점을 왼쪽으로
		look = car.origin + Vector3.UP * 0.08 + side * 0.32 - fwd * 0.05
	else:
		var d := Vector3(cos(orbit_yaw) * cos(orbit_pitch), sin(orbit_pitch), -sin(orbit_yaw) * cos(orbit_pitch))
		want = car.origin + d * orbit_dist
		look = car.origin
	var a := 1.0 if not _init else 1.0 - exp(-dt / 0.12)
	_init = true
	global_position = global_position.lerp(want, a)
	look_at(look, Vector3.UP)
