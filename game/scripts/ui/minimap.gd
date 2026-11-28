## 미니맵: 코스 윤곽(차로 경계), 계시선, 상자, 차, 접촉 위치. 길쭉한 코스는 판 모양에 맞게 90° 돌려 그린다.
## 색은 Look에서만 가져온다.
class_name Minimap
extends Control

var outline: Array = []          # [{xy: PackedFloat32Array, closed}]
var bounds := Rect2(0, 0, 1, 1)
var gates: Array = []            # [{k, x, y, yaw, wl, wr}]
var boxes: Array = []            # [{x, y, yaw, w, d}] 또는 자세 목록
var boxes_on := true
var car := Vector3.ZERO          # x, y, yaw
var has_car := false
var contacts: Array = []         # [{x, y, hard}]
var _rot := false
var _scale := 1.0
var _off := Vector2.ZERO


func set_course(p_outline: Array, p_bounds: Array) -> void:
	outline = p_outline
	if p_bounds.size() == 4:
		bounds = Rect2(p_bounds[0], p_bounds[1], p_bounds[2] - p_bounds[0], p_bounds[3] - p_bounds[1])
	queue_redraw()


func _fit() -> void:
	var r := Rect2(Vector2.ZERO, size).grow(-10.0)
	var bw := bounds.size.x
	var bh := bounds.size.y
	_rot = (bh > bw) != (r.size.y > r.size.x)
	var w := bh if _rot else bw
	var h := bw if _rot else bh
	_scale = minf(r.size.x / maxf(w, 1e-3), r.size.y / maxf(h, 1e-3))
	_off = r.position + (r.size - Vector2(w, h) * _scale) * 0.5


## 시뮬 (x, y) -> 화면. 돌릴 때는 시뮬 y가 화면 오른쪽으로(반시계 90°).
func to_screen(x: float, y: float) -> Vector2:
	var u := x - bounds.position.x
	var v := y - bounds.position.y
	if _rot:
		return _off + Vector2(v, u) * _scale
	return _off + Vector2(u, bounds.size.y - v) * _scale


func _draw() -> void:
	_fit()
	for o in outline:
		var xy: PackedFloat32Array = o.get("xy", PackedFloat32Array())
		var pts := PackedVector2Array()
		for i in xy.size() / 2:
			pts.append(to_screen(xy[2 * i], xy[2 * i + 1]))
		if bool(o.get("closed", false)) and pts.size() > 2:
			pts.append(pts[0])
		if pts.size() >= 2:
			draw_polyline(pts, Look.DIM, 2.5, true)
	for g in gates:
		var yaw := float(g.yaw)
		var n := Vector2(-sin(yaw), cos(yaw))
		var c := Vector2(float(g.x), float(g.y))
		var a := c + n * float(g.get("wl", 1.0))
		var b := c - n * float(g.get("wr", 1.0))
		draw_line(to_screen(a.x, a.y), to_screen(b.x, b.y), Look.WHITE if int(g.k) == 0 else Look.FAINT,
			3.0 if int(g.k) == 0 else 1.0)
	if boxes_on:
		for bx in boxes:
			var p := to_screen(float(bx.x), float(bx.y))
			var s := maxf(5.0, 0.32 * _scale)
			draw_rect(Rect2(p - Vector2(s, s) * 0.5, Vector2(s, s)), Look.ENT_BOX)
	for c in contacts:
		var p := to_screen(float(c.x), float(c.y))
		draw_arc(p, 9.0 if bool(c.get("hard", false)) else 6.0, 0.0, TAU, 20, Look.ENT_CONTACT, 2.5, true)
	if has_car:
		var p := to_screen(car.x, car.y)
		var yaw := car.z
		var f := to_screen(car.x + cos(yaw), car.y + sin(yaw)) - p
		f = f.normalized() * 11.0
		var side := Vector2(-f.y, f.x) * 0.55
		draw_colored_polygon(PackedVector2Array([p + f, p - f * 0.6 + side, p - f * 0.6 - side]), Look.ENT_EGO)


func _notification(what: int) -> void:
	if what == NOTIFICATION_RESIZED:
		queue_redraw()
