## LiDAR를 위에서: 차 기준 부채꼴 안의 점(벽 청록, 상자 흰색). 거리 고리 1 m 간격.
class_name LidarScope
extends Control

var ranges := PackedFloat32Array()
var hit := PackedByteArray()
var angle_min := -2.356194
var fov := 4.712389
var range_view := 4.0


func set_scan(scan: Dictionary, lidar: Dictionary) -> void:
	ranges = scan.get("ranges", PackedFloat32Array())
	hit = scan.get("hit", PackedByteArray())
	fov = float(lidar.get("fov", fov))
	angle_min = float(lidar.get("angle_min", -fov / 2.0))
	queue_redraw()


func _draw() -> void:
	var c := Vector2(size.x * 0.5, size.y * 0.78)
	var sc := minf(size.x * 0.5, size.y * 0.72) / range_view
	for r in range(1, int(range_view) + 1):
		draw_arc(c, r * sc, 0.0, TAU, 48, Look.LINE, 1.0, true)
	var n := ranges.size()
	if n < 2:
		return
	var inc := fov / float(n - 1)
	for i in range(0, n, 2):
		var r := ranges[i]
		if r <= 0.0 or r > range_view:
			continue
		var a := angle_min + i * inc
		# 차 앞 = 화면 위
		var p := c + Vector2(-sin(a), -cos(a)) * r * sc
		var col := Look.TEXT if i < hit.size() and hit[i] == 2 else Look.ENT_LIDAR
		draw_rect(Rect2(p - Vector2(1.5, 1.5), Vector2(3, 3)), col)
	draw_colored_polygon(PackedVector2Array([c + Vector2(0, -9), c + Vector2(6, 6), c + Vector2(-6, 6)]), Look.ENT_EGO)
