## 상자: 세션의 상자 목록(크기·이름)을 받고, 상태 프레임의 자세(밀리면 움직임)로 옮긴다. 치운 뒤에는 숨긴다.
extends Node3D

var _boxes: Dictionary = {}     # i -> MeshInstance3D


func setup(boxes: Array) -> void:
	for c in get_children():
		c.queue_free()
	_boxes.clear()
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Look.ENT_BOX
	mat.roughness = 0.85
	for b in boxes:
		var mi := MeshInstance3D.new()
		var bm := BoxMesh.new()
		bm.size = Vector3(float(b.w), float(b.h), float(b.d))
		mi.mesh = bm
		mi.material_override = mat
		mi.set_meta("h", float(b.h))
		add_child(mi)
		_boxes[int(b.i)] = mi
		_place(mi, float(b.x), float(b.y), float(b.yaw))


func _place(mi: MeshInstance3D, x: float, y: float, yaw: float) -> void:
	mi.transform = Transform3D(Basis(Vector3.UP, yaw), Geo.to_godot(x, y, float(mi.get_meta("h")) * 0.5))


## poses: [[i, x, y, yaw], ...] 살아 있는 상자만. 없는 번호는 숨긴다.
func set_poses(poses: Array, boxes_on: bool) -> void:
	var seen := {}
	for p in poses:
		var i := int(p[0])
		seen[i] = true
		if _boxes.has(i):
			_place(_boxes[i], float(p[1]), float(p[2]), float(p[3]))
	for i in _boxes:
		(_boxes[i] as Node3D).visible = boxes_on and (seen.has(i) or poses.is_empty())


func count_visible() -> int:
	var n := 0
	for i in _boxes:
		if (_boxes[i] as Node3D).visible:
			n += 1
	return n
