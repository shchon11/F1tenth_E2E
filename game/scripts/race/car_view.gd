## 차 한 대: 기존 콘솔과 같은 glb(ROS 좌표: 뒤 차축 중심, x 앞, z 위)를 불러 시뮬 자세(무게중심)에 둔다.
## 40 Hz 상태를 화면 주기로 매끄럽게: 목표 자세로 지수 보간(시간 상수 약 30 ms).
extends Node3D

const CAR_SCENE := "res://assets/models/f1tenth_car.glb"
const LR := 0.17145          # 무게중심 -> 뒤 차축 [m] (params.VehicleParams.lr)

var target := Transform3D()
var steer := 0.0
var _has_target := false
var _wheels := {}            # 이름 -> [노드, 처음 basis]
var model: Node3D


func _ready() -> void:
	var holder := Node3D.new()
	holder.name = "model"
	# ROS(z 위) -> Godot(y 위): x축으로 -90°. 뒤 차축 원점을 무게중심 기준으로 LR만큼 뒤로.
	holder.transform = Transform3D(Basis(Vector3.RIGHT, -PI / 2.0), Vector3(-LR, 0, 0))
	add_child(holder)
	var packed: PackedScene = load(CAR_SCENE)
	if packed:
		model = packed.instantiate()
		holder.add_child(model)
		_fix_materials(model)
		for wn in ["wheel_fl", "wheel_fr"]:
			var w := model.find_child(wn, true, false) as Node3D
			if w:
				_wheels[wn] = [w, w.transform.basis]
	else:   # 모델이 없으면 임시 상자
		var mi := MeshInstance3D.new()
		var bm := BoxMesh.new()
		bm.size = Vector3(0.58, 0.31, 0.12)
		mi.mesh = bm
		mi.position = Vector3(LR, 0, 0.08)
		holder.add_child(mi)


## glb의 정점 색을 바탕색으로 쓰게 한다(콘솔 렌더러와 같은 모양).
func _fix_materials(n: Node) -> void:
	if n is MeshInstance3D:
		var mi := n as MeshInstance3D
		for s in mi.mesh.get_surface_count():
			var m := mi.mesh.surface_get_material(s)
			if m is StandardMaterial3D:
				var sm := (m as StandardMaterial3D).duplicate() as StandardMaterial3D
				sm.vertex_color_use_as_albedo = true
				sm.vertex_color_is_srgb = true
				mi.set_surface_override_material(s, sm)
	for c in n.get_children():
		_fix_materials(c)


func set_state(car: Dictionary) -> void:
	target = Geo.pose_transform(car.x, car.y, car.yaw)
	steer = float(car.get("steer", 0.0))
	if not _has_target:
		transform = target
		_has_target = true


func _process(dt: float) -> void:
	if not _has_target:
		return
	var a := 1.0 - exp(-dt / 0.03)
	var q := transform.basis.get_rotation_quaternion().slerp(target.basis.get_rotation_quaternion(), a)
	transform = Transform3D(Basis(q), transform.origin.lerp(target.origin, a))
	for wn in _wheels:
		var w: Node3D = _wheels[wn][0]
		w.transform.basis = Basis(Vector3(0, 0, 1), steer) * _wheels[wn][1]
