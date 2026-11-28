## LiDAR 점: 서버가 보낸 거리 배열(0 = 반환 없음)과 센서 자세로 점을 만든다. 점 크기는 화면 픽셀 고정.
## 색(02_direction.md §3.5): 벽·덕트 = 옅은 청록, 상자 = 흰색, 다른 차 = 분홍.
extends MeshInstance3D

var n_beams := 1081
var angle_min := -2.356194
var increment := 0.004363
var height := 0.110
var _cos := PackedFloat32Array()
var _sin := PackedFloat32Array()
var _mesh := ArrayMesh.new()
var last_points := 0


func configure(lidar: Dictionary) -> void:
	n_beams = int(lidar.get("n", 1081))
	var fov := float(lidar.get("fov", 4.712389))
	angle_min = float(lidar.get("angle_min", -fov / 2.0))
	increment = fov / float(n_beams - 1)
	height = float(lidar.get("mount", [0, 0, 0.11])[2])
	_cos.resize(n_beams)
	_sin.resize(n_beams)
	for i in n_beams:
		var a := angle_min + i * increment
		_cos[i] = cos(a)
		_sin[i] = sin(a)
	mesh = _mesh
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.vertex_color_use_as_albedo = true
	m.use_point_size = true
	m.point_size = 3.0
	material_override = m
	cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF


func set_scan(scan: Dictionary) -> void:
	var rng: PackedFloat32Array = scan.get("ranges", PackedFloat32Array())
	var hit: PackedByteArray = scan.get("hit", PackedByteArray())
	if rng.size() != n_beams or _cos.size() != n_beams:
		return
	var pose: Array = scan.get("pose", [0, 0, 0])
	var sx := float(pose[0])
	var sy := float(pose[1])
	var cy := cos(float(pose[2]))
	var sny := sin(float(pose[2]))
	var pts := PackedVector3Array()
	var cols := PackedColorArray()
	pts.resize(n_beams)
	cols.resize(n_beams)
	var k := 0
	var c_wall := Tokens.C_ENT_LIDAR
	var c_box := Tokens.C_TEXT
	var c_car := Tokens.C_ENT_RIVAL
	for i in n_beams:
		var r := rng[i]
		if r <= 0.0:
			continue
		var ca := _cos[i] * cy - _sin[i] * sny
		var sa := _sin[i] * cy + _cos[i] * sny
		pts[k] = Vector3(sx + r * ca, height, -(sy + r * sa))
		var h := hit[i] if i < hit.size() else 1
		cols[k] = c_box if h == 2 else (c_car if h == 4 else c_wall)
		k += 1
	last_points = k
	_mesh.clear_surfaces()
	if k == 0:
		return
	pts.resize(k)
	cols.resize(k)
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = pts
	arrays[Mesh.ARRAY_COLOR] = cols
	_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_POINTS, arrays)
