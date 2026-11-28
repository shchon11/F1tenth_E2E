## 좌표 변환과 메시 만들기. 시뮬(ROS) 좌표: x 앞, y 왼쪽, z 위. Godot: y 위, 오른손.
## 시뮬 (x, y, z) -> Godot (x, z, -y). 시뮬의 yaw(위에서 보아 반시계) = Godot rotation.y.
class_name Geo
extends RefCounted


static func to_godot(x: float, y: float, z: float = 0.0) -> Vector3:
	return Vector3(x, z, -y)


static func pose_transform(x: float, y: float, yaw: float, z: float = 0.0) -> Transform3D:
	return Transform3D(Basis(Vector3.UP, yaw), to_godot(x, y, z))


## 서버 메시 사전 {pos: f4 평평한 xyz, nrm, idx: i4} -> ArrayMesh 한 면.
## 정점 순서를 뒤집는다: 서버 메시는 OpenGL(반시계 = 앞면), Godot은 시계 방향이 앞면.
static func mesh_from(d: Variant, material: Material) -> ArrayMesh:
	if typeof(d) != TYPE_DICTIONARY:
		return null
	var p: PackedFloat32Array = d["pos"]
	var n: PackedFloat32Array = d["nrm"]
	var idx: PackedInt32Array = d["idx"]
	var nv := p.size() / 3
	var verts := PackedVector3Array()
	var norms := PackedVector3Array()
	verts.resize(nv)
	norms.resize(nv)
	for i in nv:
		verts[i] = Vector3(p[3 * i], p[3 * i + 2], -p[3 * i + 1])
		norms[i] = Vector3(n[3 * i], n[3 * i + 2], -n[3 * i + 1])
	var tri := PackedInt32Array()
	tri.resize(idx.size())
	for i in range(0, idx.size() - 2, 3):
		tri[i] = idx[i]
		tri[i + 1] = idx[i + 2]
		tri[i + 2] = idx[i + 1]
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_NORMAL] = norms
	arrays[Mesh.ARRAY_INDEX] = tri
	var m := ArrayMesh.new()
	m.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	m.surface_set_material(0, material)
	return m
