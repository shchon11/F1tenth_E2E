## 코스 그리기. M1 서버의 track 메시지(`hoses`)면 덕트 호스를 둥근 주름관으로, 아니면(스파이크 서버) 옛 메시 그대로.
## 관의 중심선과 반지름은 서버가 물리·LiDAR와 같은 매끄러운 차로 경계에서 만든다(f1sim/server/course_geometry.py):
## 관의 차로 쪽 면 = 차가 닿고 LiDAR가 맞는 면.
extends Node3D

const FLOOR_GRID := preload("res://scripts/race/floor_grid.gdshader")
const FLOOR_LANE := preload("res://scripts/race/floor_lane.gdshader")
const DUCT := preload("res://scripts/race/duct_hose.gdshader")
const SIDES := 14

var bounds := Rect2()
var length_m := 0.0
var start_xform := Transform3D()
var hose_vertices := 0


func build(track: Dictionary) -> void:
	for c in get_children():
		c.queue_free()
	var b: Array = track.get("bounds", [0, 0, 10, 10])
	bounds = Rect2(b[0], b[1], b[2] - b[0], b[3] - b[1])
	length_m = float(track.get("length_m", 0.0))
	if track.has("hoses"):
		_build_hoses(track.get("hoses", []))
		_build_walls(track.get("walls_poly", []), float(track.get("duct_height", 0.33)))
		_build_floor(track.get("lane", {}))
		_build_gates(track.get("sectors", []))
	else:
		_build_spike(track)


# ------------------------------------------------------------------ 덕트 호스(둥근 관)
func _build_hoses(hoses: Array) -> void:
	var mat := ShaderMaterial.new()
	mat.shader = DUCT
	mat.set_shader_parameter("base_color", Look.ENT_DUCT)
	hose_vertices = 0
	for h in hoses:
		var m := tube_mesh(h.get("xy", PackedFloat32Array()), h.get("r", PackedFloat32Array()), bool(h.get("closed", false)))
		if m == null:
			continue
		m.surface_set_material(0, mat)
		var mi := MeshInstance3D.new()
		mi.name = "hose"
		mi.mesh = m
		add_child(mi)


## 중심선(평평한 xy, 시뮬 좌표)과 반지름 배열 -> 관 메시. 관은 바닥에 놓인다(중심 높이 = 반지름).
## UV.x = 관을 따라 잰 거리[m](주름 셰이더), TANGENT = 관 방향.
func tube_mesh(xy: PackedFloat32Array, r: PackedFloat32Array, closed: bool) -> ArrayMesh:
	var n := xy.size() / 2
	if n < 3:
		return null
	var verts := PackedVector3Array()
	var norms := PackedVector3Array()
	var tans := PackedFloat32Array()
	var uvs := PackedVector2Array()
	var idx := PackedInt32Array()
	var ring := SIDES + 1
	verts.resize(n * ring)
	norms.resize(n * ring)
	uvs.resize(n * ring)
	tans.resize(n * ring * 4)
	var arc := 0.0
	for i in n:
		var p := Vector2(xy[2 * i], xy[2 * i + 1])
		var ip := (i - 1 + n) % n if closed else maxi(i - 1, 0)
		var inx := (i + 1) % n if closed else mini(i + 1, n - 1)
		var t := Vector2(xy[2 * inx] - xy[2 * ip], xy[2 * inx + 1] - xy[2 * ip + 1]).normalized()
		if i > 0:
			arc += p.distance_to(Vector2(xy[2 * i - 2], xy[2 * i - 1]))
		var rad := float(r[i]) if i < r.size() else 0.15
		var side := Vector3(-t.y, 0.0, -t.x)        # 시뮬 왼쪽 법선 (-t.y, t.x) -> Godot (x, 0, -y)
		var up := Vector3.UP
		var c := Vector3(p.x, rad, -p.y)
		var tg := Vector3(t.x, 0.0, -t.y)
		for k in ring:
			var th := TAU * float(k) / float(SIDES)
			var d := side * cos(th) + up * sin(th)
			var j := i * ring + k
			verts[j] = c + d * rad
			norms[j] = d
			uvs[j] = Vector2(arc, float(k) / float(SIDES))
			tans[4 * j] = tg.x
			tans[4 * j + 1] = tg.y
			tans[4 * j + 2] = tg.z
			tans[4 * j + 3] = 1.0
	var segs := n if closed else n - 1
	for i in segs:
		var a := i * ring
		var bb := ((i + 1) % n) * ring
		for k in SIDES:
			idx.append_array([a + k, bb + k, a + k + 1, a + k + 1, bb + k, bb + k + 1])
	hose_vertices += verts.size()
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_NORMAL] = norms
	arrays[Mesh.ARRAY_TANGENT] = tans
	arrays[Mesh.ARRAY_TEX_UV] = uvs
	arrays[Mesh.ARRAY_INDEX] = idx
	var m := ArrayMesh.new()
	m.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	return m


# ------------------------------------------------------------------ 호스가 아닌 벽: 얇은 판
func _build_walls(walls: Array, height: float) -> void:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Look.ENT_DUCT.darkened(0.45)
	mat.roughness = 0.95
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	for w in walls:
		var xy: PackedFloat32Array = w.get("xy", PackedFloat32Array())
		var n := xy.size() / 2
		if n < 2:
			continue
		var verts := PackedVector3Array()
		var idx := PackedInt32Array()
		for i in n:
			verts.append(Vector3(xy[2 * i], 0.0, -xy[2 * i + 1]))
			verts.append(Vector3(xy[2 * i], height, -xy[2 * i + 1]))
		var segs := n if bool(w.get("closed", false)) else n - 1
		for i in segs:
			var a := 2 * i
			var bb := 2 * ((i + 1) % n)
			idx.append_array([a, bb, a + 1, a + 1, bb, bb + 1])
		var arrays := []
		arrays.resize(Mesh.ARRAY_MAX)
		arrays[Mesh.ARRAY_VERTEX] = verts
		arrays[Mesh.ARRAY_INDEX] = idx
		var st := SurfaceTool.new()
		st.create_from_arrays(arrays, Mesh.PRIMITIVE_TRIANGLES)
		st.generate_normals()
		var mi := MeshInstance3D.new()
		mi.name = "wall"
		mi.mesh = st.commit()
		mi.material_override = mat
		add_child(mi)


# ------------------------------------------------------------------ 바닥(차로 마스크) · 계시선
func _build_floor(lane: Dictionary) -> void:
	var floor_mi := MeshInstance3D.new()
	floor_mi.name = "floor"
	var pm := PlaneMesh.new()
	pm.size = bounds.size + Vector2(14, 14)
	floor_mi.mesh = pm
	var fm := ShaderMaterial.new()
	fm.shader = FLOOR_LANE
	fm.set_shader_parameter("base_color", Look.FLOOR)
	fm.set_shader_parameter("lane_color", Look.FLOOR_LANE)
	var shape: Array = lane.get("shape", [])
	var data: PackedByteArray = lane.get("data", PackedByteArray())
	if shape.size() == 2 and data.size() == int(shape[0]) * int(shape[1]):
		var px := PackedByteArray()
		px.resize(data.size())
		for i in data.size():
			px[i] = 255 if data[i] > 0 else 0
		var img := Image.create_from_data(int(shape[1]), int(shape[0]), false, Image.FORMAT_R8, px)
		fm.set_shader_parameter("lane_mask", ImageTexture.create_from_image(img))
		var o: Array = lane.get("origin", [0, 0])
		var cell := float(lane.get("cell", 0.05))
		fm.set_shader_parameter("mask_origin", Vector2(o[0], o[1]))
		fm.set_shader_parameter("mask_size", Vector2(int(shape[1]) * cell, int(shape[0]) * cell))
		fm.set_shader_parameter("has_mask", 1.0)
	floor_mi.material_override = fm
	var c := bounds.get_center()
	floor_mi.position = Geo.to_godot(c.x, c.y, -0.002)
	add_child(floor_mi)


func _build_gates(gates: Array) -> void:
	for g in gates:
		var k := int(g.get("k", 0))
		var line := MeshInstance3D.new()
		line.name = "start_line" if k == 0 else "sector_%d" % k
		var bm := BoxMesh.new()
		var wl := float(g.get("wl", 1.2))
		var wr := float(g.get("wr", 1.2))
		bm.size = Vector3(0.10 if k == 0 else 0.04, 0.002, wl + wr)
		line.mesh = bm
		var lm := StandardMaterial3D.new()
		lm.albedo_color = Look.TEXT if k == 0 else Color(Look.TEXT, 0.35)
		lm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		lm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		line.material_override = lm
		var xf := Geo.pose_transform(float(g.x), float(g.y), float(g.yaw), 0.002)
		# 선분 가운데: 중심선 점에서 (wl - wr)/2 만큼 왼쪽
		var yaw := float(g.yaw)
		var off := (wl - wr) * 0.5
		xf.origin += Geo.to_godot(-sin(yaw) * off, cos(yaw) * off, 0.0)
		line.transform = xf
		if k == 0:
			start_xform = xf
		add_child(line)


# ------------------------------------------------------------------ 스파이크 서버(옛 메시)
func _build_spike(track: Dictionary) -> void:
	var meshes: Dictionary = track.get("meshes", {})
	var duct_mat := StandardMaterial3D.new()
	duct_mat.albedo_color = Look.ENT_DUCT.darkened(0.4)
	duct_mat.roughness = 0.9
	duct_mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	var wall_mat := StandardMaterial3D.new()
	wall_mat.albedo_color = Color(0.20, 0.215, 0.225)
	wall_mat.roughness = 0.95
	wall_mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	for key in ["ducts", "walls"]:
		var m := Geo.mesh_from(meshes.get(key), duct_mat if key == "ducts" else wall_mat)
		if m:
			var mi := MeshInstance3D.new()
			mi.name = key
			mi.mesh = m
			add_child(mi)
	var floor_mi := MeshInstance3D.new()
	floor_mi.name = "floor"
	var pm := PlaneMesh.new()
	pm.size = bounds.size + Vector2(12, 12)
	floor_mi.mesh = pm
	var fm := ShaderMaterial.new()
	fm.shader = FLOOR_GRID
	floor_mi.material_override = fm
	var c := bounds.get_center()
	floor_mi.position = Geo.to_godot(c.x, c.y, -0.002)
	add_child(floor_mi)
	var st: Dictionary = track.get("start", {})
	if not st.is_empty():
		start_xform = Geo.pose_transform(st.x, st.y, st.yaw, 0.001)
		var line := MeshInstance3D.new()
		line.name = "start_line"
		var bm := BoxMesh.new()
		bm.size = Vector3(0.08, 0.002, 2.4)
		line.mesh = bm
		var lm := StandardMaterial3D.new()
		lm.albedo_color = Look.TEXT
		lm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		line.material_override = lm
		line.transform = start_xform
		add_child(line)
