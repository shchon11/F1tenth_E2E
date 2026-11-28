## 코스 그리기: 서버 track 메시지의 덕트 관·벽 메시(콘솔 워커와 같은 배열)를 그대로 올린다.
extends Node3D

const FLOOR_SHADER := preload("res://scripts/race/floor_grid.gdshader")

var bounds := Rect2()
var length_m := 0.0
var start_xform := Transform3D()


func build(track: Dictionary) -> void:
	for c in get_children():
		c.queue_free()
	var meshes: Dictionary = track.get("meshes", {})
	var duct_mat := StandardMaterial3D.new()
	duct_mat.albedo_color = Tokens.C_ENT_DUCT.darkened(0.55)
	duct_mat.roughness = 0.9
	duct_mat.metallic_specular = 0.25
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
	var b: Array = track.get("bounds", [0, 0, 10, 10])
	bounds = Rect2(b[0], b[1], b[2] - b[0], b[3] - b[1])
	length_m = float(track.get("length_m", 0.0))
	# 바닥: 코스 둘레 + 여유 6 m 판 하나(격자 셰이더)
	var floor_mi := MeshInstance3D.new()
	floor_mi.name = "floor"
	var pm := PlaneMesh.new()
	pm.size = bounds.size + Vector2(12, 12)
	floor_mi.mesh = pm
	var fm := ShaderMaterial.new()
	fm.shader = FLOOR_SHADER
	floor_mi.material_override = fm
	var c := bounds.get_center()
	floor_mi.position = Geo.to_godot(c.x, c.y, -0.002)
	add_child(floor_mi)
	# 출발선: 흰 띠 하나(중심선 법선 방향, 폭 2.4 m)
	var st: Dictionary = track.get("start", {})
	if not st.is_empty():
		start_xform = Geo.pose_transform(st.x, st.y, st.yaw, 0.001)
		var line := MeshInstance3D.new()
		line.name = "start_line"
		var bm := BoxMesh.new()
		bm.size = Vector3(0.08, 0.002, 2.4)
		line.mesh = bm
		var lm := StandardMaterial3D.new()
		lm.albedo_color = Tokens.C_TEXT
		lm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		line.material_override = lm
		line.transform = start_xform
		add_child(line)
