## 드라이버가 낸 궤적: 바닥에 붙은 띠(폭 0.10 m), 멀어질수록 흐려진다. 상태 프레임의 plan (k×3: x, y, 속도).
extends MeshInstance3D

const WIDTH := 0.10
var _im := ImmediateMesh.new()
var points := 0


func _ready() -> void:
	mesh = _im
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.vertex_color_use_as_albedo = true
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	material_override = m
	cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF


func set_plan(plan: Variant) -> void:
	_im.clear_surfaces()
	points = 0
	if not (plan is PackedFloat32Array):
		return
	var p: PackedFloat32Array = plan
	var n := p.size() / 3
	if n < 2:
		return
	points = n
	_im.surface_begin(Mesh.PRIMITIVE_TRIANGLE_STRIP)
	for i in n:
		var a := Vector2(p[3 * i], p[3 * i + 1])
		var j := mini(i + 1, n - 1)
		var k := maxi(i - 1, 0)
		var t := (Vector2(p[3 * j], p[3 * j + 1]) - Vector2(p[3 * k], p[3 * k + 1])).normalized()
		var nrm := Vector2(-t.y, t.x) * WIDTH * 0.5
		var alpha := lerpf(0.95, 0.05, float(i) / float(n - 1))
		var c := Color(Look.ENT_PLAN, alpha)
		_im.surface_set_color(c)
		_im.surface_add_vertex(Geo.to_godot(a.x + nrm.x, a.y + nrm.y, 0.012))
		_im.surface_set_color(c)
		_im.surface_add_vertex(Geo.to_godot(a.x - nrm.x, a.y - nrm.y, 0.012))
	_im.surface_end()
