## 전송 비교 장면: 합성 서버(f1sim.server.bench)의 상태 프레임을 받아 차 N대 상자 + 차마다 LiDAR 1081점을 그린다.
## 측정(App.finish가 JSON으로 씀): 화면 fps·프레임 시간, 송신->수신 지연, 왕복 지연, 해독 시간, 적용 시간.
extends Node3D

const LidarView := preload("res://scripts/race/lidar_view.gd")

var cars: Array[MeshInstance3D] = []
var lidars: Array = []


func _ready() -> void:
	var env := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_COLOR
	e.background_color = Tokens.C_BG_VOID
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	e.ambient_light_color = Color(0.8, 0.8, 0.8)
	env.environment = e
	add_child(env)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-55, -35, 0)
	add_child(sun)
	var cam := Camera3D.new()
	cam.projection = Camera3D.PROJECTION_ORTHOGONAL
	cam.size = 16.0
	cam.transform = Transform3D(Basis.from_euler(Vector3(-PI / 2.0, 0, 0)), Vector3(0, 20, 0))
	add_child(cam)
	cam.current = true
	SimLink.state_received.connect(_on_state)
	var ep := App.server_endpoint()
	SimLink.open(String(App.arg("transport", "tcp")), ep[0], ep[1])


func _ensure(n: int) -> void:
	while cars.size() < n:
		var mi := MeshInstance3D.new()
		var bm := BoxMesh.new()
		bm.size = Vector3(0.58, 0.12, 0.31)
		mi.mesh = bm
		add_child(mi)
		cars.append(mi)
		var lv := MeshInstance3D.new()
		lv.set_script(LidarView)
		add_child(lv)
		lv.configure({"n": 1081, "fov": 4.712389, "mount": [0.0, 0.0, 0.11]})
		lidars.append(lv)


func _on_state(st: Dictionary) -> void:
	var t0 := Time.get_ticks_usec()
	var cs: Array = st.get("cars", [])
	var sc: Array = st.get("scans", [])
	_ensure(cs.size())
	for i in cs.size():
		var c: Dictionary = cs[i]
		cars[i].transform = Geo.pose_transform(c.x, c.y, c.yaw, 0.06)
		if i < sc.size():
			lidars[i].set_scan(sc[i])
	App.record_apply_ms((Time.get_ticks_usec() - t0) / 1000.0)
