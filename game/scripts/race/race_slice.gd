## 수직 조각: 시뮬 서버에 붙어 코스·차·LiDAR를 3D로, 계측을 HUD로. Space 출발/일시정지, R 처음부터, C 카메라.
extends Node3D

@onready var track_view: Node3D = $Track
@onready var car_view: Node3D = $Car
@onready var lidar_view: MeshInstance3D = $Lidar
@onready var cam: Camera3D = $Camera
@onready var hud: Control = $HUD/RaceHud

var _started := false


func _ready() -> void:
	Keys.install()
	SimLink.track_received.connect(_on_track)
	SimLink.state_received.connect(_on_state)
	SimLink.connected.connect(_on_connected)
	SimLink.disconnected.connect(func(r): print("TENTH_DISCONNECTED ", r))
	cam.follow = car_view
	_set_camera(String(App.arg("camera", "chase")))
	var ep := App.server_endpoint()
	SimLink.open(String(App.arg("transport", "tcp")), ep[0], ep[1])


func _on_connected(info: Dictionary) -> void:
	print("TENTH_CONNECTED ", info.get("server", "?"), " ", info.get("transport", "?"))
	if App.args.has("autostart"):
		SimLink.command("start")


func _on_track(track: Dictionary) -> void:
	track_view.build(track)
	lidar_view.configure(track.get("lidar", {}))
	cam.track_rect = track_view.bounds
	_set_camera(cam.mode)
	var name_map := {"korea_2026_competition": "Korea 2026"}
	hud.title_venue = name_map.get(track.get("name", ""), String(track.get("name", "")))
	print("TENTH_TRACK %s %.1f m" % [track.get("name", "?"), float(track.get("length_m", 0.0))])


func _on_state(st: Dictionary) -> void:
	var t0 := Time.get_ticks_usec()
	var cars: Array = st.get("cars", [])
	if cars.size() > 0:
		car_view.set_state(cars[0])
	var scans: Array = st.get("scans", [])
	if scans.size() > 0 and lidar_view.visible:
		lidar_view.set_scan(scans[0])
	hud.set_state(st)
	App.record_apply_ms((Time.get_ticks_usec() - t0) / 1000.0)


func _unhandled_input(ev: InputEvent) -> void:
	if ev.is_action_pressed("race_toggle"):
		var s := String(SimLink.state.get("status", "ready"))
		SimLink.command("pause" if s == "running" else "start")
	elif ev.is_action_pressed("race_reset"):
		SimLink.command("reset")
	elif ev.is_action_pressed("camera_next"):
		_set_camera("top" if cam.mode == "chase" else "chase")
	elif ev.is_action_pressed("lidar_toggle"):
		lidar_view.visible = not lidar_view.visible
	elif ev.is_action_pressed("app_quit"):
		App.finish()


## 위에서 보기는 정사영이라 거리 안개(7~30 m)를 끈다(켜 두면 30 m 위 카메라에서 코스가 다 가려진다).
func _set_camera(mode: String) -> void:
	cam.set_mode(mode)
	$WorldEnvironment.environment.fog_enabled = mode == "chase"
	if lidar_view.material_override:
		lidar_view.material_override.point_size = 3.0 if mode == "chase" else 2.0
