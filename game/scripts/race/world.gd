## 주행 장면(3D): 코스·상자·궤적·차·LiDAR 점·카메라. 데이터를 받는 입구는 build/begin/apply 셋뿐.
extends Node3D

@onready var track_view: Node3D = $Track
@onready var boxes_view: Node3D = $Boxes
@onready var plan_view: MeshInstance3D = $Plan
@onready var car_view: Node3D = $Car
@onready var lidar_view: MeshInstance3D = $Lidar
@onready var cam: Camera3D = $Camera

var track: Dictionary = {}


func _ready() -> void:
	cam.follow = car_view
	cam.set_mode("chase")


func build(p_track: Dictionary) -> void:
	track = p_track
	preview_course = ""
	track_view.build(p_track)
	lidar_view.configure(p_track.get("lidar", {}))
	cam.track_rect = track_view.bounds
	boxes_view.setup(p_track.get("boxes", []))
	set_camera(cam.mode)
	print("TENTH_TRACK %s %.1f m hoses=%d hose_vertices=%d" % [p_track.get("name", "?"), float(p_track.get("length_m", 0.0)),
		p_track.get("hoses", []).size(), track_view.hose_vertices])


## 메뉴·준비 화면의 장면: 세션 없이 서버 미리 보기(코스 기하 + 상자 + 출발 자세)로 그린다.
var preview_course := ""


func build_preview(pv: Dictionary) -> void:
	if String(pv.get("course", "")) != preview_course or track.is_empty():
		preview_course = String(pv.get("course", ""))
		track = pv
		track_view.build(pv)
		cam.track_rect = track_view.bounds
	boxes_view.setup(pv.get("boxes", []))
	boxes_view.set_poses([], true)
	var st: Dictionary = pv.get("start", {})
	if not st.is_empty():
		car_view._has_target = false
		car_view.set_state({"x": st.x, "y": st.y, "yaw": st.yaw, "steer": 0.0})
	lidar_view._mesh.clear_surfaces()
	plan_view.set_plan(null)
	if cam.mode in ["show", "overview"]:
		set_camera(cam.mode)


func apply_state(st: Dictionary) -> void:
	var cars: Array = st.get("cars", [])
	if cars.size() > 0:
		car_view.set_state(cars[0])
	var scans: Array = st.get("scans", [])
	if scans.size() > 0 and lidar_view.visible:
		lidar_view.set_scan(scans[0])
	if plan_view.visible:
		plan_view.set_plan(st.get("plan"))
	if st.has("boxes"):
		boxes_view.set_poses(st.get("boxes", []), bool(st.get("boxes_on", false)))


func set_camera(mode: String) -> void:
	cam.set_mode(mode)
	# 위에서 보기는 정사영이라 거리 안개를 끈다
	$WorldEnvironment.environment.fog_enabled = mode != "top"
	if lidar_view.material_override:
		lidar_view.material_override.point_size = 3.0 if mode != "top" else 2.0


func next_camera() -> String:
	var m: String = cam.next_mode()
	set_camera(m)
	return m


func set_layer(name: String, on: bool) -> void:
	match name:
		"lidar":
			lidar_view.visible = on
		"plan":
			plan_view.visible = on
			if not on:
				plan_view.set_plan(null)
