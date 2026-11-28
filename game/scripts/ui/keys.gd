## 입력 맵(텍스트 표). project.godot의 [input] 직렬화 대신 여기 한 곳에 둔다 — 에이전트가 읽고 고치기 쉽다.
## 키 안내 띠(HintBar)도 이 표에서 글자를 가져온다(03_ia.md §6).
class_name Keys
extends RefCounted

## 동작 이름 -> [키 목록, 안내 글자, 키 딱지 글자]
const ACTIONS := {
	"race_toggle": [[KEY_SPACE], "출발 · 일시정지", "Space"],
	"race_reset": [[KEY_R], "처음부터", "R"],
	"camera_next": [[KEY_C], "카메라", "C"],
	"lidar_toggle": [[KEY_L], "LiDAR", "L"],
	"app_quit": [[KEY_F10], "앱 끝내기", "F10"],
	# M1 레이스 화면(화면에 안내 줄은 두지 않는다: PM 11-28)
	"race_end": [[KEY_ESCAPE], "끝내고 결과 보기", "Esc"],
	"speed_up": [[KEY_EQUAL, KEY_KP_ADD], "배속 올림", "+"],
	"speed_down": [[KEY_MINUS, KEY_KP_SUBTRACT], "배속 내림", "−"],
	"view_laps": [[KEY_K], "바퀴 기록", "K"],
	"view_gauge": [[KEY_G], "계기", "G"],
	"view_scope": [[KEY_O], "LiDAR 위에서", "O"],
	"view_plan": [[KEY_T], "궤적", "T"],
	"ui_go": [[KEY_ENTER, KEY_KP_ENTER], "출발", "Enter"],
}


static func install() -> void:
	for action in ACTIONS:
		if InputMap.has_action(action):
			continue
		InputMap.add_action(action)
		for k in ACTIONS[action][0]:
			var ev := InputEventKey.new()
			ev.physical_keycode = k
			InputMap.action_add_event(action, ev)
