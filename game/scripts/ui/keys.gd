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
	"app_quit": [[KEY_ESCAPE], "끝내기", "Esc"],
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
