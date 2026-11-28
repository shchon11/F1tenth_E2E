## 메인 화면 바인딩: 지금 고른 조합(차·코스·드라이버)과 메뉴 단추만. 배치는 scenes/screens/main_menu.tscn.
extends Control

signal run_pressed
signal quit_pressed


func _ready() -> void:
	%Run.pressed.connect(func(): run_pressed.emit())
	%Quit.pressed.connect(func(): quit_pressed.emit())


func bind_now(car: String, course: String, driver: String) -> void:
	%NowCar.text = car
	%NowCourse.text = course
	%NowDriver.text = driver


func set_message(text: String) -> void:
	%Message.text = text


func focus_default() -> void:
	%Run.grab_focus()
