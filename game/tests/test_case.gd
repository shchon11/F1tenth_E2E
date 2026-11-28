## 시험 기반 클래스. 시험 파일 = res://tests/test_*.gd, 시험 함수 = test_* (인자 없음).
## 실행: python3 game/tools/tenth.py test  (또는 godot --headless --path game -s res://tests/run_tests.gd)
class_name TestCase
extends RefCounted

var failures: Array[String] = []


func check(cond: bool, msg: String) -> void:
	if not cond:
		failures.append(msg)


func eq(got: Variant, want: Variant, msg: String = "") -> void:
	if typeof(got) != typeof(want) or got != want:
		failures.append("%s: 받은 값 %s (%s), 기대 %s (%s)" % [msg, str(got), type_string(typeof(got)), str(want), type_string(typeof(want))])


func near(got: float, want: float, tol: float, msg: String = "") -> void:
	if absf(got - want) > tol:
		failures.append("%s: 받은 값 %s, 기대 %s ± %s" % [msg, got, want, tol])
