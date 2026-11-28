## 헤드리스 시험 실행기(외부 애드온 없음). 종료 코드 = 실패한 시험 수(0이면 통과).
## godot --headless --path game -s res://tests/run_tests.gd [-- --only=test_msgpack]
extends SceneTree


func _initialize() -> void:
	var only := ""
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--only="):
			only = a.substr(7)
	var files := DirAccess.get_files_at("res://tests")
	var total := 0
	var failed := 0
	var t0 := Time.get_ticks_msec()
	for f in files:
		if not (f.begins_with("test_") and f.ends_with(".gd")) or f == "test_case.gd":
			continue
		if only != "" and not f.begins_with(only):
			continue
		var script: GDScript = load("res://tests/" + f)
		if script == null:
			print("TEST 실패 %s: 스크립트를 읽지 못했습니다" % f)
			failed += 1
			continue
		for m in script.get_script_method_list():
			var name: String = m["name"]
			if not name.begins_with("test_"):
				continue
			var inst = script.new()
			total += 1
			inst.call(name)
			if inst.failures.is_empty():
				print("TEST 통과 %s::%s" % [f.get_basename(), name])
			else:
				failed += 1
				print("TEST 실패 %s::%s" % [f.get_basename(), name])
				for msg in inst.failures:
					print("  - " + msg)
	print("TEST 합계 %d개 중 %d개 실패 (%d ms)" % [total, failed, Time.get_ticks_msec() - t0])
	quit(min(failed, 125))
