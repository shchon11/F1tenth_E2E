extends TestCase
## 모든 스크립트·장면이 읽히는지(문법·형 오류). `godot --check-only`는 자동 로드(App, SimLink)를 모르고
## 실패해도 종료 코드 0을 돌려줘서 쓸 수 없었다(스파이크에서 확인) — 그래서 시험 실행기 안에서 직접 읽는다.

const ROOTS := ["res://scripts", "res://bench", "res://scenes"]


func _walk(dir: String, out: Array) -> void:
	for f in DirAccess.get_files_at(dir):
		if f.ends_with(".gd") or f.ends_with(".tscn") or f.ends_with(".gdshader"):
			out.append(dir.path_join(f))
	for d in DirAccess.get_directories_at(dir):
		_walk(dir.path_join(d), out)


func test_everything_loads() -> void:
	var files := []
	for r in ROOTS:
		_walk(r, files)
	check(files.size() >= 10, "찾은 파일이 너무 적습니다: %d" % files.size())
	for path in files:
		var res = ResourceLoader.load(path, "", ResourceLoader.CACHE_MODE_IGNORE)
		if res == null:
			failures.append("읽지 못함: " + path)
		elif res is GDScript and not (res as GDScript).can_instantiate():
			failures.append("인스턴스 불가(컴파일 오류): " + path)
		elif res is PackedScene and not (res as PackedScene).can_instantiate():
			failures.append("장면 인스턴스 불가: " + path)
