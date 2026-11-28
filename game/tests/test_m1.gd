extends TestCase
## M1 화면 시험(서버 없이): 파이썬 서버가 만든 실제 목록·결과(고정 자료)를 각 화면에 넣어 값이 맞게 들어가는지.
## 고정 자료를 다시 만들려면 보고서의 명령(서버 결과 json 복사)을 돈다.

const CATALOG := "res://tests/fixtures/catalog.json"
const RESULT := "res://tests/fixtures/result_quali.json"


func _load_json(path: String) -> Dictionary:
	var d = JSON.parse_string(FileAccess.get_file_as_string(path))
	return d if typeof(d) == TYPE_DICTIONARY else {}


func _mount(path: String) -> Control:
	var inst: Control = (load(path) as PackedScene).instantiate()
	(Engine.get_main_loop() as SceneTree).root.add_child(inst)
	return inst


func _family(f: Font) -> String:
	var base: Font = (f as FontVariation).base_font if f is FontVariation else f
	return String(base.resource_path) if base else ""


func test_look_single_font_family() -> void:
	var th := Look.theme()
	check(th != null, "테마가 없습니다")
	for t in Look.TYPES:
		var f := th.get_font(&"font", t)
		check(f != null and _family(f).contains("Pretendard"), "%s 글꼴이 한 가족(Pretendard)이 아닙니다: %s" % [t, _family(f) if f else "없음"])
	for t in ["Primary", "Row", "MenuItem", "Word"]:
		check(_family(th.get_font(&"font", t)).contains("Pretendard"), "%s 단추 글꼴" % t)
	# 가장 작은 글씨 22px(디자인 A안)
	for t in Look.TYPES:
		check(int(Look.TYPES[t][1]) >= 22, "%s 글씨가 22px보다 작습니다" % t)


func test_lap_color_rules() -> void:
	eq(Look.lap_color(7.5, 7.5), Look.WHITE, "세션 최고 = 흰색")
	eq(Look.lap_color(8.0, 7.5), Look.DIM, "나머지 = 흐린 흰색(강조색 없음)")
	eq(Look.lap_color(null, 7.5), Look.DIM, "없는 값")


func test_setup_builds_spec_from_catalog() -> void:
	var cat := _load_json(CATALOG)
	check(not cat.is_empty(), "목록 고정 자료가 없습니다")
	var s := _mount("res://scenes/screens/setup.tscn")
	s.bind_catalog(cat)
	var spec: Dictionary = s.spec()
	eq(spec.course, "kr26-hall", "첫 코스")
	eq(String(s.rows.course.value_text()), "IFAC 2026", "코스 이름 = 대회 이름")
	eq(String(s.rows.driver.value_text()), "scan31b u475", "드라이버 이름 = 실행 이름 + update")
	eq(spec.rules, "kr-event-quali", "기본 규칙 = 팀이 겪은 국내 형식")
	eq(spec.physics, "default-car-pre-place", "기본 물리 = 새 기본 프로필")
	eq(spec.exec, "real", "기본 실행 방식 = 실차 방식")
	eq(int(spec.boxes.count), 3, "상자 3개")
	s.apply_spec({"driver": "ftg", "duration_s": 60.0, "remove_at_s": 30.0, "boxes": {"source": "saved", "index": 5}})
	spec = s.spec()
	eq(spec.driver, "ftg", "드라이버 고름")
	eq(float(spec.duration_s), 60.0, "세션 길이")
	eq(float(spec.remove_at_s), 30.0, "치우는 시각")
	eq(int(spec.boxes.index), 5, "저장된 배치 번호")
	eq(spec.exec, "real", "기준선은 실행 방식을 고르지 않음")
	s._on_step("driver", 1)
	check(String(s.pick.driver) != "ftg", "값 줄 오른쪽 = 다음 드라이버")
	# 화면 글자에 절대 경로가 없다
	for l in s.find_children("*", "Label", true, false):
		check(not String((l as Label).text).contains("/home"), "글자에 경로: " + (l as Label).text)
	s.queue_free()


func test_result_binds_score_and_tables() -> void:
	var res := _load_json(RESULT)
	check(not res.is_empty(), "결과 고정 자료가 없습니다")
	var r := _mount("res://scenes/screens/result.tscn")
	r.bind(res)
	eq(r.get_node("%Streak").text, str(int(res.score.best_clean_streak)), "무충돌 연속 바퀴")
	eq(r.get_node("%Best").text, Fmt.lap(res.score.best_lap), "최고 랩")
	eq(r.get_node("%NLaps").text, str(int(res.n_laps)), "바퀴 수")
	var grid: GridContainer = r.get_node("%LapTable")
	var nsec: int = res.laps[0].sectors.size()
	eq(grid.get_child_count(), (3 + nsec) * (1 + res.laps.size()), "바퀴 표 칸 수")
	var facts: GridContainer = r.get_node("%Facts")
	var texts := []
	for c in facts.get_children():
		texts.append((c as Label).text)
	check(texts.has("실행") and texts.has(String(res.labels.exec_label)), "실행 방식 표기")
	check(texts.has("차") and texts.has(String(res.labels.get("car", res.labels.physics))), "차(물리 프로필) 표기")
	r.queue_free()


func test_minimap_rotates_tall_course_into_wide_panel() -> void:
	var m := Minimap.new()
	m.size = Vector2(400, 300)
	m.set_course([], [0.0, 0.0, 4.0, 20.0])
	m._fit()
	var a := m.to_screen(0.0, 0.0)
	var b := m.to_screen(0.0, 20.0)
	check(absf(b.x - a.x) > absf(b.y - a.y), "길쭉한 코스는 가로로 돌려 그립니다")
	m.free()
