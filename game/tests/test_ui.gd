extends TestCase


func test_lap_format() -> void:
	eq(Fmt.lap(9.268), "9.268", "1분 안")
	eq(Fmt.lap(62.4184), "1:02.418", "1분 넘음")
	eq(Fmt.lap(59.9996), "1:00.000", "반올림으로 1분")
	eq(Fmt.lap(null), "—", "없는 값")
	eq(Fmt.lap(-0.021), "−0.021", "음수는 U+2212")


func test_signed() -> void:
	eq(Fmt.signed(0.058), "+0.058", "양수 부호")
	eq(Fmt.signed(-0.058), "−0.058", "음수 부호")


func test_tokens_match_css() -> void:
	# tokens.css: --p-lime-500: #c6f135 -> --c-accent
	eq(Tokens.C_ACCENT.to_html(false), "c6f135", "라임 라인")
	eq(Tokens.C_TIME_BEST.to_html(false), "b46cff", "전체 최고 보라")
	near(Tokens.C_BG_OVERLAY.a, 0.82, 1e-3, "HUD 판 불투명도")
	eq(Tokens.NAV_H, 56, "위 띠 높이")


func test_theme_variations() -> void:
	var th: Theme = load("res://theme/tenth_theme.tres")
	check(th != null, "테마를 읽지 못했습니다")
	if th == null:
		return
	eq(th.get_font_size(&"font_size", &"ReadoutXL"), 104, "HUD 속도 글자 크기")
	eq(th.get_type_variation_base(&"ReadoutXL"), &"Label", "형 변형의 기반")
	var f := th.get_font(&"font", &"ReadoutXL")
	check(f != null and f.get_font_name().begins_with("Saira"), "큰 숫자 글꼴은 Saira: %s" % (f.get_font_name() if f else "없음"))
	# 한글은 Plex KR로 떨어져야 한다(fallback)
	check(f != null and f.fallbacks.size() > 0, "Saira에 한글 대체 글꼴이 없습니다")
	var body := th.default_font
	check(body != null and body.has_char("한".unicode_at(0)), "기본 글꼴에 한글이 없습니다")


func test_keys_have_hints() -> void:
	Keys.install()
	for action in Keys.ACTIONS:
		check(InputMap.has_action(action), "입력 동작 %s가 없습니다" % action)
		check(String(Keys.ACTIONS[action][1]) != "", "%s 안내 글자가 비었습니다" % action)


func test_geo_conversion() -> void:
	eq(Geo.to_godot(1.0, 2.0, 3.0), Vector3(1.0, 3.0, -2.0), "시뮬 -> Godot")
	var t := Geo.pose_transform(0.0, 0.0, PI / 2.0)
	var fwd := t.basis.x
	# 시뮬 yaw 90° = 시뮬 +y(왼쪽) = Godot -z
	near(fwd.x, 0.0, 1e-6, "앞 x")
	near(fwd.z, -1.0, 1e-6, "앞 z")
