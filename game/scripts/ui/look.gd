## 화면의 모양은 이 파일 한 곳에서만 온다(M1, 디자인 A안 「장면」 2026-11-29 확정).
## - 색: 장면 + 흰색 + 흐린 흰색. 강조색 없음. 3D 장면 속 개체 색(덕트·상자·LiDAR·궤적)만 따로.
## - 글꼴: Pretendard 한 가족(가변 굵기). 바뀌는 숫자(속도·시간)에만 고정폭 숫자(tnum)를 켠 변형을 쓴다.
## - 크기: 가장 작은 글씨 22px, 값 28~36, 버튼 44, 메뉴·제목 62 이상(1920×1200 기준).
## 장면 파일(scenes/screens/*.tscn)은 배치와 형 변형 이름만, 바인딩 스크립트(scripts/screens/*.gd)는 값만 넣는다.
class_name Look
extends RefCounted

# ------------------------------------------------------------------ 색
const INK := Color(0.063, 0.067, 0.078)            # 흰 바탕 위 글자
const WHITE := Color(0.957, 0.957, 0.945)          # #F4F4F1
const DIM := Color(0.957, 0.957, 0.945, 0.60)      # 흐린 흰색
const FAINT := Color(0.957, 0.957, 0.945, 0.32)    # 비활성
const VEIL := Color(0.035, 0.039, 0.047, 0.88)     # 글 뒤 막(왼쪽 끝)
# 예전 이름(스파이크 HUD·3D 코드가 쓰던 것)은 모두 흰색 계열로
const TEXT := WHITE
const TEXT_2 := DIM
const TEXT_3 := DIM
const LINE := Color(0.957, 0.957, 0.945, 0.18)
# 3D·미니맵 개체(장면의 색)
const ENT_DUCT := Color(0.66, 0.68, 0.70)
const ENT_BOX := Color(0.72, 0.58, 0.38)
const ENT_LIDAR := Color(0.50, 0.89, 0.95)
const ENT_PLAN := Color(0.80, 0.86, 0.55)
const ENT_EGO := WHITE
const ENT_CONTACT := WHITE
const FLOOR := Color(0.075, 0.08, 0.086)
const FLOOR_LANE := Color(0.115, 0.122, 0.13)

# ------------------------------------------------------------------ 글꼴(한 가족)
const FONT_FILE := "res://assets/fonts/PretendardVariable.ttf"

# 형 변형 -> [굵기, 크기, 색, 고정폭 숫자]
const TYPES := {
	"Name": [500, 24, DIM, false],         # 이름표
	"Value": [700, 32, WHITE, false],      # 값
	"ValueNum": [700, 32, WHITE, true],    # 바뀌는 숫자 값
	"Big": [700, 56, WHITE, true],         # 큰 숫자(채점·시간)
	"Huge": [800, 96, WHITE, true],        # 속도
	"Title": [800, 64, WHITE, false],
	"Mark": [800, 30, WHITE, false],
	"Menu": [700, 62, DIM, false],
	"Small": [500, 22, DIM, false],
	"Cell": [600, 26, WHITE, true],         # 표 칸(숫자 고정폭)
}
const RADIUS := 10

static var _theme: Theme
static var _fonts := {}


static func font(weight: int, tnum: bool = false) -> Font:
	var key := "%d_%s" % [weight, tnum]
	if _fonts.has(key):
		return _fonts[key]
	var fv := FontVariation.new()
	fv.base_font = load(FONT_FILE) as Font
	fv.variation_opentype = {TextServerManager.get_primary_interface().name_to_tag("wght"): weight}
	if tnum:
		fv.opentype_features = {TextServerManager.get_primary_interface().name_to_tag("tnum"): 1}
	_fonts[key] = fv
	return fv


static func box(color: Color, border: Color = Color(0, 0, 0, 0), pad_h: int = 36, pad_v: int = 12, width: int = 3) -> StyleBoxFlat:
	var sb := StyleBoxFlat.new()
	sb.bg_color = color
	sb.set_corner_radius_all(RADIUS)
	sb.content_margin_left = pad_h
	sb.content_margin_right = pad_h
	sb.content_margin_top = pad_v
	sb.content_margin_bottom = pad_v
	if border.a > 0.0:
		sb.border_color = border
		sb.set_border_width_all(width)
	return sb


static func _button(th: Theme, t: String, weight: int, size: int, normal: Color, normal_text: Color,
		focus: Color, focus_text: Color, focus_border: Color = Color(0, 0, 0, 0), pad_h: int = 36) -> void:
	var none := Color(0, 0, 0, 0)
	if t != "Button":
		th.set_type_variation(t, "Button")
	th.set_stylebox("normal", t, box(normal, none, pad_h))
	th.set_stylebox("hover", t, box(focus, focus_border, pad_h))
	th.set_stylebox("pressed", t, box(focus, focus_border, pad_h))
	th.set_stylebox("hover_pressed", t, box(focus, focus_border, pad_h))
	th.set_stylebox("focus", t, box(focus, focus_border, pad_h))
	th.set_stylebox("disabled", t, box(normal, none, pad_h))
	th.set_font("font", t, font(weight))
	th.set_font_size("font_size", t, size)
	th.set_color("font_color", t, normal_text)
	for k in ["font_hover_color", "font_pressed_color", "font_hover_pressed_color", "font_focus_color"]:
		th.set_color(k, t, focus_text)
	th.set_color("font_disabled_color", t, FAINT)


static func theme() -> Theme:
	if _theme != null:
		return _theme
	var th := Theme.new()
	th.default_font = font(500)
	th.default_font_size = 24
	th.set_color("font_color", "Label", WHITE)
	for t in TYPES:
		var spec: Array = TYPES[t]
		th.set_type_variation(t, "Label")
		th.set_font("font", t, font(spec[0], spec[3]))
		th.set_font_size("font_size", t, spec[1])
		th.set_color("font_color", t, spec[2])
	var none := Color(0, 0, 0, 0)
	# 메인 메뉴 낱말: 초점 = 흰색으로 채운 띠
	_button(th, "MenuItem", 700, 62, none, DIM, WHITE, INK)
	# 값 줄: 초점 = 밝은 띠 + 흰 테
	_button(th, "Row", 600, 28, none, DIM, Color(WHITE, 0.16), WHITE, WHITE)
	# 주 행동(출발): 늘 흰색으로 채운 가장 큰 버튼
	_button(th, "Primary", 800, 44, WHITE, INK, WHITE, INK, INK)
	th.set_stylebox("focus", "Primary", box(WHITE, Color(INK, 0.7), 36, 12, 3))
	# 낱말 버튼(뒤로·설정·나가기 같은 덜 중요한 것)
	_button(th, "Word", 600, 28, none, DIM, Color(WHITE, 0.10), WHITE, none, 18)
	# 줄 안의 화살표
	_button(th, "Arrow", 700, 30, none, WHITE, Color(WHITE, 0.12), WHITE, none, 12)
	_button(th, "Button", 600, 28, none, DIM, Color(WHITE, 0.12), WHITE)
	_theme = th
	return th


## 기록 표시: 세션 최고만 흰색, 나머지는 흐린 흰색(강조색 없음).
static func lap_color(t: Variant, best: Variant, _prev_best: Variant = null) -> Color:
	if t == null:
		return DIM
	if best != null and absf(float(t) - float(best)) < 1e-6:
		return WHITE
	return DIM
