# 자동 생성: game/tools/tokens_to_theme.py <- docs/pm/sim-design-2026-11-28/tokens.css
# 손으로 고치지 말 것. 컴포넌트는 C_* · S_* · T_* 만 쓴다(원색 PALETTE 직접 사용 금지).
class_name Tokens
extends RefCounted

const C_ACCENT := Color(0.776471, 0.945098, 0.207843, 1)
const C_ACCENT_HOVER := Color(0.862745, 0.976471, 0.478431, 1)
const C_ACCENT_LINE := Color(0.776471, 0.945098, 0.207843, 0.45)
const C_ACCENT_PRESS := Color(0.560784, 0.682353, 0.133333, 1)
const C_ACCENT_TINT := Color(0.776471, 0.945098, 0.207843, 0.12)
const C_BG_APP := Color(0.0470588, 0.0509804, 0.054902, 1)
const C_BG_HOVER := Color(0.113725, 0.12549, 0.137255, 1)
const C_BG_OVERLAY := Color(0.0352941, 0.0392157, 0.0431373, 0.82)
const C_BG_PANEL := Color(0.0666667, 0.0745098, 0.0823529, 1)
const C_BG_RAISED := Color(0.0862745, 0.0980392, 0.105882, 1)
const C_BG_VOID := Color(0.027451, 0.0313725, 0.0313725, 1)
const C_BG_WELL := Color(0.0352941, 0.0392157, 0.0431373, 1)
const C_CHART_BAND := Color(0.776471, 0.945098, 0.207843, 0.14)
const C_CHART_GRID := Color(0.94902, 0.956863, 0.945098, 0.07)
const C_DANGER := Color(1, 0.301961, 0.239216, 1)
const C_DANGER_TINT := Color(1, 0.301961, 0.239216, 0.14)
const C_ENT_BOX := Color(0.788235, 0.635294, 0.419608, 1)
const C_ENT_CONTACT := Color(1, 0.301961, 0.239216, 1)
const C_ENT_DUCT := Color(0.764706, 0.784314, 0.796078, 1)
const C_ENT_EGO := Color(0.94902, 0.956863, 0.945098, 1)
const C_ENT_LIDAR := Color(0.498039, 0.890196, 0.94902, 1)
const C_ENT_PLAN := Color(0.776471, 0.945098, 0.207843, 1)
const C_ENT_RIVAL := Color(1, 0.435294, 0.709804, 1)
const C_FOCUS := Color(0.862745, 0.976471, 0.478431, 1)
const C_LINE := Color(0.94902, 0.956863, 0.945098, 0.09)
const C_LINE_INK := Color(0.94902, 0.956863, 0.945098, 1)
const C_LINE_STRONG := Color(0.94902, 0.956863, 0.945098, 0.2)
const C_OK := Color(0.180392, 0.862745, 0.54902, 1)
const C_OK_TINT := Color(0.180392, 0.862745, 0.54902, 0.12)
const C_SCRIM := Color(0.0156863, 0.0196078, 0.0196078, 0.66)
const C_SERIES_1 := Color(0.776471, 0.945098, 0.207843, 1)
const C_SERIES_2 := Color(0.788235, 0.807843, 0.784314, 1)
const C_SERIES_3 := Color(0.705882, 0.423529, 1, 1)
const C_SERIES_4 := Color(0.498039, 0.890196, 0.94902, 1)
const C_SERIES_5 := Color(1, 0.721569, 0.121569, 1)
const C_SERIES_6 := Color(1, 0.435294, 0.709804, 1)
const C_SRC_ASSUMED := Color(1, 0.721569, 0.121569, 1)
const C_SRC_ESTIMATED := Color(0.94902, 0.956863, 0.945098, 1)
const C_SRC_LOWER := Color(1, 0.721569, 0.121569, 1)
const C_SRC_MEASURED := Color(0.94902, 0.956863, 0.945098, 1)
const C_SRC_SPEC := Color(0.788235, 0.807843, 0.784314, 1)
const C_TEXT := Color(0.94902, 0.956863, 0.945098, 1)
const C_TEXT_2 := Color(0.788235, 0.807843, 0.784314, 1)
const C_TEXT_3 := Color(0.658824, 0.686275, 0.662745, 1)
const C_TEXT_4 := Color(0.517647, 0.545098, 0.52549, 1)
const C_TEXT_DISABLED := Color(0.278431, 0.301961, 0.286275, 1)
const C_TEXT_ON_ACCENT := Color(0.0431373, 0.0470588, 0.0470588, 1)
const C_TIME_BEST := Color(0.705882, 0.423529, 1, 1)
const C_TIME_OWN := Color(0.180392, 0.862745, 0.54902, 1)
const C_TIME_SLOWER := Color(1, 0.721569, 0.121569, 1)
const C_WARN := Color(1, 0.721569, 0.121569, 1)
const C_WARN_TINT := Color(1, 0.721569, 0.121569, 0.12)

const CHAMFER := 10
const CONTROL_H := 32
const CONTROL_H_L := 48
const GUTTER := 24
const HINT_H := 36
const LINE_W := 1
const LINE_W_HEAD := 2
const NAV_H := 56
const PAGE_MARGIN := 32
const RADIUS_0 := 0
const RADIUS_1 := 2
const ROW_H := 36
const ROW_H_DENSE := 28
const S_1 := 4
const S_2 := 8
const S_3 := 12
const S_4 := 16
const S_5 := 24
const S_6 := 32
const S_7 := 48
const S_8 := 64
const TICK := 10
const U := 4
const DUR_1_MS := 90
const DUR_2_MS := 160
const DUR_3_MS := 280
const DUR_4_MS := 600

## 글자 단계: [테마 형 변형 이름, 글꼴 묶음, 굵기, 크기 px, 행간 px]
const TYPE := {
	"t-readout-xl": ["ReadoutXL", "font-display", 600, 104, 93.6],
	"t-readout-l": ["ReadoutL", "font-display", 600, 56, 56],
	"t-readout-m": ["ReadoutM", "font-display", 600, 34, 34],
	"t-readout-s": ["ReadoutS", "font-display", 600, 22, 22],
	"t-title-l": ["TitleL", "font-ui", 700, 28, 36],
	"t-title-m": ["TitleM", "font-ui", 600, 20, 28],
	"t-title-s": ["TitleS", "font-ui", 600, 16, 24],
	"t-body": ["Body", "font-ui", 400, 15, 24],
	"t-body-s": ["BodyS", "font-ui", 400, 13, 20],
	"t-label": ["Caps", "font-label", 600, 12, 16],
	"t-micro": ["Micro", "font-label", 600, 11, 14],
	"t-data": ["Data", "font-data", 500, 14, 20],
	"t-data-s": ["DataS", "font-data", 500, 12, 16],
}

const PALETTE := {
	"p-amber-500": Color(1, 0.721569, 0.121569, 1),
	"p-box-400": Color(0.788235, 0.635294, 0.419608, 1),
	"p-carbon-500": Color(0.278431, 0.301961, 0.286275, 1),
	"p-carbon-600": Color(0.207843, 0.227451, 0.243137, 1),
	"p-carbon-700": Color(0.14902, 0.164706, 0.176471, 1),
	"p-carbon-750": Color(0.113725, 0.12549, 0.137255, 1),
	"p-carbon-800": Color(0.0862745, 0.0980392, 0.105882, 1),
	"p-carbon-850": Color(0.0666667, 0.0745098, 0.0823529, 1),
	"p-carbon-900": Color(0.0470588, 0.0509804, 0.054902, 1),
	"p-carbon-950": Color(0.027451, 0.0313725, 0.0313725, 1),
	"p-cyan-400": Color(0.498039, 0.890196, 0.94902, 1),
	"p-duct-300": Color(0.764706, 0.784314, 0.796078, 1),
	"p-duct-600": Color(0.352941, 0.372549, 0.388235, 1),
	"p-green-500": Color(0.180392, 0.862745, 0.54902, 1),
	"p-ink-100": Color(0.94902, 0.956863, 0.945098, 1),
	"p-ink-200": Color(0.788235, 0.807843, 0.784314, 1),
	"p-ink-300": Color(0.658824, 0.686275, 0.662745, 1),
	"p-ink-400": Color(0.517647, 0.545098, 0.52549, 1),
	"p-lime-300": Color(0.862745, 0.976471, 0.478431, 1),
	"p-lime-500": Color(0.776471, 0.945098, 0.207843, 1),
	"p-lime-700": Color(0.560784, 0.682353, 0.133333, 1),
	"p-pink-400": Color(1, 0.435294, 0.709804, 1),
	"p-purple-500": Color(0.705882, 0.423529, 1, 1),
	"p-red-500": Color(1, 0.301961, 0.239216, 1),
}
