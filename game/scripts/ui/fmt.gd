## 숫자 표기 규칙(02_direction.md §8.1): 랩은 초 아래 세 자리, 1분 넘으면 m:ss.sss, 빼기는 U+2212, 없는 값은 —.
class_name Fmt
extends RefCounted

const DASH := "—"
const MINUS := "−"


static func lap(t: Variant) -> String:
	if t == null or (typeof(t) == TYPE_FLOAT and (is_nan(t) or is_inf(t))):
		return DASH
	var x := float(t)
	var neg := x < 0.0
	x = absf(x)
	var ms := int(round(x * 1000.0))
	var s := ""
	if ms >= 60000:
		s = "%d:%02d.%03d" % [ms / 60000, (ms / 1000) % 60, ms % 1000]
	else:
		s = "%d.%03d" % [ms / 1000, ms % 1000]
	return (MINUS + s) if neg else s


static func signed(x: float, decimals: int = 3) -> String:
	var body := ("%." + str(decimals) + "f") % absf(x)
	if x > 0.0:
		return "+" + body
	if x < 0.0:
		return MINUS + body
	return body


static func speed(v: float) -> String:
	return "%.2f" % absf(v)


static func deg(rad: float) -> String:
	var d := rad_to_deg(rad)
	return (MINUS if d < -0.05 else "") + "%.1f" % absf(d)
