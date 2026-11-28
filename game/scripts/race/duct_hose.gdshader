// 덕트 호스: 둥근 관 + 주름(관 길이 방향으로 반복되는 골). UV.x = 관을 따라 잰 거리[m], TANGENT = 관 방향.
// 주름 간격·깊이는 실물 알루미늄 덕트 호스 정도(약 3.5 cm). 멀리서는 줄무늬가 깜빡이지 않게 화면 크기로 흐리게 한다.
shader_type spatial;
render_mode cull_disabled;

uniform vec4 base_color : source_color = vec4(0.62, 0.64, 0.66, 1.0);
uniform float period = 0.035;
uniform float depth = 0.28;

void fragment() {
	float u = UV.x / period;
	float fw = fwidth(u);
	float fade = clamp(1.6 - fw * 3.0, 0.0, 1.0);
	float ph = u * 6.2831853;
	float ridge = sin(ph);
	NORMAL = normalize(NORMAL + TANGENT * cos(ph) * depth * fade);
	float shade = mix(1.0, 0.72 + 0.28 * (0.5 + 0.5 * ridge), fade);
	ALBEDO = base_color.rgb * shade;
	ROUGHNESS = 0.62;
	METALLIC = 0.25;
	SPECULAR = 0.45;
}
