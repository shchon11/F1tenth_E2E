// 바닥: 어두운 회색 + 1 m 격자(5 m마다 조금 진하게). 02_direction.md §9.1 「경기장, 밤」
shader_type spatial;
render_mode cull_disabled;

uniform vec4 base_color : source_color = vec4(0.075, 0.082, 0.086, 1.0);
uniform vec4 line_color : source_color = vec4(0.115, 0.125, 0.13, 1.0);
uniform vec4 major_color : source_color = vec4(0.15, 0.162, 0.168, 1.0);

varying vec3 world_pos;

void vertex() {
	world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

float grid(vec2 p, float pitch, float width) {
	vec2 g = abs(fract(p / pitch - 0.5) - 0.5) * pitch;
	vec2 fw = fwidth(p);
	vec2 l = 1.0 - smoothstep(vec2(width) - fw, vec2(width) + fw, g);
	return max(l.x, l.y);
}

void fragment() {
	vec2 p = world_pos.xz;
	float minor = grid(p, 1.0, 0.004);
	float major = grid(p, 5.0, 0.008);
	vec3 c = mix(base_color.rgb, line_color.rgb, minor);
	c = mix(c, major_color.rgb, major);
	ALBEDO = c;
	ROUGHNESS = 0.92;
	SPECULAR = 0.2;
}
