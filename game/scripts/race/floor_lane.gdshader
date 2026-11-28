// 바닥: 어두운 회색 + 1 m 격자, 달릴 수 있는 면(서버가 보낸 차로 마스크)은 한 단 밝게.
shader_type spatial;
render_mode cull_disabled;

uniform vec4 base_color : source_color = vec4(0.075, 0.082, 0.086, 1.0);
uniform vec4 lane_color : source_color = vec4(0.115, 0.124, 0.13, 1.0);
uniform vec4 line_color : source_color = vec4(0.15, 0.16, 0.165, 1.0);
uniform sampler2D lane_mask : filter_linear;
uniform vec2 mask_origin = vec2(0.0);   // 시뮬 (x, y)
uniform vec2 mask_size = vec2(1.0);     // [m]
uniform float has_mask = 0.0;

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
	vec2 sim = vec2(world_pos.x, -world_pos.z);
	vec2 uv = (sim - mask_origin) / mask_size;
	float lane = has_mask * texture(lane_mask, uv).r;
	vec3 c = mix(base_color.rgb, lane_color.rgb, lane);
	c = mix(c, line_color.rgb, grid(sim, 1.0, 0.004) * 0.6);
	ALBEDO = c;
	ROUGHNESS = 0.9;
	SPECULAR = 0.2;
}
