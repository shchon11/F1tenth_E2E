"""moderngl scene: shaders, geometry builders, instanced car meshes, shadow map, HUD/labels."""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------- matrices (math convention, v' = M v)
def perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    M = np.zeros((4, 4), np.float32)
    M[0, 0] = f / aspect; M[1, 1] = f; M[2, 2] = (far + near) / (near - far); M[2, 3] = 2 * far * near / (near - far); M[3, 2] = -1
    return M

def ortho(l, r, b, t, n, f):
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = 2 / (r - l); M[1, 1] = 2 / (t - b); M[2, 2] = -2 / (f - n)
    M[0, 3] = -(r + l) / (r - l); M[1, 3] = -(t + b) / (t - b); M[2, 3] = -(f + n) / (f - n)
    return M

def look_at(eye, target, up):
    eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f) + 1e-12
    s = np.cross(f, up); s /= np.linalg.norm(s) + 1e-12
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[0, 3], M[1, 3], M[2, 3] = -s @ eye, -u @ eye, f @ eye
    return M

def trans(v):
    M = np.eye(4, dtype=np.float32); M[:3, 3] = v; return M

def rot_x(a):
    c, s = math.cos(a), math.sin(a); M = np.eye(4, dtype=np.float32); M[1, 1], M[1, 2], M[2, 1], M[2, 2] = c, -s, s, c; return M

def rot_y(a):
    c, s = math.cos(a), math.sin(a); M = np.eye(4, dtype=np.float32); M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, s, -s, c; return M

def rot_z(a):
    c, s = math.cos(a), math.sin(a); M = np.eye(4, dtype=np.float32); M[0, 0], M[0, 1], M[1, 0], M[1, 1] = c, -s, s, c; return M

def _u(M):
    return np.ascontiguousarray(M.T.astype(np.float32)).tobytes()

# ---------------------------------------------------------------- shaders
SCENE_VS = """
#version 330
in vec3 in_pos; in vec3 in_nrm; in vec4 in_col;
in vec4 i_c0; in vec4 i_c1; in vec4 i_c2; in vec4 i_c3; in vec4 i_tint;
uniform mat4 u_view; uniform mat4 u_proj; uniform mat4 u_light_vp;
out vec3 v_wpos; out vec3 v_nrm; out vec4 v_col; out vec4 v_lpos; out vec3 v_tint;
void main() {
    mat4 m = mat4(i_c0, i_c1, i_c2, i_c3);
    vec4 wp = m * vec4(in_pos, 1.0);
    v_wpos = wp.xyz; v_nrm = normalize(mat3(m) * in_nrm); v_col = in_col; v_tint = i_tint.rgb;
    v_lpos = u_light_vp * wp;
    gl_Position = u_proj * u_view * wp;
}
"""
SCENE_FS = """
#version 330
uniform vec3 u_eye; uniform vec3 u_light_dir; uniform sampler2DShadow u_shadow;
uniform float u_shininess; uniform float u_spec; uniform float u_stripes; uniform float u_shadow_on; uniform float u_grid;
uniform float u_texel; uniform float u_alpha;
in vec3 v_wpos; in vec3 v_nrm; in vec4 v_col; in vec4 v_lpos; in vec3 v_tint;
out vec4 f_col;
float shadow() {
    vec3 p = v_lpos.xyz / v_lpos.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) return 1.0;
    float s = 0.0;
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
        s += texture(u_shadow, vec3(p.xy + vec2(i, j) * u_texel, p.z - 0.0012));
    return s / 9.0;
}
void main() {
    vec3 n = normalize(v_nrm); if (!gl_FrontFacing) n = -n;
    vec3 base = v_col.rgb * v_tint;
    if (u_stripes > 0.5) base *= 0.74 + 0.26 * sin(v_col.a * 6.2831853 / 0.024);
    if (u_grid > 0.5) {
        vec2 g = abs(fract(v_wpos.xy) - 0.5);
        float l = smoothstep(0.47, 0.5, max(g.x, g.y));
        base = mix(base, base * 1.5, l * 0.5);
    }
    vec3 l = normalize(u_light_dir);
    float nl = max(dot(n, l), 0.0);
    float sh = u_shadow_on > 0.5 ? shadow() : 1.0;
    vec3 hemi = mix(vec3(0.20, 0.21, 0.24), vec3(0.85, 0.9, 1.0), n.z * 0.5 + 0.5);
    vec3 v = normalize(u_eye - v_wpos); vec3 h = normalize(l + v);
    float spec = pow(max(dot(n, h), 0.0), u_shininess) * u_spec;
    vec3 c = base * (hemi * 0.5 + vec3(1.0, 0.97, 0.92) * 1.25 * nl * sh) + vec3(1.0) * spec * nl * sh;
    float d = length(u_eye - v_wpos);
    c = mix(c, vec3(0.055, 0.065, 0.085), clamp((d - 50.0) / 150.0, 0.0, 1.0));
    c = c / (1.0 + c * 0.35);
    f_col = vec4(pow(c, vec3(1.0 / 2.2)), u_alpha);
}
"""
SHADOW_VS = """
#version 330
in vec3 in_pos; in vec4 i_c0; in vec4 i_c1; in vec4 i_c2; in vec4 i_c3;
uniform mat4 u_light_vp;
void main() { mat4 m = mat4(i_c0, i_c1, i_c2, i_c3); gl_Position = u_light_vp * m * vec4(in_pos, 1.0); }
"""
SHADOW_FS = "#version 330\nvoid main() {}\n"
LINE_VS = """
#version 330
in vec3 in_pos; in vec4 in_col; uniform mat4 u_vp; out vec4 v_col;
void main() { v_col = in_col; gl_Position = u_vp * vec4(in_pos, 1.0); }
"""
LINE_FS = "#version 330\nin vec4 v_col; out vec4 f_col; void main() { f_col = v_col; }\n"
POINT_VS = """
#version 330
in vec3 in_pos; in vec4 in_col; uniform mat4 u_vp; uniform float u_size; out vec4 v_col;
void main() { v_col = in_col; gl_Position = u_vp * vec4(in_pos, 1.0); gl_PointSize = clamp(u_size / max(gl_Position.w, 0.1), 2.0, 18.0); }
"""
POINT_FS = """
#version 330
in vec4 v_col; out vec4 f_col;
void main() { vec2 d = gl_PointCoord - 0.5; if (dot(d, d) > 0.25) discard; f_col = v_col; }
"""
HUD_VS = """
#version 330
in vec2 in_uv; uniform vec4 u_rect; out vec2 v_uv;
void main() { v_uv = vec2(in_uv.x, 1.0 - in_uv.y); vec2 p = u_rect.xy + in_uv * u_rect.zw; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }
"""
HUD_FS = "#version 330\nuniform sampler2D u_tex; in vec2 v_uv; out vec4 f_col; void main() { f_col = texture(u_tex, v_uv); }\n"
LABEL_VS = """
#version 330
in vec2 in_uv; in vec3 i_pos; in float i_idx;
uniform mat4 u_view; uniform mat4 u_proj; uniform float u_size; out vec2 v_uv;
void main() {
    vec4 p = u_view * vec4(i_pos, 1.0);
    p.xy += (in_uv - 0.5) * vec2(u_size * 1.6, u_size);
    gl_Position = u_proj * p;
    float col = mod(i_idx, 8.0); float row = floor(i_idx / 8.0);
    v_uv = (vec2(col, row) + vec2(in_uv.x, 1.0 - in_uv.y)) / 8.0;
}
"""
LABEL_FS = "#version 330\nuniform sampler2D u_tex; in vec2 v_uv; out vec4 f_col; void main() { f_col = texture(u_tex, v_uv); if (f_col.a < 0.05) discard; }\n"

MATERIALS = {   # name suffix -> (shininess, specular strength)
    "rubber": (8.0, 0.03), "plastic": (24.0, 0.18), "metal": (60.0, 0.9), "blue": (70.0, 0.9),
    "glass": (90.0, 0.7), "paint": (40.0, 0.35), "pcb": (30.0, 0.2),
}
IDENT_INST = np.concatenate([np.eye(4, dtype=np.float32).T.reshape(-1), np.ones(4, np.float32)])


def _unit_cube():
    """Axis-aligned unit cube centred at the origin with per-face normals -> (verts, normals, indices)."""
    faces = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)), ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
             ((0, -1, 0), (1, 0, 0), (0, 0, 1)), ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0))]
    v, n, idx = [], [], []
    for k, (nn, a, b) in enumerate(faces):
        nn, a, b = np.array(nn, np.float32), np.array(a, np.float32), np.array(b, np.float32)
        for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            v.append(0.5 * (nn + sa * a + sb * b)); n.append(nn)
        idx += [4 * k, 4 * k + 1, 4 * k + 2, 4 * k, 4 * k + 2, 4 * k + 3]
    return np.array(v, np.float32), np.array(n, np.float32), np.array(idx, np.int32)


class Mesh:
    """Static geometry + dynamic instance buffer (mat4 as 4 columns + tint)."""

    def __init__(self, ctx, prog, shadow_prog, pos, nrm, col, idx, max_instances=1, material="plastic",
                 stripes=False, grid=False, cull=True, casts=True):
        self.ctx = ctx
        v = np.hstack([pos.astype(np.float32), nrm.astype(np.float32), col.astype(np.float32)])
        self.vbo = ctx.buffer(np.ascontiguousarray(v).tobytes())
        self.ibo = ctx.buffer(np.ascontiguousarray(idx.astype(np.int32)).tobytes())
        self.inst = ctx.buffer(reserve=max_instances * 20 * 4, dynamic=True)
        self.inst.write(np.tile(IDENT_INST, max_instances).tobytes())
        self.n_inst = 1
        self.vao = ctx.vertex_array(prog, [(self.vbo, "3f 3f 4f", "in_pos", "in_nrm", "in_col"),
                                           (self.inst, "4f 4f 4f 4f 4f/i", "i_c0", "i_c1", "i_c2", "i_c3", "i_tint")], self.ibo)
        self.vao_shadow = ctx.vertex_array(shadow_prog, [(self.vbo, "3f 7x4", "in_pos"),
                                                         (self.inst, "4f 4f 4f 4f 4x4/i", "i_c0", "i_c1", "i_c2", "i_c3")], self.ibo) if shadow_prog else None
        self.shininess, self.spec = MATERIALS.get(material, MATERIALS["plastic"])
        self.stripes, self.grid, self.cull, self.casts = stripes, grid, cull, casts

    def set_instances(self, mats, tints):
        n = mats.shape[0]
        data = np.concatenate([mats.transpose(0, 2, 1).reshape(n, 16), tints.reshape(n, 4)], 1).astype(np.float32)
        self.inst.write(np.ascontiguousarray(data).tobytes()); self.n_inst = n

    def draw(self, prog, shadow=False):
        import moderngl
        if shadow:
            if self.casts and self.vao_shadow is not None:
                self.vao_shadow.render(moderngl.TRIANGLES, instances=self.n_inst)
            return
        prog["u_shininess"].value = self.shininess; prog["u_spec"].value = self.spec
        prog["u_stripes"].value = 1.0 if self.stripes else 0.0; prog["u_grid"].value = 1.0 if self.grid else 0.0
        alpha = getattr(self, "alpha", 1.0)
        if "u_alpha" in prog: prog["u_alpha"].value = alpha
        if self.cull: self.ctx.enable(moderngl.CULL_FACE)
        else: self.ctx.disable(moderngl.CULL_FACE)
        if alpha < 1.0:                                   # translucent: blend, keep depth of what is behind
            self.ctx.enable(moderngl.BLEND); self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.ctx.depth_mask = False
            self.vao.render(moderngl.TRIANGLES, instances=self.n_inst)
            self.ctx.depth_mask = True; self.ctx.disable(moderngl.BLEND)
        else:
            self.vao.render(moderngl.TRIANGLES, instances=self.n_inst)


class Scene:
    def __init__(self, ctx, width, height, headless=False, msaa=4, shadows=True, shadow_size=2048):
        import moderngl
        self.ctx, self.width, self.height, self.headless, self.msaa = ctx, width, height, headless, msaa
        self.prog = ctx.program(vertex_shader=SCENE_VS, fragment_shader=SCENE_FS)
        self.shadow_prog = ctx.program(vertex_shader=SHADOW_VS, fragment_shader=SHADOW_FS) if shadows else None
        self.line_prog = ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
        self.point_prog = ctx.program(vertex_shader=POINT_VS, fragment_shader=POINT_FS)
        self.hud_prog = ctx.program(vertex_shader=HUD_VS, fragment_shader=HUD_FS)
        self.label_prog = ctx.program(vertex_shader=LABEL_VS, fragment_shader=LABEL_FS)
        self.shadows = shadows
        if shadows:
            self.shadow_tex = ctx.depth_texture((shadow_size, shadow_size))
            self.shadow_tex.compare_func = "<="; self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
            self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)
            self.prog["u_texel"].value = 1.0 / shadow_size
        self.light_dir = np.array([-0.45, -0.6, 1.0]); self.light_dir /= np.linalg.norm(self.light_dir)
        self.static: List[Mesh] = []
        self.car_meshes: List[tuple] = []        # (slot, Mesh)
        self.car_pivots: Dict[str, np.ndarray] = {}
        self.lines: Dict[str, tuple] = {}
        self._quad = ctx.buffer(np.array([0, 0, 1, 0, 0, 1, 1, 1], np.float32).tobytes())
        self.hud_vao = ctx.vertex_array(self.hud_prog, [(self._quad, "2f", "in_uv")])
        self.hud_tex = None; self.hud_size = (0, 0)
        self._build_label_atlas()
        self._make_targets()
        ctx.enable(moderngl.DEPTH_TEST); ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    # ---------------------------------------------------------------- render targets
    def _make_targets(self):
        ctx = self.ctx
        if self.headless:
            size = (self.width, self.height)
            self.fbo_ms = ctx.framebuffer(color_attachments=[ctx.renderbuffer(size, samples=self.msaa)],
                                          depth_attachment=ctx.depth_renderbuffer(size, samples=self.msaa)) if self.msaa > 1 else None
            self.fbo = ctx.framebuffer(color_attachments=[ctx.texture(size, 4)], depth_attachment=ctx.depth_renderbuffer(size))
            self.target = self.fbo_ms or self.fbo
        else:
            self.target = ctx.screen

    def resize(self, w, h):
        self.width, self.height = w, h
        if self.headless:
            self._make_targets()
        else:
            self.ctx.viewport = (0, 0, w, h)

    def clear(self):
        self.target.use(); self.ctx.clear(0.055, 0.065, 0.085)

    def screenshot(self, path):
        from PIL import Image
        if self.headless:
            if self.fbo_ms is not None:
                self.ctx.copy_framebuffer(self.fbo, self.fbo_ms)
            data = self.fbo.read(components=3)
        else:
            data = self.ctx.screen.read(components=3)
        Image.frombytes("RGB", (self.width, self.height), data).transpose(Image.FLIP_TOP_BOTTOM).save(path)

    # ---------------------------------------------------------------- geometry builders
    def _add_static(self, pos, nrm, col, idx, **kw):
        m = Mesh(self.ctx, self.prog, self.shadow_prog, pos, nrm, col, idx, 1, **kw)
        self.static.append(m); return m

    def build_floor(self, bounds, margin=3.0):
        x0, y0, x1, y1 = bounds[0] - margin, bounds[1] - margin, bounds[2] + margin, bounds[3] + margin
        pos = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], np.float32)
        nrm = np.tile([0, 0, 1.0], (4, 1)); col = np.tile([0.30, 0.31, 0.34, 0.0], (4, 1))
        self._add_static(pos, nrm, col, np.array([0, 1, 2, 0, 2, 3]), material="rubber", grid=True, casts=False)

    def build_ducts(self, contours, diameter, radial=14):
        R = diameter / 2
        P, N, C, I = [], [], [], []
        base = 0
        for c in contours:
            pts = _smooth_closed(_simplify(np.asarray(c, np.float64), 0.05), 2)
            n = len(pts)
            if n < 4: continue
            tang = np.roll(pts, -1, 0) - np.roll(pts, 1, 0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
            seg = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1); u = np.concatenate([[0], np.cumsum(seg)[:-1]])
            n1 = np.stack([-tang[:, 1], tang[:, 0], np.zeros(n)], 1); n2 = np.array([0, 0, 1.0])
            ang = np.linspace(0, 2 * math.pi, radial, endpoint=False)
            ring_n = (np.cos(ang)[None, :, None] * n1[:, None, :] + np.sin(ang)[None, :, None] * n2[None, None, :])   # (n, radial, 3)
            centers = np.concatenate([pts, np.full((n, 1), R)], 1)
            vp = centers[:, None, :] + ring_n * R
            col = np.tile([0.86, 0.87, 0.9, 0.0], (n, radial, 1)); col[..., 3] = u[:, None]
            P.append(vp.reshape(-1, 3)); N.append(ring_n.reshape(-1, 3)); C.append(col.reshape(-1, 4))
            i = np.arange(n)[:, None] * radial + np.arange(radial)[None, :]
            i_next_ring = (np.roll(np.arange(n), -1)[:, None] * radial + np.arange(radial)[None, :])
            i_next_rad = np.arange(n)[:, None] * radial + (np.arange(radial)[None, :] + 1) % radial
            i_next_both = np.roll(np.arange(n), -1)[:, None] * radial + (np.arange(radial)[None, :] + 1) % radial
            tri = np.stack([i, i_next_ring, i_next_rad, i_next_rad, i_next_ring, i_next_both], -1).reshape(-1, 3)
            I.append(tri + base); base += n * radial
        if P:
            self._add_static(np.vstack(P), np.vstack(N), np.vstack(C), np.vstack(I).reshape(-1), material="metal", stripes=True)

    def build_walls(self, contours, height=1.0, is_solid=None):
        """contours: closed polylines of tall objects. is_solid(xy) -> bool tells whether a point is
        inside a solid object; contours enclosing solid space get a roof (clutter), the room
        boundary (enclosing floor) does not."""
        try:
            import mapbox_earcut as earcut
        except ImportError:
            earcut = None
        P, N, C, I = [], [], [], []
        base = 0
        for c in contours:
            pts = _simplify(np.asarray(c, np.float64), 0.04); n = len(pts)
            if n < 3: continue
            nxt = np.roll(pts, -1, 0)
            for a, b in zip(pts, nxt):
                d = b - a; nrm = np.array([d[1], -d[0], 0.0]); nrm /= np.linalg.norm(nrm) + 1e-9
                quad = np.array([[a[0], a[1], 0], [b[0], b[1], 0], [b[0], b[1], height], [a[0], a[1], height]])
                P.append(quad); N.append(np.tile(nrm, (4, 1))); C.append(np.tile([0.62, 0.63, 0.66, 0.0], (4, 1)))
                I.append(np.array([0, 1, 2, 0, 2, 3]) + base); base += 4
            solid = is_solid(pts.mean(0)) if is_solid is not None else (pts.max(0) - pts.min(0)).max() < 8.0
            if earcut is not None and n >= 3 and solid:
                tri = earcut.triangulate_float32(pts.astype(np.float32), np.array([n], np.uint32))
                top = np.concatenate([pts, np.full((n, 1), height)], 1)
                P.append(top); N.append(np.tile([0, 0, 1.0], (n, 1))); C.append(np.tile([0.66, 0.67, 0.7, 0.0], (n, 1)))
                I.append(tri.astype(np.int64) + base); base += n
        if P:
            self._add_static(np.vstack(P), np.vstack(N), np.vstack(C), np.concatenate([i.reshape(-1) for i in I]), material="plastic", cull=False)

    def add_line(self, name, pts, color, z=0.01):
        pts = np.asarray(pts, np.float64)
        p = np.concatenate([pts[:, :2], np.full((len(pts), 1), z)], 1).astype(np.float32)
        col = np.tile(np.asarray(color, np.float32), (len(pts), 1)) if np.ndim(color) == 1 else np.asarray(color, np.float32)
        vbo = self.ctx.buffer(np.hstack([p, col]).astype(np.float32).tobytes())
        vao = self.ctx.vertex_array(self.line_prog, [(vbo, "3f 4f", "in_pos", "in_col")])
        self.lines[name] = (vbo, vao, len(pts))

    # ---------------------------------------------------------------- car
    def load_car(self, glb_path, max_cars):
        import trimesh
        sc = trimesh.load(glb_path, force="scene")
        parents = sc.graph.transforms.parents
        pivots = {"chassis": np.eye(4, dtype=np.float32)}
        slots = {"chassis": 0, "lidar": 1, "wheel_fl": 2, "wheel_fr": 3, "wheel_rl": 4, "wheel_rr": 5}
        for node in sc.graph.nodes:
            if node in slots and node != "chassis":
                pivots[node] = np.asarray(sc.graph.get(node)[0], np.float32)
        self.car_pivots = pivots
        for node in sc.graph.nodes_geometry:
            M, gname = sc.graph.get(node)
            geom = sc.geometry[gname]
            parent = parents.get(node, "world")
            group = parent if parent in slots else "chassis"
            local = np.linalg.inv(pivots[group]) @ np.asarray(M, np.float64)
            v = trimesh.transform_points(geom.vertices, local).astype(np.float32)
            nrm = (np.asarray(geom.vertex_normals) @ local[:3, :3].T).astype(np.float32)
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
            col = np.asarray(geom.visual.vertex_colors, np.float32)[:, :4] / 255.0
            col[:, 3] = 0.0
            mat = next((k for k in MATERIALS if gname.endswith(k)), "plastic")
            m = Mesh(self.ctx, self.prog, self.shadow_prog, v, nrm, col, np.asarray(geom.faces).reshape(-1), max_cars, material=mat)
            self.car_meshes.append((slots[group], m))
        # the rule-mandated rear detection box: unit cube, scaled per car by its instance matrix
        v, nrm, idx = _unit_cube()
        col = np.tile(np.array([0.62, 0.48, 0.30, 0.0], np.float32), (len(v), 1))          # cardboard
        box = Mesh(self.ctx, self.prog, self.shadow_prog, v, nrm, col, idx, max_cars, material="rubber", casts=False)
        box.alpha = 0.45                                                                   # drawn translucent
        self.car_meshes.append((6, box))
        self.max_cars = max_cars
        self.label_pos = self.ctx.buffer(reserve=max_cars * 16, dynamic=True)
        self.label_vao = self.ctx.vertex_array(self.label_prog, [(self._quad, "2f", "in_uv"), (self.label_pos, "3f 1f/i", "i_pos", "i_idx")])

    def set_car_instances(self, mats, tints, labels=None):
        n = mats.shape[0]
        for slot, m in self.car_meshes:
            m.set_instances(mats[:, slot], tints)
        n_lab = min(n, 64)
        ids = (np.arange(n_lab) if labels is None else np.asarray(labels)[:n_lab]) % 64
        lab = np.concatenate([mats[:n_lab, 0, :3, 3] + np.array([0, 0, 0.42]), ids.astype(np.float32)[:, None]], 1).astype(np.float32)
        self.label_pos.write(np.ascontiguousarray(lab).tobytes()); self.n_labels = n_lab

    # ---------------------------------------------------------------- points / trails
    def alloc_points(self, n):
        self.pts_vbo = self.ctx.buffer(reserve=n * 7 * 4, dynamic=True)
        self.pts_vao = self.ctx.vertex_array(self.point_prog, [(self.pts_vbo, "3f 4f", "in_pos", "in_col")])
        self.n_pts = n

    def set_points(self, pts, cols):
        self.pts_vbo.write(np.ascontiguousarray(np.hstack([pts, cols]).astype(np.float32)).tobytes())

    def alloc_trails(self, max_cars, length):
        self.trail_len = length
        self.trails = np.zeros((max_cars, length, 3), np.float32); self.trail_fill = 0
        self.trail_col = np.tile(np.array([0.45, 0.55, 0.7, 0.55], np.float32), (length, 1))
        self.trail_vbo = self.ctx.buffer(reserve=max_cars * length * 7 * 4, dynamic=True)
        self.trail_vao = self.ctx.vertex_array(self.line_prog, [(self.trail_vbo, "3f 4f", "in_pos", "in_col")])

    def set_plan(self, pts, cols=None, col=(0.2, 1.0, 0.4, 0.95), slot=0):
        """Polyline (K,3) world coordinates with per-vertex colours (K,4) or one colour; slot 0 = the
        plan (drawn thick), slot 1 = the tracker's predicted motion (thin). None clears the slot."""
        n_attr = f"plan{slot}_n"
        if pts is None or len(pts) < 2:
            setattr(self, n_attr, 0); return
        cols = np.tile(np.asarray(col, np.float32), (len(pts), 1)) if cols is None else np.asarray(cols, np.float32)
        data = np.concatenate([np.asarray(pts, np.float32), cols], 1)
        vbo = getattr(self, f"plan{slot}_vbo", None)
        if vbo is None or vbo.size < data.nbytes:
            vbo = self.ctx.buffer(reserve=max(data.nbytes, 64 * 7 * 4), dynamic=True)
            setattr(self, f"plan{slot}_vbo", vbo)
            setattr(self, f"plan{slot}_vao", self.ctx.vertex_array(self.line_prog, [(vbo, "3f 4f", "in_pos", "in_col")]))
        vbo.write(np.ascontiguousarray(data).tobytes()); setattr(self, n_attr, len(pts))

    def push_trail_points(self, x, y):
        n = min(len(x), self.trails.shape[0]); x, y = x[:n], y[:n]
        self.trails[:n] = np.roll(self.trails[:n], -1, axis=1)
        self.trails[:n, -1, 0] = x; self.trails[:n, -1, 1] = y; self.trails[:n, -1, 2] = 0.02
        if self.trail_fill < self.trail_len:
            self.trail_fill += 1
        data = np.concatenate([self.trails[:n], np.broadcast_to(self.trail_col, (n, self.trail_len, 4))], 2)
        self.trail_vbo.write(np.ascontiguousarray(data.astype(np.float32)).tobytes())

    # ---------------------------------------------------------------- hud / labels
    def set_hud(self, lines):
        from PIL import Image, ImageDraw, ImageFont
        font = _font(17)
        w, h = max(470, int(max((font.getlength(t) for t in lines), default=0)) + 28), 24 * len(lines) + 16
        img = Image.new("RGBA", (w, h), (12, 15, 22, 190))
        d = ImageDraw.Draw(img)
        for i, t in enumerate(lines):
            d.text((12, 8 + 24 * i), t, font=font, fill=(235, 238, 245, 255) if i else (255, 255, 255, 255))
        if self.hud_tex is None or self.hud_size != (w, h):
            self.hud_tex = self.ctx.texture((w, h), 4); self.hud_size = (w, h)
        self.hud_tex.write(img.tobytes())

    def set_panel(self, img, slot: int = 0):
        """RGBA PIL image overlay. slot 0: top-right (policy panel), slot 1: bottom-centre (dash)."""
        if not hasattr(self, "panels"): self.panels = {}
        tex, size = self.panels.get(slot, (None, None))
        if tex is None or size != img.size:
            tex = self.ctx.texture(img.size, 4); size = img.size
        tex.write(img.tobytes()); self.panels[slot] = (tex, size)

    def draw_panel(self):
        import moderngl
        if not getattr(self, "panels", None): return
        self.ctx.disable(moderngl.DEPTH_TEST); self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        for slot, (tex, (w, h)) in self.panels.items():
            if slot == 0: x0, y0 = 1 - (w + 12) / self.width, 1 - (h + 12) / self.height
            else: x0, y0 = 0.5 - 0.5 * w / self.width, 12 / self.height
            self.hud_prog["u_rect"].value = (x0, y0, w / self.width, h / self.height)
            tex.use(0); self.hud_prog["u_tex"].value = 0
            self.hud_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST); self.ctx.disable(moderngl.BLEND)

    def _build_label_atlas(self):
        from PIL import Image, ImageDraw
        font = _font(40)
        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
        for i in range(64):
            cx, cy = (i % 8) * 64, (i // 8) * 64
            d.rounded_rectangle((cx + 6, cy + 10, cx + 58, cy + 54), 8, fill=(0, 0, 0, 150))
            d.text((cx + 32, cy + 32), str(i), font=font, fill=(255, 255, 255, 255), anchor="mm")
        self.label_tex = self.ctx.texture((512, 512), 4, img.tobytes()); self.n_labels = 0

    def draw_hud(self):
        import moderngl
        if self.hud_tex is None: return
        self.ctx.disable(moderngl.DEPTH_TEST); self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        w, h = self.hud_size
        self.hud_prog["u_rect"].value = (12 / self.width, 1 - (h + 12) / self.height, w / self.width, h / self.height)
        self.hud_tex.use(0); self.hud_prog["u_tex"].value = 0
        self.hud_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST); self.ctx.disable(moderngl.BLEND)

    # ---------------------------------------------------------------- frame
    def draw_frame(self, view, proj, eye, light_center, show_points=True, show_race=True, show_trails=True, n_cars=1, focus_xy=(0, 0)):
        import moderngl
        ctx = self.ctx
        # shadow pass
        if self.shadows:
            lc = np.asarray(light_center, np.float64)
            lview = look_at(lc + self.light_dir * 30.0, lc, np.array([0, 0, 1.0]))
            lproj = ortho(-14, 14, -14, 14, 1.0, 80.0)
            light_vp = lproj @ lview
            self.shadow_fbo.use(); ctx.viewport = (0, 0, self.shadow_tex.width, self.shadow_tex.height)
            ctx.clear(depth=1.0); ctx.enable(moderngl.CULL_FACE); ctx.cull_face = "front"
            self.shadow_prog["u_light_vp"].write(_u(light_vp))
            for m in self.static: m.draw(self.shadow_prog, shadow=True)
            for _, m in self.car_meshes: m.draw(self.shadow_prog, shadow=True)
            ctx.cull_face = "back"
            self.prog["u_light_vp"].write(_u(light_vp)); self.shadow_tex.use(1); self.prog["u_shadow"].value = 1
        # main pass
        self.target.use(); ctx.viewport = (0, 0, self.width, self.height)
        ctx.clear(0.055, 0.065, 0.085)
        self.prog["u_view"].write(_u(view)); self.prog["u_proj"].write(_u(proj))
        self.prog["u_eye"].value = tuple(float(v) for v in eye); self.prog["u_light_dir"].value = tuple(float(v) for v in self.light_dir)
        self.prog["u_shadow_on"].value = 1.0 if self.shadows else 0.0
        for m in self.static: m.draw(self.prog)
        for _, m in self.car_meshes: m.draw(self.prog)
        ctx.disable(moderngl.CULL_FACE)
        # lines / points (blended)
        vp = proj @ view
        ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.line_prog["u_vp"].write(_u(vp))
        for name, (vbo, vao, n) in self.lines.items():
            if name == "raceline" and not show_race: continue
            vao.render(moderngl.LINE_STRIP, vertices=n)
        if show_trails and self.trail_fill > 1:
            for i in range(min(n_cars, self.trails.shape[0])):
                self.trail_vao.render(moderngl.LINE_STRIP, vertices=self.trail_fill, first=i * self.trail_len + self.trail_len - self.trail_fill)
        for slot, width in ((1, 1.5), (0, 4.0)):
            n_ = getattr(self, f"plan{slot}_n", 0)
            if n_ > 1:
                self.ctx.line_width = width
                getattr(self, f"plan{slot}_vao").render(moderngl.LINE_STRIP, vertices=n_)
        self.ctx.line_width = 1.0
        if show_points:
            self.point_prog["u_vp"].write(_u(vp)); self.point_prog["u_size"].value = 22.0
            self.pts_vao.render(moderngl.POINTS, vertices=self.n_pts)
        # labels
        if self.n_labels:
            self.label_prog["u_view"].write(_u(view)); self.label_prog["u_proj"].write(_u(proj)); self.label_prog["u_size"].value = 0.09
            self.label_tex.use(0); self.label_prog["u_tex"].value = 0
            self.label_vao.render(moderngl.TRIANGLE_STRIP, instances=self.n_labels)
        ctx.disable(moderngl.BLEND)


# ---------------------------------------------------------------- helpers
def _simplify(pts, min_dist):
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) >= min_dist:
            out.append(p)
    if len(out) > 2 and np.linalg.norm(out[-1] - out[0]) < min_dist:
        out.pop()
    return np.asarray(out)

def _smooth_closed(pts, iters=2):
    for _ in range(iters):                        # Chaikin corner cutting on a closed loop
        nxt = np.roll(pts, -1, 0)
        out = np.empty((2 * len(pts), 2))
        out[0::2] = 0.75 * pts + 0.25 * nxt; out[1::2] = 0.25 * pts + 0.75 * nxt
        pts = out
    return pts

_FONT = {}
def _font(size):
    if size not in _FONT:
        from PIL import ImageFont
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if os.path.exists(path):
                _FONT[size] = ImageFont.truetype(path, size); break
        else:
            _FONT[size] = ImageFont.load_default()
    return _FONT[size]
