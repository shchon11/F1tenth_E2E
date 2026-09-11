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
uniform float u_fog0; uniform float u_fog1;
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
    if (u_stripes > 0.5) {
        // Band-limit the hose stripe the way the grid below is band-limited. The period is 24 mm;
        // down a 190 m straight one pixel spans far more than that, so the sine is sampled at a
        // frequency it cannot carry and turns into moire. `fwidth` gives the arc length this pixel
        // covers, and the amplitude is faded out as that approaches a period. It fades to 0.74,
        // which is the mean of the stripe over a period -- so a pixel too small to resolve the
        // banding gets exactly the average it should have had, rather than a stripe of its own.
        float sw = fwidth(v_col.a);
        float amp = 0.26 * (1.0 - smoothstep(0.25 * 0.024, 0.9 * 0.024, sw));
        base *= 0.74 + amp * sin(v_col.a * 6.2831853 / 0.024);
    }
    if (u_grid > 0.5) {
        // Screen-space width, not a fixed 0.03 m one. The ground plate is now sized to the track,
        // and on a 190 m circuit seen whole a 1 m grid line lands well inside a single pixel: which
        // lines survive sampling then depends on where the pixel centre falls, and the plate reads
        // as a field of white dots. `fwidth` gives the metres this pixel spans, so the line can be
        // kept at least a pixel wide and faded out entirely once a whole cell is down to a few
        // pixels -- there is no grid left to show at that point, only aliasing.
        float px = max(fwidth(v_wpos.x), fwidth(v_wpos.y));
        float w = clamp(px, 0.02, 0.35);
        float fade = 1.0 - smoothstep(0.12, 0.45, px);
        vec2 g = abs(fract(v_wpos.xy) - 0.5);
        float l = smoothstep(0.5 - w, 0.5, max(g.x, g.y));
        base = mix(base, base * 1.5, l * 0.5 * fade);
    }
    vec3 l = normalize(u_light_dir);
    float nl = max(dot(n, l), 0.0);
    float sh = u_shadow_on > 0.5 ? shadow() : 1.0;
    vec3 hemi = mix(vec3(0.20, 0.21, 0.24), vec3(0.85, 0.9, 1.0), n.z * 0.5 + 0.5);
    vec3 v = normalize(u_eye - v_wpos); vec3 h = normalize(l + v);
    float spec = pow(max(dot(n, h), 0.0), u_shininess) * u_spec;
    vec3 c = base * (hemi * 0.5 + vec3(1.0, 0.97, 0.92) * 1.25 * nl * sh) + vec3(1.0) * spec * nl * sh;
    float d = length(u_eye - v_wpos);
    // Fog range is a uniform, not a constant. Fixed 50-200 m is only right for a camera a few
    // metres from its subject: pull back far enough to frame a whole circuit and every surface in
    // the scene sits past the far end, so the entire map flattens to the fog colour and the walls
    // vanish. The viewer scales this with the camera's own distance.
    c = mix(c, vec3(0.055, 0.065, 0.085), clamp((d - u_fog0) / max(1.0, u_fog1 - u_fog0), 0.0, 1.0));
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
        # Defaults reproduce the previous hard-coded range exactly, so anything that never calls
        # `set_fog` renders as it did before.
        self.set_fog(50.0, 200.0)
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

    def set_fog(self, start: float, end: float) -> None:
        """Distance at which surfaces begin, and finish, fading into the background.

        The caller owns this because only the caller knows how far away the camera is. A range that
        suits a chase camera three metres behind a car makes an overhead view of a whole circuit a
        blank sheet -- everything in the scene is beyond `end`, so everything ends up the same
        colour and the walls stop being visible at all.
        """
        start = float(start)
        end = max(float(end), start + 1.0)
        self.fog_range = (start, end)
        self.prog["u_fog0"].value = start
        self.prog["u_fog1"].value = end

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
            # `self.target`, not `ctx.screen`. For a GLFW window the two are the same thing, but
            # under Qt the target is the widget's own FBO and framebuffer 0 holds someone else's
            # pixels -- or none at all.
            data = self.target.read(components=3)
        Image.frombytes("RGB", (self.width, self.height), data).transpose(Image.FLIP_TOP_BOTTOM).save(path)

    # ---------------------------------------------------------------- teardown
    def release(self):
        """Free every GL object this Scene created. Idempotent, and safe to call twice.

        Only what we made. `self.target` may be a framebuffer belonging to someone else (a Qt
        widget's default FBO, or `ctx.screen`), and releasing that would delete a surface we do not
        own -- in moderngl 5.12.0 `Framebuffer.release()` does not check the `_is_reference` flag
        that `__del__` honours, so an explicit release on a detected framebuffer really does call
        `glDeleteFramebuffers`.

        Call this while the context is still current. Leaving it to garbage collection means the
        deletes may run after the context is gone, which is undefined at best.
        """
        if getattr(self, "_released", False):
            return
        self._released = True

        def drop(obj):
            if obj is None:
                return
            try:
                obj.release()
            except Exception:
                pass

        self.clear_static()
        for name in list(self.lines):
            self.clear_line(name)
        for _, mesh in self.car_meshes:
            for attr in ("vao", "vao_shadow", "vbo", "ibo", "inst"):
                drop(getattr(mesh, attr, None))
        self.car_meshes = []
        for attr in ("shadow_fbo", "shadow_tex", "hud_tex", "hud_vao", "_quad", "label_tex",
                     "label_pos", "label_vao", "pts_vao", "pts_vbo", "trail_vao", "trail_vbo",
                     "plan0_vao", "plan0_vbo", "plan1_vao", "plan1_vbo"):
            drop(getattr(self, attr, None))
            if hasattr(self, attr):
                setattr(self, attr, None)
        for tex, _size in list(getattr(self, "panels", {}).values()):
            drop(tex)
        self.panels = {}
        for prog in ("prog", "shadow_prog", "line_prog", "point_prog", "hud_prog", "label_prog"):
            drop(getattr(self, prog, None))
            setattr(self, prog, None)
        self.target = None
        self.n_labels = 0

    # ---------------------------------------------------------------- geometry builders
    def _add_static(self, pos, nrm, col, idx, **kw):
        m = Mesh(self.ctx, self.prog, self.shadow_prog, pos, nrm, col, idx, 1, **kw)
        self.static.append(m); return m

    def add_static_mesh(self, pos, nrm, col, idx, **kw):
        """Upload an already-built static mesh. The GL half of `build_ducts` / `build_walls`.

        The split exists so the vertex generation -- which is pure numpy and, on a large map, the
        expensive half -- can run somewhere other than the thread that owns the GL context. The Qt
        console builds these arrays in its simulation worker process and hands them here, so a map
        switch costs the render thread an upload rather than a marching-squares pass."""
        return self._add_static(pos, nrm, col, idx, **kw)

    def clear_static(self):
        """Drop the floor / ducts / walls and free their GPU buffers.

        `build_floor`, `build_ducts` and `build_walls` append, so rebuilding for another track without
        this leaves the previous track's geometry drawn underneath the new one."""
        for m in self.static:
            for attr in ("vao", "vao_shadow", "vbo", "ibo", "inst"):
                obj = getattr(m, attr, None)
                if obj is not None:
                    try:
                        obj.release()
                    except Exception:
                        pass
        self.static = []

    def clear_line(self, name):
        """Remove a named line and free its buffers (add_line overwrites the dict entry but the old
        VAO/VBO would otherwise leak on every track switch)."""
        old = self.lines.pop(name, None)
        if old is not None:
            for obj in old[:2]:
                try:
                    obj.release()
                except Exception:
                    pass

    def build_floor(self, bounds, margin=3.0):
        self._add_static(*floor_mesh_arrays(bounds, margin), material="rubber", grid=True, casts=False)

    def build_ducts(self, contours, diameter, radial=14, occupied=None):
        """occupied(xy (n,2)) -> bool: which side of a contour is solid. See `duct_mesh_arrays`."""
        arrays = duct_mesh_arrays(contours, diameter, radial=radial, occupied=occupied)
        if arrays is not None:
            self._add_static(*arrays, material="metal", stripes=True)

    def build_walls(self, contours, height=1.0, is_solid=None):
        """contours: closed polylines of tall objects. See `wall_mesh_arrays`."""
        arrays = wall_mesh_arrays(contours, height=height, is_solid=is_solid)
        if arrays is not None:
            self._add_static(*arrays, material="plastic", cull=False)

    def add_line(self, name, pts, color, z=0.01):
        pts = np.asarray(pts, np.float64)
        p = np.concatenate([pts[:, :2], np.full((len(pts), 1), z)], 1).astype(np.float32)
        col = np.tile(np.asarray(color, np.float32), (len(pts), 1)) if np.ndim(color) == 1 else np.asarray(color, np.float32)
        vbo = self.ctx.buffer(np.hstack([p, col]).astype(np.float32).tobytes())
        vao = self.ctx.vertex_array(self.line_prog, [(vbo, "3f 4f", "in_pos", "in_col")])
        self.clear_line(name)                                  # free the buffers this entry replaces
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
        """RGBA PIL image overlay. slot 0: top-right (the policy's scan + plan, from above),
        slot 1: bottom-centre (dash), slot 2: top-left (raw activations, --internals)."""
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
            if slot == 0: x0, y0 = 1 - (w + 12) / self.width, 1 - (h + 12) / self.height   # top-right
            elif slot == 2: x0, y0 = 12 / self.width, 1 - (h + 12) / self.height            # top-left
            else: x0, y0 = 0.5 - 0.5 * w / self.width, 12 / self.height                     # bottom-centre
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
        for slot, width in ((1, 2.0), (0, 6.0)):    # this driver reports an aliased range of 1-10 px,
                                                    # so these are honoured rather than clamped to 1
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


# ---------------------------------------------------------------- static mesh builders (no GL)
# These are the vertex-generation halves of Scene.build_ducts / build_walls, split out so they can
# run somewhere other than the thread holding the GL context. On a large map the marching-squares
# contours and the tube generation are the expensive part, and doing them on the render thread is
# what makes a map switch hitch. The Qt console runs them in its simulation worker process and
# hands the arrays to Scene.add_static_mesh, which only uploads.
#
# Each returns (pos, nrm, col, idx) or None when the contours produced nothing drawable.

def _tangent_arclength(pts, win):
    """Unit tangents from a chord spanning +-`win` metres of arc length around a closed contour.

    Resampling by arc length with `np.interp` is what makes this independent of vertex spacing,
    which is the whole point: the estimate should describe the wall, not the simplifier's output.
    """
    seg = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    total = float(u[-1] + seg[-1])
    central = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    central /= np.linalg.norm(central, axis=1, keepdims=True) + 1e-9
    if total <= 2 * win or win <= 0:
        return central
    uu = np.concatenate([u - total, u, u + total])
    xx = np.concatenate([pts, pts, pts])
    ahead = np.stack([np.interp(u + win, uu, xx[:, d]) for d in range(2)], 1)
    behind = np.stack([np.interp(u - win, uu, xx[:, d]) for d in range(2)], 1)
    t = ahead - behind
    n = np.linalg.norm(t, axis=1, keepdims=True)
    bad = (n < 1e-9)[:, 0]
    t[bad] = central[bad]
    n[bad] = 1.0
    return t / (n + 1e-9)


def duct_mesh_arrays(contours, diameter, radial=14, occupied=None, resolution=0.05, probe=None):
    """Tube mesh along each contour.

    occupied(xy (n,2)) -> bool: which side of a contour is solid. The tube's axis sits R *inside*
    the boundary, so its outer face lies on the boundary -- which is where the LiDAR tracer puts
    the hose (it hits the occupancy edge and adds the cylinder correction from there). Centred on
    the contour, the drawn hose stood R = 16 cm into the lane and every beam looked as if it went
    through the wall.

    The solid side is decided **once per contour**, not per vertex. A closed contour bounds a
    region, so its interior lies on one side of it for the whole loop -- a per-vertex answer is not
    merely noisier, it is a question that cannot have two answers. Probing each vertex independently
    let a thin wall or a tight corner flip the sign, and a flipped sign moves that ring's axis by 2R
    across a single segment: on Korea's duct, adjacent axis samples jumped 0.330 m, one full hose
    diameter, which is the twisted-foil look in the wide screenshots. Contour 6 there flipped four
    times over 68 points, so most of that loop was folded rather than a stray vertex or two.

    The vote is weighted by segment length so a dense staircase of short segments cannot outvote the
    long smooth runs, and it falls back to the winding when a contour is genuinely ambiguous.
    """
    R = diameter / 2
    probe = max(1.2 * resolution, 0.06) if probe is None else probe
    P, N, C, I = [], [], [], []
    base = 0
    for c in contours:
        pts = smooth_contour(np.asarray(c, np.float64), resolution)
        n = len(pts)
        if n < 4: continue
        # Tangent over a fixed *physical* window, not over vertex indices. The axis below is this
        # contour offset by a full R, and offsetting scales the local step by (1 - kR): wherever the
        # estimated curvature exceeds 1/R the offset curve reverses and the tube folds through
        # itself. `pts[i+1] - pts[i-1]` measures curvature at whatever spacing RDP happened to
        # leave, which on Korea's duct is ~16 mm -- a tenth of R -- so it reads sampling noise as
        # curvature and produced 439 cusps on one contour. A window of R is the scale a hose can
        # actually follow. The offset itself is unchanged, so the outer surface still passes through
        # every contour vertex and the wall does not move.
        tang = _tangent_arclength(pts, R)
        seg = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1); u = np.concatenate([[0], np.cumsum(seg)[:-1]])
        n1 = np.stack([-tang[:, 1], tang[:, 0], np.zeros(n)], 1); n2 = np.array([0, 0, 1.0])
        if occupied is not None:
            inward = (occupied(pts + probe * n1[:, :2]).astype(np.float64)
                      - occupied(pts - probe * n1[:, :2]).astype(np.float64))
            vote = float(np.sum(seg * inward))
            if vote > 0:
                sign = 1.0
            elif vote < 0:
                sign = -1.0
            else:                                   # no probe saw a difference: fall back to winding
                area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
                sign = 1.0 if area >= 0 else -1.0
            # Full R, deliberately: the tube's outer face has to stay on the boundary because that is
            # where the tracer puts the hose. Offsetting by less to keep the axis inside a thin band
            # would move the drawn surface R - depth into the lane while the LiDAR and the contact
            # test went on using the boundary -- the same visible-penetration gap the docstring above
            # records, reintroduced as a cosmetic fix. Where a mapped band is thinner than 2R the map
            # and the hose diameter genuinely disagree; that is measured and reported, not papered over.
            pts = pts + (R * sign) * n1[:, :2]
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
        # Wind the faces the way their own normals point. For a tube running along +x the ring
        # normal at angle 0 is +y, while cross(v1-v0, v2-v0) for [i, next_ring, next_rad] comes out
        # along x cross z = -y: every triangle was wound inward. On a plain 64-point circle -- no
        # folds, no staircase -- all 1792 triangles disagree with their supplied normal, mean dot
        # -0.997. `Mesh.cull` is on and the main pass culls back faces, so the surface facing the
        # camera was the one being discarded and what showed was the inside of the far wall. The
        # shadow pass culls *front*, so it was shadow-mapping the outward surface as well.
        tri = np.stack([i, i_next_rad, i_next_ring,
                        i_next_rad, i_next_both, i_next_ring], -1).reshape(-1, 3)
        I.append(tri + base); base += n * radial
    if not P:
        return None
    return (np.vstack(P).astype(np.float32), np.vstack(N).astype(np.float32),
            np.vstack(C).astype(np.float32), np.vstack(I).reshape(-1).astype(np.int32))


def wall_mesh_arrays(contours, height=1.0, is_solid=None, resolution=None):
    """Extruded walls. contours: closed polylines of tall objects.

    is_solid(xy) -> bool tells whether a point is inside a solid object; contours enclosing solid
    space get a roof (clutter), the room boundary (enclosing floor) does not.

    `resolution` opts into the shared contour cleanup (see `smooth_contour`); without it the old
    fixed 0.04 m spacing filter is kept so existing callers get exactly what they got before."""
    try:
        import mapbox_earcut as earcut
    except ImportError:
        earcut = None
    P, N, C, I = [], [], [], []
    base = 0
    for c in contours:
        raw = np.asarray(c, np.float64)
        closed = True if resolution is None else is_closed(raw, resolution)
        pts = (_simplify(raw, 0.04) if resolution is None else smooth_contour(raw, resolution, closed=closed))
        n = len(pts)
        if n < 3: continue
        nxt = np.roll(pts, -1, 0)
        pairs = list(zip(pts, nxt))[:-1] if not closed else list(zip(pts, nxt))
        for a, b in pairs:
            d = b - a; nrm = np.array([d[1], -d[0], 0.0]); nrm /= np.linalg.norm(nrm) + 1e-9
            quad = np.array([[a[0], a[1], 0], [b[0], b[1], 0], [b[0], b[1], height], [a[0], a[1], height]])
            P.append(quad); N.append(np.tile(nrm, (4, 1))); C.append(np.tile([0.62, 0.63, 0.66, 0.0], (4, 1)))
            I.append(np.array([0, 1, 2, 0, 2, 3]) + base); base += 4
        solid = is_solid(pts.mean(0)) if is_solid is not None else (pts.max(0) - pts.min(0)).max() < 8.0
        if earcut is not None and n >= 3 and solid and closed:      # an open run does not enclose a roof
            tri = earcut.triangulate_float32(pts.astype(np.float32), np.array([n], np.uint32))
            top = np.concatenate([pts, np.full((n, 1), height)], 1)
            P.append(top); N.append(np.tile([0, 0, 1.0], (n, 1))); C.append(np.tile([0.66, 0.67, 0.7, 0.0], (n, 1)))
            I.append(tri.astype(np.int64) + base); base += n
    if not P:
        return None
    return (np.vstack(P).astype(np.float32), np.vstack(N).astype(np.float32), np.vstack(C).astype(np.float32),
            np.concatenate([i.reshape(-1) for i in I]).astype(np.int32))


def floor_mesh_arrays(bounds, margin=3.0):
    """The ground quad under a track, as arrays (see Scene.build_floor)."""
    x0, y0, x1, y1 = bounds[0] - margin, bounds[1] - margin, bounds[2] + margin, bounds[3] + margin
    pos = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], np.float32)
    nrm = np.tile([0, 0, 1.0], (4, 1)).astype(np.float32)
    col = np.tile([0.30, 0.31, 0.34, 0.0], (4, 1)).astype(np.float32)
    return pos, nrm, col, np.array([0, 1, 2, 0, 2, 3], np.int32)


# ---------------------------------------------------------------- framing
#
# The ground used to be a quad over the map *file's* canvas. That is not where the track is: Monza's
# canvas is a 191.7 m square holding a circuit that occupies a fraction of it, so the plate ran far
# past the racing surface in every direction and the track sat small and off-centre inside it. These
# derive the frame from the geometry that actually gets drawn instead.

def _stack(point_sets):
    p = [np.asarray(s, np.float64).reshape(-1, 2) for s in point_sets if s is not None and len(s)]
    return np.vstack(p) if p else None


def _min_area_angle(pts):
    """Angle of the minimum-area rectangle over the convex hull (rotating calipers), in radians."""
    from scipy.spatial import ConvexHull
    hull = pts[ConvexHull(pts).vertices]
    best = None
    for e in np.roll(hull, -1, 0) - hull:
        L = np.linalg.norm(e)
        if L < 1e-9:
            continue
        t = e / L
        n = np.array([-t[1], t[0]])
        u, v = hull @ t, hull @ n
        area = float(np.ptp(u) * np.ptp(v))
        if best is None or area < best[0]:
            best = (area, float(math.atan2(t[1], t[0])))
    return best[1]


def content_frame(orient_sets, extent_sets):
    """Oriented frame of the drawn content: (angle_rad, centre (2,), extent (2,)).

    Orientation and size come from deliberately different inputs.

    `orient_sets` is what defines which way the *track* lies -- the centerline, or the duct contours
    when there is no centerline. Not the walls: on Korea the tall layer is the SLAM scan's unknown
    surround, whose minimum-area rectangle is the cropped canvas and which therefore reports 90
    degrees however the track inside it is turned. The centerline reports 82.95 degrees, which is the
    7-degree tilt actually visible in the map. Fitting the *free* pixels does not work either -- on
    Monza the free region includes everything outside the circuit, so its rectangle is the canvas
    square at 0 degrees with no area saved at all.

    `extent_sets` is everything that will be drawn, so the frame cannot crop any of it. Size and
    orientation must come from different sets for exactly that reason: the surround says nothing
    about heading but it does have to fit on the plate."""
    o = _stack(orient_sets)
    e = _stack(extent_sets)
    if e is None:
        e = o
    if o is None or len(o) < 3:
        o = e
    angle = _min_area_angle(o) if len(o) >= 3 else 0.0
    t = np.array([math.cos(angle), math.sin(angle)])
    n = np.array([-t[1], t[0]])
    u, v = e @ t, e @ n
    centre = 0.5 * (u.min() + u.max()) * t + 0.5 * (v.min() + v.max()) * n
    return angle, centre, np.array([float(np.ptp(u)), float(np.ptp(v))])


def _rounded_rect(half, radius, per_corner=6):
    """Rounded rectangle polygon (CCW) in a local frame centred at the origin."""
    hx, hy = half
    r = float(min(radius, hx, hy))
    out = []
    for sx, sy, a0 in ((1, 1, 0.0), (-1, 1, math.pi / 2), (-1, -1, math.pi), (1, -1, 1.5 * math.pi)):
        cx, cy = sx * (hx - r), sy * (hy - r)
        a = a0 + np.linspace(0.0, math.pi / 2, per_corner)
        out.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1))
    return np.vstack(out)


def _fan(poly, z, colour, centre_colour=None):
    """Triangle fan around the centroid, as (pos, nrm, col, idx)."""
    c = poly.mean(0)
    pts = np.vstack([c[None], poly])
    pos = np.concatenate([pts, np.full((len(pts), 1), z)], 1)
    nrm = np.tile([0, 0, 1.0], (len(pts), 1))
    col = np.tile(np.asarray(colour, np.float64), (len(pts), 1))
    if centre_colour is not None:
        col[0] = centre_colour
    k = np.arange(1, len(pts))
    idx = np.stack([np.zeros_like(k), k, np.roll(k, -1)], 1).reshape(-1)
    return (pos.astype(np.float32), nrm.astype(np.float32), col.astype(np.float32), idx.astype(np.int32))


def apron_mesh_arrays(angle, centre, extent, margin=2.5, colour=(0.235, 0.245, 0.275, 0.0)):
    """The ground plate, oriented with the track and sized to it rather than to the map file."""
    half = 0.5 * np.asarray(extent, np.float64) + margin
    poly = _rounded_rect(half, radius=min(2.0, 0.18 * float(min(half))))
    ca, sa = math.cos(angle), math.sin(angle)
    R = np.array([[ca, -sa], [sa, ca]])
    return _fan(poly @ R.T + np.asarray(centre, np.float64), 0.0, colour)


def backdrop_mesh_arrays(centre, extent, margin=2.5, segments=64,
                         inner=(0.115, 0.128, 0.152, 0.0), outer=(0.055, 0.065, 0.085, 0.0)):
    """A disc under and around the apron whose rim fades to the clear colour.

    Without it the ground ends in a hard edge and the eye reads that edge as the extent of the world.
    `outer` is the scene's own clear colour (`Scene.clear`), so the rim dissolves into the background
    instead of drawing a boundary that nothing in the simulation corresponds to. It is a backdrop,
    not a wall: nothing here is collidable and nothing here is claimed to be."""
    half = 0.5 * np.asarray(extent, np.float64) + margin
    r = 1.7 * float(np.linalg.norm(half))
    a = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    ring = np.stack([r * np.cos(a), r * np.sin(a)], 1) + np.asarray(centre, np.float64)
    return _fan(ring, -0.02, outer, centre_colour=inner)


# ---------------------------------------------------------------- helpers
def _simplify(pts, min_dist):
    """Drop points closer together than `min_dist`. Kept for callers that still pass a spacing.

    Note this cannot remove a staircase: its criterion is the gap between consecutive points, and a
    marching-squares staircase's points are exactly one cell apart, so any `min_dist` at or below the
    grid resolution keeps every one of them and any value above it eats real detail as readily as
    treads. `_rdp` is the criterion that actually applies -- see there."""
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) >= min_dist:
            out.append(p)
    if len(out) > 2 and np.linalg.norm(out[-1] - out[0]) < min_dist:
        out.pop()
    return np.asarray(out)

def _rdp(pts, eps, closed=True):
    """Douglas-Peucker simplification: keep a point only if dropping it would move the polyline by
    more than `eps`.

    Perpendicular deviation is the criterion that collapses a staircase, because a run of treads and
    risers deviates from its own diagonal by half a cell however many points it holds. Measured on
    Korea's duct at eps = 0.25 cell it takes the axis-aligned segment share from 64 % to 13 %, which
    point spacing at any threshold cannot do.

    A closed loop has no endpoints to anchor the recursion on, so the two mutually farthest points
    are used as the seed pair and the two spans between them are simplified separately."""
    P = np.asarray(pts, np.float64)
    if closed and len(P) > 2 and np.allclose(P[0], P[-1]):
        P = P[:-1]
    n = len(P)
    if n < 3:
        return P
    if not closed:
        keep = np.zeros(n, bool); keep[0] = keep[-1] = True
        _rdp_span(P, 0, n - 1, eps, keep, n)
        return P[keep]
    a = int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))
    b = int(np.argmax(np.linalg.norm(P - P[a], axis=1)))
    if a > b:
        a, b = b, a
    keep = np.zeros(n, bool); keep[a] = keep[b] = True
    _rdp_span(P, a, b, eps, keep, n)
    _rdp_span(P, b, a + n, eps, keep, n)          # the span that wraps past index 0
    return P[keep]


def _rdp_span(P, i, j, eps, keep, n):
    stack = [(i, j)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        idx = np.arange(i + 1, j)
        p0, p1 = P[i % n], P[j % n]
        d = p1 - p0
        L = np.linalg.norm(d)
        Q = P[idx % n]
        if L < 1e-12:
            dev = np.linalg.norm(Q - p0, axis=1)
        else:
            dev = np.abs(d[0] * (Q[:, 1] - p0[1]) - d[1] * (Q[:, 0] - p0[0])) / L
        m = int(np.argmax(dev))
        if dev[m] > eps:
            k = int(idx[m])
            keep[k % n] = True
            stack.append((i, k)); stack.append((k, j))


def _smooth_closed(pts, iters=2):
    for _ in range(iters):                        # Chaikin corner cutting on a closed loop
        nxt = np.roll(pts, -1, 0)
        out = np.empty((2 * len(pts), 2))
        out[0::2] = 0.75 * pts + 0.25 * nxt; out[1::2] = 0.25 * pts + 0.75 * nxt
        pts = out
    return pts


def _smooth_capped(pts, iters=2, cap=0.02):
    """Chaikin corner cutting whose cut length is capped at `cap` metres.

    Plain Chaikin cuts every corner at a fixed quarter of the adjacent edges. That is harmless while
    the edges are one cell long, but `_rdp` deliberately produces long edges, so a genuine
    right-angle track corner between two multi-metre segments gets rounded by a quarter of them --
    1.8 cells of deviation measured on Korea's duct, which no simplification tolerance can undo.

    Capping the parameter at `cap / edge_length` bounds the deviation at `cap` and leaves long
    straight edges alone, which costs nothing: cutting the corner off a straight line is a no-op."""
    P = np.asarray(pts, np.float64)
    for _ in range(iters):
        nxt = np.roll(P, -1, 0)
        seg = nxt - P
        t = np.minimum(0.25, cap / np.maximum(np.linalg.norm(seg, axis=1), 1e-12))[:, None]
        out = np.empty((2 * len(P), 2))
        out[0::2] = P + t * seg
        out[1::2] = nxt - t * seg
        P = out
    return P


def is_closed(pts, resolution):
    """Whether a contour is a loop. Clipping the canvas-edge silhouette leaves open polylines."""
    P = np.asarray(pts)
    return len(P) > 2 and float(np.linalg.norm(P[0] - P[-1])) <= 2.0 * resolution


def smooth_contour(pts, resolution, eps_cells=0.25, cap_cells=0.35, iters=2, closed=None):
    """The contour cleanup the map geometry builders share: simplify, then round what is left.

    Both tolerances are expressed in grid cells because that is the only scale at which the input
    means anything -- marching squares over a binary occupancy grid picks the sub-cell digits
    arbitrarily, so moving the polyline by a fraction of a cell does not move the wall."""
    P = np.asarray(pts, np.float64)
    if closed is None:
        closed = is_closed(P, resolution)
    P = _rdp(P, eps_cells * resolution, closed=closed)
    if len(P) < 4:
        return P
    if closed:
        return _smooth_capped(P, iters, cap_cells * resolution)
    ends = (P[0].copy(), P[-1].copy())
    P = _smooth_capped(P, iters, cap_cells * resolution)
    # _smooth_capped wraps, so an open run would otherwise grow a chord joining its two ends
    keep = int(len(P) - 2 ** iters + 1)
    P = P[:max(2, keep)]
    P[0], P[-1] = ends[0], ends[1]
    return P

_FONT = {}
# blue -> cyan -> green -> yellow -> red. One ramp for every speed shown anywhere, so a colour means
# the same thing in the 3D plan line and in the BEV panel; two ramps in one window is worse than none.
SPEED_STOPS = np.array([[0.10, 0.20, 0.90],      # 0.00  blue
                        [0.00, 0.85, 0.95],      # 0.25  cyan
                        [0.10, 0.95, 0.25],      # 0.50  green
                        [1.00, 0.90, 0.10],      # 0.75  yellow
                        [1.00, 0.25, 0.10]], np.float32)   # 1.00  red


def speed_colors(v, v_max: float, alpha: float = 1.0):
    """(N,) speeds -> (N,4) RGBA against a v_max top of scale."""
    t = np.clip(np.asarray(v, np.float32) / max(1e-3, float(v_max)), 0.0, 1.0)
    f = t * (len(SPEED_STOPS) - 1)
    i = np.clip(f.astype(np.int32), 0, len(SPEED_STOPS) - 2)
    a = (f - i)[:, None]
    rgb = SPEED_STOPS[i] * (1 - a) + SPEED_STOPS[i + 1] * a
    return np.concatenate([rgb, np.full((len(rgb), 1), alpha, np.float32)], 1)


def _font(size):
    if size not in _FONT:
        from PIL import ImageFont
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if os.path.exists(path):
                _FONT[size] = ImageFont.truetype(path, size); break
        else:
            _FONT[size] = ImageFont.load_default()
    return _FONT[size]
