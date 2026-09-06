"""Detailed procedural F1TENTH car (Traxxas Slash 4x4 based, two-level autonomy platform) -> GLB.

Frame: base_link = rear axle center on the ground, x forward, y left, z up (ROS).
Nodes: chassis_* (by material class), wheel_{fl,fr,rl,rr} (children: tire/rim/nut), lidar.
Geometry names end with a material class the viewer maps to PBR params:
    rubber, plastic, metal, blue, glass, paint, pcb
"""
import sys
import numpy as np
import trimesh
from trimesh.creation import box, cylinder, revolve, extrude_polygon, icosphere
from trimesh.transformations import rotation_matrix as rot
from shapely.geometry import Polygon, box as sbox

# ---------------------------------------------------------------- dimensions (Traxxas Slash 4x4, 1/10)
WB = 0.3302            # wheelbase
TW = 0.290             # track width (wheel centers)
TIRE_OD, TIRE_W = 0.112, 0.045
ZC = TIRE_OD / 2       # axle height
Z_PLATE = 0.026        # chassis plate height
Z_UP = 0.100           # upper platform height
LIDAR_X = 0.27         # matches LidarParams.mount_x

def rgba(h, a=255):
    h = h.lstrip("#"); return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]

def paint(m, col):
    m.visual.face_colors = np.tile(rgba(col), (len(m.faces), 1)); return m

def B(ext, center, col, yaw=0.0, pitch=0.0):
    m = box(ext)
    if pitch: m.apply_transform(rot(pitch, [0, 1, 0]))
    if yaw: m.apply_transform(rot(yaw, [0, 0, 1]))
    m.apply_translation(center); return paint(m, col)

def C(r, h, center, col, axis="z", sections=24):
    m = cylinder(radius=r, height=h, sections=sections)
    if axis == "y": m.apply_transform(rot(np.pi / 2, [1, 0, 0]))
    if axis == "x": m.apply_transform(rot(np.pi / 2, [0, 1, 0]))
    m.apply_translation(center); return paint(m, col)

def seg(p0, p1, r, col, sections=12):
    """cylinder from p0 to p1"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; L = np.linalg.norm(d)
    m = cylinder(radius=r, height=L, sections=sections)
    z = np.array([0, 0, 1.0]); dn = d / L
    v = np.cross(z, dn); s = np.linalg.norm(v); c = np.dot(z, dn)
    if s > 1e-8:
        m.apply_transform(rot(np.arctan2(s, c), v / s))
    elif c < 0:
        m.apply_transform(rot(np.pi, [1, 0, 0]))
    m.apply_translation((p0 + p1) / 2); return paint(m, col)

def tube(pts, r, col):
    parts = [seg(pts[i], pts[i + 1], r, col, 10) for i in range(len(pts) - 1)]
    for p in pts[1:-1]:
        s = icosphere(1, r); s.apply_translation(p); parts.append(paint(s, col))
    return parts

def bezier(p0, p1, p2, p3, n=14):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(lambda p: np.asarray(p, float), (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3

def rounded_plate(lx, ly, t, r, center, col, holes=()):
    poly = sbox(-lx / 2, -ly / 2, lx / 2, ly / 2).buffer(-r).buffer(r, join_style=1)
    for (hx, hy, hr) in holes:
        poly = poly.difference(Polygon([(hx + hr * np.cos(a), hy + hr * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24)]))
    m = extrude_polygon(poly, t); m.apply_translation([center[0], center[1], center[2] - t / 2]); return paint(m, col)

def revolve_y(profile, center, col, sections=48):
    """profile: (r, axial) pairs, revolved around the wheel axle (y)."""
    m = revolve(np.asarray(profile, float), sections=sections)
    m.apply_transform(rot(-np.pi / 2, [1, 0, 0]))     # z axis -> y axis
    m.apply_translation(center); return paint(m, col)

# ---------------------------------------------------------------- wheel
def wheel(side):
    """returns dict material-class -> mesh, centered at the wheel, axle along y. side=+1 left."""
    w = TIRE_W; R = TIRE_OD / 2; rr = 0.036
    tire = revolve_y([[rr, -w / 2], [R - 0.012, -w / 2], [R - 0.004, -w / 2 + 0.005], [R - 0.002, -0.012],
                      [R - 0.002, 0.012], [R - 0.004, w / 2 - 0.005], [R - 0.012, w / 2], [rr, w / 2], [rr, -w / 2]],
                     [0, 0, 0], "#161616", 56)
    tread = []
    n = 26
    for i in range(n):
        a = 2 * np.pi * i / n
        for k, (yy, skew) in enumerate([(-w * 0.28, 0.35), (w * 0.28, -0.35)]):
            blk = box([0.014, w * 0.36, 0.006])
            blk.apply_transform(rot(skew, [0, 0, 1]))
            blk.apply_translation([0, yy, R - 0.001])
            blk.apply_transform(rot(a + (0.12 if k else 0), [0, 1, 0]))
            tread.append(paint(blk, "#1c1c1c"))
    rim = revolve_y([[0.012, -w * 0.42], [rr + 0.002, -w * 0.42], [rr + 0.002, w * 0.42], [rr - 0.006, w * 0.42],
                     [rr - 0.006, -w * 0.30], [0.012, -w * 0.30], [0.012, -w * 0.42]], [0, 0, 0], "#2a2a2a", 40)
    # dish face on the outside with 6 split spokes
    spokes = []
    for i in range(6):
        a = 2 * np.pi * i / 6
        for off in (-0.004, 0.004):
            sp = box([0.0045, 0.006, rr - 0.010]); sp.apply_translation([off, side * w * 0.36, (rr + 0.012) / 2])
            sp.apply_transform(rot(a, [0, 1, 0])); spokes.append(paint(sp, "#3a3a3a"))
    hub = C(0.012, w * 0.9, [0, 0, 0], "#3a3a3a", "y", 18)
    nut = C(0.007, 0.007, [0, side * (w * 0.42 + 0.003), 0], "#b8bcc4", "y", 6)
    return {"rubber": trimesh.util.concatenate([tire] + tread),
            "plastic": trimesh.util.concatenate([rim, hub] + spokes),
            "metal": nut}

# ---------------------------------------------------------------- chassis assembly
def chassis():
    P = {"plastic": [], "metal": [], "blue": [], "rubber": [], "glass": [], "paint": [], "pcb": []}
    add = lambda cls, *ms: P[cls].extend(ms)
    xm = WB / 2
    # --- LCG chassis tub: plate, side rails, spine, rear/front bulkheads with diff housings
    add("plastic", rounded_plate(0.40, 0.125, 0.004, 0.02, [xm, 0, Z_PLATE], "#1d1d1f"))
    for y in (-1, 1):
        add("plastic", B([0.30, 0.010, 0.022], [xm, y * 0.062, Z_PLATE + 0.013], "#232326"))
    add("plastic", B([0.28, 0.024, 0.010], [xm, 0, Z_PLATE + 0.007], "#232326"))
    for x, d in ((WB, 1), (0.0, -1)):
        add("plastic", B([0.06, 0.088, 0.048], [x + d * 0.006, 0, Z_PLATE + 0.026], "#28282b"))       # bulkhead
        add("plastic", C(0.022, 0.05, [x, 0, ZC], "#303033", "y", 20))                                # diff housing
        # shock tower (trapezoid plate) with two shock mounts
        tower = extrude_polygon(Polygon([(-0.075, 0), (0.075, 0), (0.048, 0.078), (-0.048, 0.078)]), 0.005)
        tower.apply_transform(rot(np.pi / 2, [1, 0, 0])); tower.apply_transform(rot(np.pi / 2, [0, 0, 1]))
        tower.apply_translation([x + d * 0.036, 0, Z_PLATE + 0.02]); add("plastic", paint(tower, "#28282b"))
        # suspension per side
        for y in (-1, 1):
            # lower A-arm (two legs)
            add("plastic", B([0.012, 0.09, 0.008], [x + 0.018, y * 0.085, Z_PLATE + 0.012], "#28282b"))
            add("plastic", B([0.012, 0.09, 0.008], [x - 0.018, y * 0.085, Z_PLATE + 0.012], "#28282b"))
            add("plastic", B([0.048, 0.014, 0.008], [x, y * 0.125, Z_PLATE + 0.012], "#28282b"))
            # upper camber link
            add("metal", seg([x, y * 0.042, Z_PLATE + 0.062], [x, y * 0.118, ZC + 0.02], 0.0028, "#c9ccd2"))
            # hub carrier / knuckle + axle
            add("plastic", B([0.026, 0.018, 0.044], [x, y * 0.130, ZC], "#2a2a2d"))
            add("metal", C(0.005, 0.03, [x, y * 0.140, ZC], "#a8adb5", "y", 12))
            # drive shaft
            add("metal", seg([x, y * 0.03, ZC - 0.002], [x, y * 0.122, ZC], 0.0035, "#b0b4bb"))
            # oil shock: body (blue anodized cap + grey body), shaft, spring
            top = np.array([x + d * 0.036, y * 0.052, Z_PLATE + 0.096]); bot = np.array([x + d * 0.004, y * 0.108, Z_PLATE + 0.018])
            axis = (bot - top) / np.linalg.norm(bot - top); L = np.linalg.norm(bot - top)
            add("blue", seg(top, top + axis * 0.012, 0.0075, "#2d6cdf"))
            add("metal", seg(top + axis * 0.012, top + axis * (L * 0.55), 0.0065, "#8d939c"))
            add("metal", seg(top + axis * (L * 0.55), bot, 0.0025, "#d8dbe0"))
            add("plastic", seg(bot - axis * 0.006, bot + axis * 0.004, 0.006, "#2a2a2d"))
            # spring as a helix
            for t in np.linspace(0, 1, 7 * 10):
                a = 2 * np.pi * 7 * t
                q = top + axis * (0.010 + t * (L - 0.020))
                # perpendicular frame
                u = np.cross(axis, [1, 0, 0]); u /= np.linalg.norm(u); v = np.cross(axis, u)
                p = q + 0.0105 * (np.cos(a) * u + np.sin(a) * v)
                s = icosphere(1, 0.0016); s.apply_translation(p); add("metal", paint(s, "#e2e4e8"))
    # --- bumpers
    add("plastic", B([0.05, 0.05, 0.012], [WB + 0.07, 0, Z_PLATE + 0.008], "#1d1d1f"))                 # bumper mount
    add("plastic", B([0.022, 0.19, 0.03], [WB + 0.118, 0, Z_PLATE + 0.014], "#1d1d1f"))                # front bumper bar
    for y in (-1, 1):
        add("plastic", B([0.05, 0.02, 0.03], [WB + 0.10, y * 0.105, Z_PLATE + 0.014], "#1d1d1f", yaw=y * 0.55))
    add("rubber", B([0.026, 0.165, 0.05], [WB + 0.142, 0, Z_PLATE + 0.026], "#3b3b3e"))                # foam pad
    add("plastic", B([0.018, 0.16, 0.022], [-0.062, 0, Z_PLATE + 0.012], "#1d1d1f"))                   # rear bumper
    # --- side nerf bars
    for y in (-1, 1):
        add("plastic", B([0.24, 0.012, 0.005], [xm, y * 0.138, Z_PLATE + 0.03], "#232326"))
        for x in (xm - 0.09, xm + 0.09):
            add("plastic", B([0.012, 0.05, 0.005], [x, y * 0.115, Z_PLATE + 0.03], "#232326"))
    # --- drivetrain: motor, spur cover, center driveshaft
    add("metal", C(0.018, 0.052, [0.062, -0.034, Z_PLATE + 0.036], "#4a4d55", "y", 28))                 # motor can
    add("metal", C(0.019, 0.004, [0.062, -0.062, Z_PLATE + 0.036], "#8d939c", "y", 28))                 # end bell
    add("plastic", C(0.028, 0.008, [0.03, -0.012, Z_PLATE + 0.034], "#1d1d1f", "y", 30))               # spur gear cover
    add("metal", seg([0.035, 0, Z_PLATE + 0.03], [WB - 0.035, 0, Z_PLATE + 0.03], 0.004, "#b0b4bb"))   # center shaft
    # --- battery (LiPo, left tray) with strap, VESC 6 (right tray) with heatsink + caps
    add("paint", B([0.138, 0.047, 0.026], [0.165, 0.058, Z_PLATE + 0.020], "#1f3f9c"))
    add("paint", B([0.06, 0.047, 0.0265], [0.165, 0.058, Z_PLATE + 0.020], "#dcdcdc"))                 # label
    add("rubber", B([0.012, 0.055, 0.032], [0.165, 0.058, Z_PLATE + 0.020], "#111111"))                 # strap
    add("plastic", B([0.062, 0.062, 0.018], [0.235, -0.052, Z_PLATE + 0.016], "#141416"))               # VESC body
    for i in range(7):
        add("metal", B([0.058, 0.004, 0.012], [0.235, -0.052 - 0.03 + i * 0.010, Z_PLATE + 0.031], "#9a9fa8"))
    add("plastic", C(0.008, 0.02, [0.20, -0.052, Z_PLATE + 0.028], "#101010", "z", 16))                 # capacitor
    add("plastic", B([0.04, 0.03, 0.016], [0.10, 0.02, Z_PLATE + 0.016], "#1d1d1f"))                    # receiver box
    # --- standoffs + upper platform (laser cut, rounded, with the Hokuyo cutout at the front)
    for x in (0.03, xm, WB - 0.03):
        for y in (-0.058, 0.058):
            add("metal", C(0.0045, Z_UP - Z_PLATE, [x, y, (Z_UP + Z_PLATE) / 2], "#c9ccd2", "z", 12))
            add("metal", C(0.006, 0.003, [x, y, Z_UP + 0.004], "#c9ccd2", "z", 6))                      # nut on top
    add("plastic", rounded_plate(0.375, 0.205, 0.005, 0.025, [xm + 0.015, 0, Z_UP], "#26282c",
                                 holes=[(0.135, 0.075, 0.012), (0.135, -0.075, 0.012)]))
    # --- Jetson AGX (dev kit: black frame, brushed top, side ports), rear of platform
    jx = 0.095
    add("plastic", B([0.105, 0.105, 0.012], [jx, 0, Z_UP + 0.0085], "#101012"))
    add("metal", B([0.104, 0.104, 0.052], [jx, 0, Z_UP + 0.040], "#3b3d42"))
    add("plastic", B([0.106, 0.106, 0.004], [jx, 0, Z_UP + 0.068], "#131316"))
    for i in range(9):
        add("metal", B([0.096, 0.003, 0.010], [jx, -0.044 + i * 0.011, Z_UP + 0.074], "#5b5e66"))       # heatsink slats
    add("glass", B([0.010, 0.004, 0.003], [jx + 0.05, 0.03, Z_UP + 0.03], "#38ff7a"))                    # LED
    for i in range(3):
        add("plastic", B([0.004, 0.012, 0.006], [jx + 0.0535, -0.03 + i * 0.016, Z_UP + 0.02], "#0a0a0a"))  # USB ports
    # --- power board (green PCB), USB hub, wifi antenna
    add("pcb", B([0.09, 0.055, 0.0016], [0.225, -0.062, Z_UP + 0.0035], "#1f6b3a"))
    for i, (dx, dy) in enumerate([(-0.03, 0.0), (0.0, 0.018), (0.028, -0.01)]):
        add("plastic", B([0.014, 0.012, 0.010], [0.225 + dx, -0.062 + dy, Z_UP + 0.0095], "#111111"))
    add("plastic", C(0.004, 0.012, [0.245, -0.075, Z_UP + 0.008], "#e0b23c", "z", 12))                   # fuse
    add("metal", B([0.07, 0.028, 0.012], [0.215, 0.07, Z_UP + 0.0085], "#a9aeb6"))                     # USB hub
    add("plastic", seg([0.05, -0.085, Z_UP + 0.004], [0.05, -0.085, Z_UP + 0.09], 0.003, "#141416"))    # antenna
    # --- cables (bezier tubes)
    cab = lambda a, b, c, d, r, col: add("rubber", *tube(bezier(a, b, c, d), r, col))
    cab([0.235, -0.052, Z_PLATE + 0.026], [0.20, -0.06, Z_PLATE + 0.05], [0.19, 0.03, Z_PLATE + 0.05], [0.165, 0.055, Z_PLATE + 0.034], 0.0025, "#c81e1e")
    cab([0.235, -0.045, Z_PLATE + 0.026], [0.20, -0.05, Z_PLATE + 0.055], [0.19, 0.04, Z_PLATE + 0.055], [0.170, 0.06, Z_PLATE + 0.034], 0.0025, "#111111")
    for k in range(3):
        cab([0.21, -0.06 + k * 0.006, Z_PLATE + 0.02], [0.14, -0.07, Z_PLATE + 0.04], [0.10, -0.06, Z_PLATE + 0.045], [0.075, -0.05, Z_PLATE + 0.036], 0.0018, "#111111")
    cab([0.245, -0.062, Z_UP + 0.005], [0.26, -0.03, Z_UP + 0.05], [0.16, 0.0, Z_UP + 0.05], [0.148, 0.0, Z_UP + 0.03], 0.0018, "#111111")   # power -> jetson
    cab([LIDAR_X - 0.02, 0.0, Z_UP + 0.02], [0.24, 0.02, Z_UP + 0.05], [0.18, 0.04, Z_UP + 0.05], [0.148, 0.03, Z_UP + 0.03], 0.0022, "#3a6fb0")  # ethernet lidar -> jetson
    cab([0.215, 0.07, Z_UP + 0.012], [0.19, 0.075, Z_UP + 0.04], [0.16, 0.06, Z_UP + 0.04], [0.148, 0.04, Z_UP + 0.02], 0.0018, "#111111")   # usb hub -> jetson
    return {k: trimesh.util.concatenate(v) for k, v in P.items() if v}

def lidar():
    """Hokuyo UST-10LX (50x50x70): black body, dark scanner head with window, orange band."""
    P = {"plastic": [], "glass": [], "paint": [], "metal": []}
    P["plastic"].append(B([0.05, 0.05, 0.040], [0, 0, 0.020], "#141416"))
    P["plastic"].append(B([0.046, 0.046, 0.004], [0, 0, 0.042], "#0c0c0e"))
    P["glass"].append(C(0.0215, 0.022, [0, 0, 0.055], "#0e1a24", "z", 40))
    P["paint"].append(C(0.0225, 0.005, [0, 0, 0.0685], "#f28c1c", "z", 40))
    P["plastic"].append(C(0.0215, 0.004, [0, 0, 0.073], "#141416", "z", 40))
    P["metal"].append(B([0.012, 0.016, 0.006], [-0.024, 0, 0.012], "#a9aeb6"))                          # connector
    return {k: trimesh.util.concatenate(v) for k, v in P.items()}

def build():
    sc = trimesh.Scene()
    for cls, m in chassis().items():
        sc.add_geometry(m, node_name=f"chassis_{cls}", geom_name=f"chassis_{cls}")
    T = np.eye(4); T[:3, 3] = [LIDAR_X, 0, Z_UP + 0.0025]
    sc.graph.update(frame_from="world", frame_to="lidar", matrix=T)
    for cls, m in lidar().items():
        sc.add_geometry(m, node_name=f"lidar_{cls}", geom_name=f"lidar_{cls}", parent_node_name="lidar")
    for name, x, y, side in (("wheel_fl", WB, TW / 2, 1), ("wheel_fr", WB, -TW / 2, -1),
                             ("wheel_rl", 0.0, TW / 2, 1), ("wheel_rr", 0.0, -TW / 2, -1)):
        T = np.eye(4); T[:3, 3] = [x, y, ZC]
        sc.graph.update(frame_from="world", frame_to=name, matrix=T)
        for cls, m in wheel(side).items():
            sc.add_geometry(m, node_name=f"{name}_{cls}", geom_name=f"{name}_{cls}", parent_node_name=name)
    return sc

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "f1sim/assets/f1tenth_car.glb"
    sc = build(); sc.export(out, include_normals=True)
    b = sc.bounds
    print(f"saved {out}: {sum(len(g.faces) for g in sc.geometry.values())} faces, {len(sc.geometry)} meshes, "
          f"x[{b[0,0]:.3f},{b[1,0]:.3f}] y[{b[0,1]:.3f},{b[1,1]:.3f}] z[{b[0,2]:.3f},{b[1,2]:.3f}]")
