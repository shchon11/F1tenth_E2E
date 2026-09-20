"""Screenshot of the real console on its 환경 page, headless, on CPU.

    cd f1sim && xvfb-run -a -s "-screen 0 1920x1200x24" env LIBGL_ALWAYS_SOFTWARE=1 QT_QPA_PLATFORM=xcb \
        F1SIM_SCENES=/tmp/f1sim_scenes_capture python ../work/env-editor/capture_editor.py

Nothing is mocked: the actual `ConsoleWindow`, the actual `EnvEditorPage` with its `EditorViewport`,
the actual `SceneDoc` and the actual geometry build. The scene is built through the page's own tool
handlers (the same signals the mouse produces), a real mesh is imported from a trimesh-generated
STL, and the window is photographed the way a compositor sees it. No worker is started: the editor
page needs none.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import time

import numpy as np

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
W, H = 1920, 1200


def main():
    from PyQt5 import QtCore, QtWidgets
    import trimesh

    from f1sim.viewer.console import app as console_app
    from f1sim.viewer.console.window import ConsoleWindow

    os.makedirs(OUT_DIR, exist_ok=True)
    app = console_app.create_app(["capture"])
    win = ConsoleWindow()
    win.resize(W, H)
    win.show()
    win.set_mode("edit")
    page = win.editor
    vp = page.viewport

    builds = {"n": 0}
    page._builder.built.connect(lambda *_: builds.__setitem__("n", builds["n"] + 1))

    def pump(seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.01)

    def wait_build(n_before, timeout=20.0):
        t0 = time.time()
        while page._last_geometry is None or builds["n"] <= n_before:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.01)
            if time.time() - t0 > timeout:
                raise SystemExit("geometry build did not arrive")

    pump(1.0)
    # the real catalogue, as the worker would send it (this script may import torch; the console must not)
    from f1sim.viewer.console.catalog import MapCatalog
    from f1sim.viewer.sim_worker import SimWorker
    win.set_maps(MapCatalog(groups=SimWorker.map_catalog(SimWorker.__new__(SimWorker)), ready=True))
    page.map_picker.select("real:korea_2026_competition")

    # -- a hall: a closed *track path* (straight sides, curved hairpin) drawn with the track tool,
    # plus a duct path chicane, all through the page's own handlers
    page.spin_w.setValue(18.0)
    page.spin_h.setValue(12.0)
    page.chk_border.setChecked(True)
    page.new_blank()
    page.state.doc.name = "capture_hall"
    page.chk_snap.setChecked(False)
    LEFT, NOMOD = int(QtCore.Qt.LeftButton), int(QtCore.Qt.NoModifier)
    CTRL = int(QtCore.Qt.ControlModifier)

    def path(tool, pts, close=True):
        page.set_tool(tool)
        page.chk_close.setChecked(close)
        for x, y, sm in pts:
            mods = CTRL if sm else NOMOD
            page._on_hover(x, y, "")
            page._on_press(x, y, LEFT, mods, "")
            page._on_release(x, y, LEFT, mods)
        page._on_key(int(QtCore.Qt.Key_Return), NOMOD)

    page.spin_lane.setValue(1.7)
    path("path_track", [(3.0, 3.0, False), (12.0, 3.0, False), (15.0, 4.5, True), (15.0, 7.5, True),
                        (12.0, 9.0, False), (8.5, 9.0, False), (8.5, 6.2, True), (6.0, 5.0, True),
                        (3.0, 6.5, True), (2.5, 9.0, True), (5.0, 9.6, True), (7.0, 9.0, False)][:0] or
         [(3.0, 3.0, False), (12.0, 3.0, False), (15.0, 4.5, True), (15.0, 7.5, True), (12.0, 9.0, False),
          (3.0, 9.0, False), (2.0, 6.0, True)])
    page.spin_brush.setValue(0.17)
    path("path_duct", [(8.0, 5.2, False), (9.5, 6.0, True), (11.0, 5.2, False)], close=False)
    # a tall pillar with the rectangle tool
    page.set_tool("rect")
    page._on_press(12.6, 6.6, LEFT, NOMOD, "")
    page._on_move(13.2, 7.2, LEFT, NOMOD)
    page._on_release(13.2, 7.2, LEFT, NOMOD)
    # built-in props
    page.set_tool("place")
    for row, (x, y, yaw) in {0: (6.0, 3.0, 0.3), 2: (13.4, 3.4, 0.0), 4: (9.5, 9.0, 1.2)}.items():
        page.lib_list.setCurrentRow(row)
        page._tools["place"].yaw = yaw
        page._on_hover(x, y, "")
        page._on_press(x, y, LEFT, NOMOD, "")
        page._on_release(x, y, LEFT, NOMOD)
    # an imported mesh: a tyre stack generated with trimesh, saved first (assets need a scene dir)
    page.save()
    m = trimesh.creation.cylinder(radius=0.28, height=0.16, sections=48)
    stack = trimesh.util.concatenate([m.copy().apply_translation([0, 0, 0.08 + 0.17 * i]) for i in range(3)])
    stl = os.path.join(OUT_DIR, "tire_stack.stl")
    stack.export(stl)
    aid = page.import_asset(stl)
    assert aid, page.status_line.text()
    for x, y in [(9.8, 3.0), (5.0, 9.0)]:
        page._on_hover(x, y, "")
        page._on_press(x, y, LEFT, NOMOD, "")
        page._on_release(x, y, LEFT, NOMOD)
    page.save()
    pump(0.5)
    n = builds["n"]
    page._rebuild_full()
    wait_build(n - 1)
    pump(0.5)
    # select the track path at its hairpin vertex: handles, control polygon and lane edges show
    doc = page.state.doc
    page.set_tool("select")
    track = doc.track_path
    page.state.select_path(track.id, 2)
    page._on_hover(13.4, 3.4, doc.props[1].id)
    page.refresh_overlays()
    vp.set_top_view(False)
    vp.az, vp.el, vp.dist = math.radians(-62), math.radians(40), 19.5
    vp.pivot = np.array([9.0, 6.0])
    vp._camera_moved() if hasattr(vp, "_camera_moved") else None
    vp.update()
    pump(1.5)

    shots = {}
    pix = app.primaryScreen().grabWindow(int(win.winId()))
    path = os.path.join(OUT_DIR, "f1tenth-environment-editor.png")
    pix.save(path)
    shots["window"] = path
    # the viewport alone through the widget's own screenshot path
    vpath = os.path.join(OUT_DIR, "f1tenth-environment-editor-viewport.png")
    vp.grab_png(vpath) if hasattr(vp, "grab_png") else vp.grabFramebuffer().save(vpath)
    shots["viewport"] = vpath
    # top view, a wall brush stroke still in progress: the decal under the brush and the wall
    # already rebuilt behind it
    page.state.select_path(None, None)
    page.set_tool("brush_tall")
    page.spin_brush.setValue(0.2)
    page._on_press(4.0, 4.6, LEFT, NOMOD, "")
    for i, x in enumerate(np.linspace(4.0, 7.0, 16)):
        page._on_move(x, 4.6 + 0.6 * math.sin(i / 4.0), LEFT, NOMOD)
    vp.set_top_view(True)
    vp.frame_all()
    pump(1.2)
    tpath = os.path.join(OUT_DIR, "f1tenth-environment-editor-top.png")
    app.primaryScreen().grabWindow(int(win.winId())).save(tpath)
    shots["top"] = tpath
    page._on_release(7.0, 4.6, LEFT, NOMOD)
    page.save()

    gl = getattr(vp, "gl_info", None)
    manifest = {
        "captured": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scene_dir": doc.dir, "props": [(p.id, p.style, p.asset, round(p.x, 2), round(p.y, 2)) for p in doc.props],
        "assets": [(a.id, a.name, a.tris, a.size) for a in doc.assets],
        "shape": list(doc.shape), "resolution": doc.resolution,
        "renderer": str(gl) if gl else None,
        "software_gl": os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1",
        "files": {k: {"path": v, "sha256": hashlib.sha256(open(v, "rb").read()).hexdigest()} for k, v in shots.items()},
        "status_line": page.status_line.text(),
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    page.shutdown()
    win.editor.viewport.teardown()
    win.viewport.teardown()
    app.quit()


if __name__ == "__main__":
    main()
