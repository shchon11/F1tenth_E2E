"""The environment editor page: tools, undo, persistence and the hand-off to the driving page.

The page is exercised through the same signals the real viewport emits, with a stand-in viewport
that records what it was told to draw. No GL and no torch in this process: the geometry builder
thread runs the real numpy build, and the one test that needs a `Track` (validation) runs the
real `python -m f1sim.scene` subprocess.
"""
import json
import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytest.importorskip("PyQt5")
from PyQt5 import QtCore, QtWidgets                                             # noqa: E402

LEFT = int(QtCore.Qt.LeftButton)
NOMOD = int(QtCore.Qt.NoModifier)
from f1sim.viewer.console.theme import C as _C                                  # noqa: E402
C_WARN = _C["warn"]


@pytest.fixture(scope="module")
def qapp():
    _prev = os.environ.get("QT_QPA_PLATFORM")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from f1sim.viewer.console import app as A
    existing = QtWidgets.QApplication.instance()
    yield existing or A.create_app(["test"])
    if _prev is None:
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ["QT_QPA_PLATFORM"] = _prev


@pytest.fixture
def scenes_root(tmp_path, monkeypatch):
    root = tmp_path / "scenes"
    root.mkdir()
    monkeypatch.setenv("F1SIM_SCENES", str(root))
    return root


class FakeViewport(QtWidgets.QWidget):
    """Speaks the EditorViewport contract; draws nothing; remembers what it was told."""
    ground_pressed = QtCore.pyqtSignal(float, float, int, int, str)
    ground_moved = QtCore.pyqtSignal(float, float, int, int)
    ground_released = QtCore.pyqtSignal(float, float, int, int)
    hovered = QtCore.pyqtSignal(float, float, str)
    left_widget = QtCore.pyqtSignal()
    wheel_edit = QtCore.pyqtSignal(int, int)
    key_pressed = QtCore.pyqtSignal(int, int)
    camera_changed = QtCore.pyqtSignal()
    double_clicked = QtCore.pyqtSignal(float, float)
    upload_timed = QtCore.pyqtSignal(str, float)
    draw_timed = QtCore.pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.geometry = None
        self.geometries = 0
        self.pickables = []
        self.selection = []
        self.hover = None
        self.cursor = ("none", 0, 0, 0, 0)
        self.preview = None
        self.gizmo = None
        self.visibility = {}
        self.corner = ""
        self.framed = 0
        self.fills = {}
        self.handles = []
        self.lines = {}
        self.marquee = None

    def set_geometry(self, g):
        self.geometry = g
        self.geometries += 1

    def set_pickables(self, items):
        self.pickables = list(items)

    def hit_test(self, x, y):
        for pid, poly in reversed(self.pickables):
            if _inside(poly, x, y):
                return pid
        return ""

    def set_selection(self, polys, colour=None):
        self.selection = list(polys)

    def set_hover(self, poly):
        self.hover = poly

    def set_cursor(self, kind, x, y, radius=0.0, yaw=0.0, colour=None):
        self.cursor = (kind, x, y, radius, yaw)

    def set_preview(self, pts, width, closed=False, colour=None):
        self.preview = None if pts is None else (np.asarray(pts), width, closed)

    def set_gizmo(self, x, y, yaw, radius=0.4, on=True):
        self.gizmo = (x, y, yaw, radius) if on else None

    def set_fill(self, name, shapes, colour):
        self.fills[name] = (list(shapes or []), colour)

    def set_handles(self, items):
        self.handles = list(items or [])

    def set_lines(self, name, items):
        self.lines[name] = list(items or [])

    def set_marquee(self, rect):
        self.marquee = rect

    def handle_radius(self):
        return 0.08

    def set_layer_visibility(self, **kw):
        self.visibility = kw

    def set_corner_text(self, t):
        self.corner = t

    def frame_all(self):
        self.framed += 1

    def set_top_view(self, on):
        pass

    def look_at_point(self, x, y):
        self.looked_at = (x, y)

    def teardown(self):
        pass

    # helpers for the tests
    def press(self, x, y, mods=NOMOD, button=LEFT):
        self.ground_pressed.emit(x, y, button, mods, self.hit_test(x, y))

    def move(self, x, y, mods=NOMOD, buttons=LEFT):
        self.ground_moved.emit(x, y, buttons, mods)

    def release(self, x, y, mods=NOMOD, button=LEFT):
        self.ground_released.emit(x, y, button, mods)

    def hover_at(self, x, y):
        self.hovered.emit(x, y, self.hit_test(x, y))


def _inside(poly, x, y):
    poly = np.asarray(poly)
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _wait_geometry(page, vp, before, timeout=10.0):
    """Spin the event loop until the viewport received a geometry newer than `before`."""
    t0 = time.time()
    while vp.geometries <= before and time.time() - t0 < timeout:
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
        time.sleep(0.01)
    assert vp.geometries > before, "geometry build did not arrive"
    return vp.geometry


def _wait_until(pred, timeout=10.0):
    t0 = time.time()
    while not pred() and time.time() - t0 < timeout:
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
        time.sleep(0.01)
    assert pred(), "condition not reached"


@pytest.fixture
def page(qapp, scenes_root):
    from f1sim.viewer.console.env_editor import EnvEditorPage
    vp = FakeViewport()
    pg = EnvEditorPage(viewport_factory=lambda: vp)
    pg.set_active(True)
    yield pg, vp
    pg.shutdown()
    pg.deleteLater()


def _new_scene(page, vp, w=10.0, h=8.0):
    page.spin_w.setValue(w)
    page.spin_h.setValue(h)
    page.chk_border.setChecked(True)
    n = vp.geometries
    page.new_blank()
    _wait_geometry(page, vp, n)
    return page.state.doc


# ================================================================ document + tools
def test_new_blank_builds_geometry_and_frames(page):
    pg, vp = page
    doc = _new_scene(pg, vp)
    assert doc is not None and doc.shape[0] > 0
    assert vp.geometry is not None and vp.geometry.walls is not None     # the border wall is drawn
    assert pg.state.dirty
    assert "*" in pg.title_label.text()
    QtWidgets.QApplication.processEvents()
    assert vp.framed >= 1


def test_brush_stroke_is_one_undo_step(page):
    pg, vp = page
    doc = _new_scene(pg, vp)
    pg.set_tool("brush_duct")
    pg.spin_brush.setValue(0.3)
    before = int(doc.duct.sum())
    vp.press(3.0, 3.0)
    for x in np.linspace(3.0, 5.0, 12):
        vp.move(x, 3.0)
    vp.release(5.0, 3.0)
    painted = int(doc.duct.sum()) - before
    assert painted > 0
    assert doc.tall[doc.world_to_cell(4.0, 3.0)] == False                # noqa: E712 -- only the duct layer
    assert pg.state.can_undo() and pg.state.undo_label() == "덕트 칠하기"
    pg.undo()
    assert int(pg.state.doc.duct.sum()) == before
    pg.redo()
    assert int(pg.state.doc.duct.sum()) == before + painted


def test_brush_over_nothing_records_no_undo(page):
    pg, vp = page
    _new_scene(pg, vp)
    pg.set_tool("erase")
    vp.press(5.0, 4.0)                                    # the middle of the free canvas
    vp.release(5.0, 4.0)
    assert not pg.state.can_undo()


def test_rect_and_polygon_tools_fill_tall(page):
    pg, vp = page
    doc = _new_scene(pg, vp)
    pg.set_tool("rect")
    vp.press(2.0, 2.0)
    vp.move(3.0, 3.0)
    assert vp.preview is not None and vp.preview[2] is True
    vp.release(3.0, 3.0)
    assert doc.tall[doc.world_to_cell(2.5, 2.5)]
    assert vp.preview is None
    pg.set_tool("polygon")
    for x, y in [(6.0, 2.0), (8.0, 2.0), (7.0, 4.0)]:
        vp.press(x, y)
        vp.release(x, y)
    vp.double_clicked.emit(7.0, 4.0)
    assert doc.tall[doc.world_to_cell(7.0, 2.6)]
    assert not doc.tall[doc.world_to_cell(6.0, 3.8)]


def test_path_tool_makes_an_editable_duct_path(page):
    pg, vp = page
    doc = _new_scene(pg, vp)
    pg.set_tool("path_duct")
    assert "곡선" in pg.tool_hint.text()
    pg.spin_brush.setValue(0.15)                          # 0.3 m wide band
    pg.chk_snap.setChecked(False)
    for x, y in [(2.0, 5.0), (6.0, 5.0)]:
        vp.hover_at(x, y)
        assert vp.cursor[0] == "cross"
        vp.press(x, y)
        vp.release(x, y)
    # while drawing: the band decal and the vertex handles are shown
    vp.hover_at(7.0, 6.0)
    assert vp.fills["tool"][0] and vp.fills["tool"][0][0][0] == "band"
    assert len(vp.handles) == 3 and vp.handles[-1][3] == "hover"
    pg._on_key(int(QtCore.Qt.Key_Return), NOMOD)
    assert len(doc.paths) == 1 and doc.paths[0].kind == "duct" and len(doc.paths[0].points) == 2
    assert doc.duct[doc.world_to_cell(4.0, 5.0)]
    assert not doc.duct[doc.world_to_cell(4.0, 5.6)]
    assert not doc.tall[doc.world_to_cell(4.0, 5.0)]
    assert not doc.painted_duct[doc.world_to_cell(4.0, 5.0)], "a path is vector, not paint"
    assert pg.state.path_sel == (doc.paths[0].id, None)
    assert pg.path_form.isVisibleTo(pg)
    # the eraser cannot touch it; selecting + Delete removes it
    pg.set_tool("erase")
    vp.press(4.0, 5.0)
    vp.release(4.0, 5.0)
    assert doc.duct[doc.world_to_cell(4.0, 5.0)]
    pg.set_tool("select")
    vp.hover_at(4.0, 5.0)
    vp.press(4.0, 5.0)
    vp.release(4.0, 5.0)
    assert pg.state.path_sel[0] == doc.paths[0].id
    pg._on_key(int(QtCore.Qt.Key_Delete), NOMOD)
    assert doc.paths == [] and not doc.duct[doc.world_to_cell(4.0, 5.0)]
    pg.undo()
    assert len(pg.state.doc.paths) == 1


def test_track_path_with_a_curve_sets_the_centerline_and_two_hoses(page):
    pg, vp = page
    doc = _new_scene(pg, vp, w=14.0, h=10.0)
    pg.set_tool("path_track")
    pg.spin_lane.setValue(2.0)
    pg.chk_snap.setChecked(False)
    CTRL = int(QtCore.Qt.ControlModifier)
    for (x, y), mods in [((3.0, 3.0), NOMOD), ((11.0, 3.0), NOMOD), ((11.0, 7.0), CTRL), ((3.0, 7.0), CTRL)]:
        vp.press(x, y, mods=mods)
        vp.release(x, y, mods=mods)
    vp.double_clicked.emit(3.0, 7.0)
    assert len(doc.paths) == 1
    q = doc.paths[0]
    assert q.kind == "track" and q.closed and [p[2] for p in q.points] == [False, False, True, True]
    assert doc.centerline is not None and len(doc.centerline) > 100
    # hoses 1 m either side of the straight bottom edge, nothing on the line itself
    assert doc.duct[doc.world_to_cell(7.0, 2.0)] and doc.duct[doc.world_to_cell(7.0, 4.0)]
    assert not doc.duct[doc.world_to_cell(7.0, 3.0)]
    # the smooth top edge bulges: the curve passes outside the straight chord
    top = doc.centerline[:, 1].max()
    assert top > 7.05, top
    # dragging a vertex re-rasterises (on the live tick) and moves the hoses with it
    pg.set_tool("select")
    assert pg.state.path_sel == (q.id, None), "the path just drawn stays selected"
    vp.hover_at(11.0, 3.0)
    vp.press(11.0, 3.0)
    assert pg.state.path_sel == (q.id, 1)
    vp.move(12.0, 3.0)
    assert vp.fills["pathsel"][0], "a live decal is shown while dragging"
    vp.release(12.0, 3.0)
    assert q.points[1][:2] == (12.0, 3.0)
    assert doc.duct[doc.world_to_cell(11.5, 2.0)]
    assert pg.state.undo_label() == "꼭짓점 이동"
    # Tab toggles the selected vertex's smoothness; Alt+click inserts one
    pg._on_key(int(QtCore.Qt.Key_Tab), NOMOD)
    assert q.points[1][2] is True
    vp.press(7.5, 3.0, mods=int(QtCore.Qt.AltModifier))
    vp.release(7.5, 3.0, mods=int(QtCore.Qt.AltModifier))
    assert len(q.points) == 5
    # the validator keeps a track-path centerline rather than re-extracting it
    assert doc.track_path is not None


def test_brush_stroke_shows_a_live_decal_and_rebuilds_on_a_cadence(page):
    pg, vp = page
    doc = _new_scene(pg, vp)
    pg.set_tool("brush_tall")
    n0 = vp.geometries
    vp.press(3.0, 3.0)
    assert vp.fills["tool"][0] and vp.fills["tool"][1] != C_WARN, "wall strokes are not duct-coloured"
    for x in np.linspace(3.0, 5.0, 8):
        vp.move(x, 3.0)
    assert len(vp.fills["tool"][0][0][1]) >= 8            # the band follows the trail
    assert pg._live_timer.isActive()
    _wait_until(lambda: vp.geometries > n0, timeout=5.0)  # geometry grew while the button was down
    vp.release(5.0, 3.0)
    assert not pg._live_timer.isActive()
    assert "tool" not in vp.fills or not vp.fills["tool"][0]


def test_place_select_move_rotate_delete(page):
    pg, vp = page
    doc = _new_scene(pg, vp)
    # library row 0 is the cardboard box
    pg.lib_list.setCurrentRow(0)
    pg.set_tool("place")
    pg.chk_snap.setChecked(False)
    vp.hover_at(4.0, 4.0)
    assert vp.hover is not None and len(vp.hover) >= 4                   # footprint preview
    vp.press(4.0, 4.0)
    vp.release(4.0, 4.0)
    assert len(doc.props) == 1 and doc.props[0].style == "cardboard_box"
    pid = doc.props[0].id
    assert pg.state.selection == [pid]
    QtWidgets.QApplication.processEvents()
    assert vp.gizmo is not None and abs(vp.gizmo[0] - 4.0) < 1e-6

    pg.set_tool("select")
    vp.hover_at(4.0, 4.0)
    assert vp.hover is not None
    vp.press(4.05, 4.0)
    vp.move(5.05, 4.5)
    vp.release(5.05, 4.5)
    p = doc.get_prop(pid)
    assert abs(p.x - 5.0) < 1e-6 and abs(p.y - 4.5) < 1e-6
    assert pg.state.undo_label() == "이동"

    pg.rotate_selection(math.radians(90))
    assert abs(doc.get_prop(pid).yaw - math.pi / 2) < 1e-9
    pg._on_key(int(QtCore.Qt.Key_Delete), NOMOD)
    assert doc.props == [] and pg.state.selection == []
    pg.undo()
    assert len(pg.state.doc.props) == 1


def test_inspector_edits_commit(page):
    pg, vp = page
    doc = _new_scene(pg, vp)
    p = pg.state.commit("배치", lambda: doc.add_prop("steel_drum", 3.0, 3.0, 0.0, seed=2), "props")
    pg.state.select([p.id])
    QtWidgets.QApplication.processEvents()
    assert pg.insp_form.isVisibleTo(pg)
    pg.spin_x.setValue(6.5)
    assert abs(doc.get_prop(p.id).x - 6.5) < 1e-9
    assert "radius" in pg._dim_spins
    pg._dim_spins["radius"].setValue(0.2)
    assert abs(doc.get_prop(p.id).dims["radius"] - 0.2) < 1e-9
    fp = dict(doc.prop_footprints())[p.id]
    assert np.max(np.hypot(fp[:, 0] - 6.5, fp[:, 1] - 3.0)) > 0.19


def test_duplicate_and_bake(page):
    pg, vp = page
    doc = _new_scene(pg, vp)
    p = pg.state.commit("배치", lambda: doc.add_prop("wooden_crate", 3.0, 3.0, 0.0), "props")
    pg.state.select([p.id])
    pg.duplicate_selection()
    assert len(doc.props) == 2 and len(pg.state.selection) == 1 and pg.state.selection[0] != p.id
    pg.state.select([p.id])
    pg.bake_selection()
    assert len(doc.props) == 1
    assert doc.tall[doc.world_to_cell(3.0, 3.0)]


def test_save_lists_and_reopens(page, scenes_root):
    pg, vp = page
    doc = _new_scene(pg, vp)
    doc.name = "hall_a"
    pg.state.commit("배치", lambda: doc.add_prop("marker_post", 2.0, 2.0, 0.3), "props")
    got = []
    pg.scenes_changed.connect(lambda: got.append(1))
    assert pg.save()
    assert not pg.state.dirty and got
    assert os.path.isfile(scenes_root / "hall_a" / "scene.json")
    names = [pg.scene_list.item(i).data(QtCore.Qt.UserRole) for i in range(pg.scene_list.count())]
    assert names == ["hall_a"]
    pg.scene_list.setCurrentRow(0)
    pg.new_blank()                                  # a different, unsaved document...
    pg.state.mark_saved()                           # ...that we pretend is clean, so no dialog
    assert pg.open_scene("hall_a")
    assert pg.state.doc.name == "hall_a" and len(pg.state.doc.props) == 1
    pg.duplicate_selected()
    names = sorted(pg.scene_list.item(i).data(QtCore.Qt.UserRole) for i in range(pg.scene_list.count()))
    assert names == ["hall_a", "hall_a_copy"]


def test_import_mesh_asset_and_place(page, tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pg, vp = page
    doc = _new_scene(pg, vp)
    doc.name = "with_mesh"
    m = trimesh.creation.cylinder(radius=0.3, height=0.5)
    m.apply_translation([0, 0, 0.25])
    path = tmp_path / "tire_stack.stl"
    m.export(str(path))
    aid = pg.import_asset(str(path))
    assert aid is not None
    assert pg.current_tool() == "place"
    assert pg.library_item() == ("asset", aid)
    pg.chk_snap.setChecked(False)
    vp.press(5.0, 4.0)
    vp.release(5.0, 4.0)
    assert len(doc.props) == 1 and doc.props[0].style == "mesh" and doc.props[0].asset == aid
    _wait_until(lambda: vp.geometry is not None and vp.geometry.props)
    props = vp.geometry.props
    assert props and sum(int(b["n_tris"]) for b in props) > 0
    fp = dict(doc.prop_footprints())[doc.props[0].id]
    assert 0.28 < np.max(np.hypot(fp[:, 0] - 5.0, fp[:, 1] - 4.0)) < 0.33


def test_validate_extracts_centerline_and_hands_off(page, scenes_root):
    """Real subprocess: `python -m f1sim.scene validate` imports torch and finds the loop."""
    pg, vp = page
    doc = _new_scene(pg, vp, w=12.0, h=9.0)
    doc.name = "loop"
    # a rectangular loop: outer border already there; add an island in the middle
    pg.state.commit("island", lambda: doc.paint_rect("tall", 3.0, 3.0, 9.0, 6.0, True), "grid")
    handed = []
    pg.drive_requested.connect(handed.append)
    pg._drive()
    t0 = time.time()
    while pg._job is not None and time.time() - t0 < 120:
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
        time.sleep(0.02)
    assert pg._job is None, "validation subprocess did not finish"
    assert pg._validation is not None, pg.status_line.text()
    assert pg._validation.get("ok"), pg._validation
    assert pg.state.doc.centerline is not None and len(pg.state.doc.centerline) > 50
    assert handed == ["scene:loop"]
    saved = json.load(open(scenes_root / "loop" / "scene.json"))
    assert saved["centerline"] is not None


# ================================================================ window wiring
def test_window_has_editor_mode_and_hand_off(qapp, scenes_root, monkeypatch):
    from f1sim.viewer.console import window as W
    from f1sim.viewer.console.catalog import MapCatalog
    monkeypatch.setattr("f1sim.viewer.console.catalog.list_scenes",
                        lambda: [{"name": "hall_a", "dir": "", "modified": 0, "props": 0, "shape": (1, 1)}])
    win = W.ConsoleWindow()
    try:
        assert win.mode_buttons.button("edit") is not None
        win.set_mode("edit")
        assert win.centre_stack.currentWidget() is win.editor
        assert not win.left_panel.isVisibleTo(win)
        win.set_maps(MapCatalog(groups={"기본 평가셋": ["gen:competition:0"]}, ready=True))
        assert win.maps.groups.get(W.ConsoleWindow.SCENE_GROUP) == ["scene:hall_a"]
        # the editor's picker got the catalogue too, without the scene group
        assert win.editor.map_picker.select("gen:competition:0")
        assert not win.editor.map_picker.select("scene:hall_a")
        assert win.editor.catalog_choice() == "gen:competition:0"
        assert win.map_group.itemData(0) == W.ConsoleWindow.SCENE_GROUP
        win._drive_from_editor("scene:hall_a")
        assert win.current_mode() == "drive"
        assert win.current_config().map_name == "scene:hall_a"
        assert win.map_list.selected() == "scene:hall_a"
    finally:
        win.editor.shutdown()
        win.deleteLater()


def test_driving_shortcuts_are_inert_on_editor_page(qapp, scenes_root):
    from f1sim.viewer.console import window as W
    win = W.ConsoleWindow()
    try:
        win.set_mode("edit")
        shots = []
        win._on_screenshot = lambda: shots.append(1)
        for s in win.findChildren(QtWidgets.QShortcut):
            if s.key().toString() == "S":
                s.activated.emit()
        assert shots == []
        win.set_mode("drive")
        for s in win.findChildren(QtWidgets.QShortcut):
            if s.key().toString() == "S":
                s.activated.emit()
        assert shots == [1]
    finally:
        win.editor.shutdown()
        win.deleteLater()


def test_painted_duct_becomes_an_editable_path(page):
    pg, vp = page
    doc = _new_scene(pg, vp, w=12.0, h=9.0)
    pg.set_tool("brush_duct")
    vp.press(2.0, 3.0)
    for x in np.linspace(2.0, 9.0, 30):
        vp.move(x, 3.0 + 0.7 * math.sin((x - 2.0) / 1.5))
    vp.release(9.0, 3.0)
    assert doc.paths == [] and doc.painted_duct.any()
    cells_before = int(doc.duct.sum())
    pg.vectorize("duct")
    assert len(doc.paths) == 1 and doc.paths[0].kind == "duct" and not doc.paths[0].closed
    assert not doc.painted_duct.any(), "the paint is replaced by the path"
    assert 0.8 < doc.duct.sum() / cells_before < 1.25
    assert any(p[2] for p in doc.paths[0].points), "a wavy stroke gets curve vertices"
    assert pg.state.path_sel == (doc.paths[0].id, None) and pg.current_tool() == "select"
    # and it is now editable like any path
    q = doc.paths[0]
    vi = len(q.points) // 2
    x, y, _ = q.points[vi]
    vp.hover_at(x, y)
    vp.press(x, y)
    vp.move(x, y + 1.0)
    vp.release(x, y + 1.0)
    assert abs(q.points[vi][1] - (y + 1.0)) < 0.06       # grid snap is on
    pg.undo()                         # the drag
    pg.undo()                         # the conversion
    assert pg.state.doc.paths == [] and pg.state.doc.painted_duct.any()


def test_generator_card_makes_an_editable_track(page):
    pg, vp = page
    pg.spin_gen_seed.setValue(4)
    pg.combo_gen_size.setCurrentIndex(0)                      # 소
    chk, cnt = pg._gen_feature_widgets["hairpin"]
    chk.setChecked(True)
    cnt.setValue(1)
    pg._gen_feature_widgets["cone_slalom"][0].setChecked(False)
    n = vp.geometries
    assert pg.generate_track()
    _wait_geometry(pg, vp, n)
    doc = pg.state.doc
    assert doc.track_path is not None and doc.centerline is not None
    assert doc.source["generator"] == "trackgen" and doc.source["features"].count("hairpin") == 1
    assert doc.name.startswith("rand_4")
    assert pg.state.dirty
    seed_before = pg.spin_gen_seed.value()
    pg.state.mark_saved()                                     # no discard dialog
    pg._regenerate()
    assert pg.spin_gen_seed.value() == seed_before + 1 and pg.state.doc.name.startswith("rand_5")
    # the generated path is editable like any track path
    q = pg.state.doc.track_path
    pg.state.select_path(q.id, 0)
    pg.toggle_vertex_smooth()
    assert pg.state.undo_label() == "직선↔곡선"
