"""Turn a SLAM toolbox map (yaml + pgm/png) into a track this simulator can train on.

`Track.from_ros_map` has always read that pair, and `maps.load("/path/to/map.yaml")` has always
worked -- but only as far as the occupancy grid. What it came back without is a **centerline**, and
without one there is no raceline, so no teacher, so no DAgger and no lap-time reward: the map loaded
and then failed at the first thing anyone wanted to do with it. This module is the missing step, and
the checks that say whether the map is fit to drive before an hour of training finds out for you.

    python -m f1sim.slam_map ~/maps/venue.yaml                     # report only
    python -m f1sim.slam_map ~/maps/venue.yaml --install venue     # + add it to the catalogue
    python -m f1sim.slam_map ~/maps/venue.png --seed-xy 1.2 0.4 --preview out.png

`--install <id>` registers the map in the map registry (`map_registry`: one folder per map under
`$F1SIM_MAPS` or `~/.f1sim/maps`, the pair as built + map.json with its loader keywords), after
which it is an ordinary catalogue entry -- `--tracks real:venue` (`user:venue` is the same map), the
console's map list (`real/<slug>`), modifiers (`real:venue~rev`, `real:venue+warp3`), obstacles
(`real/venue#line:3`) -- and nothing downstream knows it came from SLAM. No code edit per map.
Without it, the track is still usable by path (`--tracks /path/to/venue.yaml`), which is the
quicker way to try one. `--from-bag <name>_bag.json` (written by `f1sim.bag_map`) carries the lane
seed, the driving direction and the provenance over from the recording.

What the defaults assume, and when to override:

* **The boundary is a duct hose** (`--boundary duct`), which is what an F1TENTH venue is taped out
  with: the occupied shell next to free space is ~0.33 m tall, so a beam that clears it sees the
  floor beyond. A track walled by buildings or lab furniture is `--boundary wall`.
* **The lane is the free region you drove**, so scan spray leaking through doorways is cut
  (`keep_region`), and if the map has more than one free region the largest is taken -- pass
  `--seed-xy X Y` (a point you know is on the lane, in map metres) when that is the wrong one.
* **The track is a closed loop.** The centerline is the curve equidistant from the outer boundary
  and the infield, which only exists if the lane closes. A map with a gap in it fails here with a
  message saying so, and the fix is a better map, not a flag.
"""
from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

from .track import Track

#: Where `--install` puts maps (`map_registry.maps_dir()`). Outside the repository on purpose: a venue
#: map is a recording, it belongs to whoever made it, and a training run that depends on one should
#: say where it came from.
USER_MAPS = os.path.abspath(os.path.expanduser(
    os.environ.get("F1SIM_MAPS") or os.path.join("~", ".f1sim", "maps")))

#: Car geometry the report judges the map against (`params.Config().vehicle`), so "does it fit" is
#: answered in the same numbers the simulator drives with.
CAR_WIDTH = 0.31

#: A lane this much narrower than the car plus a margin on each side is reported as impassable.
MIN_MARGIN = 0.05


@dataclass
class MapReport:
    """What the import found. Every field is measured, none is a preference."""
    name: str
    yaml_path: str
    image_path: str
    resolution_m: float
    size_m: Tuple[float, float]
    free_area_m2: float
    lane_length_m: float
    half_width_min_m: float
    half_width_median_m: float
    raceline_clearance_m: float
    raceline_lap_s: Optional[float]
    fits: bool
    problems: Tuple[str, ...]

    def text(self) -> str:
        w, h = self.size_m
        lines = [
            f"map           {self.name}   ({os.path.basename(self.image_path)} + "
            f"{os.path.basename(self.yaml_path)})",
            f"grid          {w:.1f} x {h:.1f} m at {1000 * self.resolution_m:.0f} mm/cell, "
            f"{self.free_area_m2:.0f} m^2 of free space",
            f"lane          {self.lane_length_m:.1f} m round, half-width "
            f"{self.half_width_median_m:.2f} m median / {self.half_width_min_m:.2f} m at its tightest",
            f"raceline      {self.raceline_clearance_m:.2f} m clearance"
            + (f", {self.raceline_lap_s:.2f} s point-mass lap" if self.raceline_lap_s else " (not built)"),
        ]
        if self.problems:
            lines += ["", "problems:"] + [f"  - {p}" for p in self.problems]
        else:
            lines += ["", f"ready to drive: the car ({CAR_WIDTH:.2f} m wide) fits everywhere with "
                          f"{self.half_width_min_m - CAR_WIDTH / 2:.2f} m to spare at the tightest point"]
        return "\n".join(lines)


def resolve_paths(path: str) -> Tuple[str, str]:
    """(yaml, image) from whatever the user has in hand: the yaml, the image, or the directory.

    SLAM toolbox writes the pair together and people reach for either half of it, so both work.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        ys = sorted(f for f in os.listdir(path) if f.endswith((".yaml", ".yml")))
        if not ys:
            raise FileNotFoundError(f"no map yaml in {path}")
        if len(ys) > 1:
            raise FileNotFoundError(f"{len(ys)} map yamls in {path}: name the one you mean ({', '.join(ys[:4])})")
        path = os.path.join(path, ys[0])
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith((".yaml", ".yml")):
        yaml_path = path
    else:                                            # an image: the yaml is its sibling
        base = os.path.splitext(path)[0]
        for ext in (".yaml", ".yml"):
            if os.path.exists(base + ext):
                yaml_path = base + ext
                break
        else:
            raise FileNotFoundError(
                f"{os.path.basename(path)} has no {os.path.basename(base)}.yaml beside it. SLAM "
                f"toolbox writes both; the yaml carries the resolution and origin, which the image "
                f"does not, so the map cannot be placed in metres without it.")
    import yaml as _yaml
    with open(yaml_path) as f:
        meta = _yaml.safe_load(f)
    img = meta.get("image", "")
    if not os.path.isabs(img):
        img = os.path.join(os.path.dirname(yaml_path), img)
    if not os.path.exists(img):                      # the yaml may name .png while map_saver wrote .pgm
        base = os.path.splitext(img)[0]
        for ext in (".pgm", ".png", ".jpg", ".jpeg"):
            if os.path.exists(base + ext):
                img = base + ext
                break
        else:
            raise FileNotFoundError(f"{yaml_path} points at {meta.get('image')!r}, which is not next to it")
    return yaml_path, img


def load_map(path: str, *, boundary: str = "duct", seed_xy=None, min_clearance: float = 0.35,
             keep_region: bool = True, despeckle_m2: float = 0.05, name: Optional[str] = None,
             n_points: int = 800, **kw) -> Track:
    """A SLAM map as a drivable `Track`, centerline and all.

    `min_clearance` is how far the centerline is kept off anything solid while it is extracted; it
    is not the racing margin (the raceline optimiser has its own) but a floor under the mid-lane
    curve, so a doorway narrower than this is treated as not part of the lane.
    """
    yaml_path, img_path = resolve_paths(path)
    name = name or os.path.splitext(os.path.basename(yaml_path))[0]
    track = Track.from_ros_map(yaml_path, name=name, boundary=boundary, keep_region=keep_region,
                               seed_xy=seed_xy, despeckle_m2=despeckle_m2, **kw)
    if track.centerline is None:
        try:
            track.centerline_from_free_space(n_points=n_points, min_clearance=min_clearance,
                                             seed_xy=seed_xy)
        except ValueError as exc:
            raise ValueError(
                f"{name}: could not trace a lane through this map ({exc}). A centerline is the curve "
                f"equidistant from the outer boundary and the infield, so the lane has to close into "
                f"a loop with something in the middle. Check the preview (--preview out.png): the "
                f"usual causes are a gap where the loop did not close, scan spray joining the lane to "
                f"a room next door (--seed-xy on the lane fixes that), or a lane narrower than "
                f"--min-clearance {min_clearance:.2f} m.") from exc
    return track


def inspect(track: Track, *, build_raceline: bool = True) -> MapReport:
    """Measure the map against the car. Nothing here is fatal; the report says what it found."""
    from scipy import ndimage
    from .raceline import track_widths, Raceline

    def raceline_clearance(t, rl) -> float:
        """Closest the optimised line comes to anything solid. Same measure `learn.common` drops a
        track on, restated here so importing a map does not pull in torch."""
        edt = ndimage.distance_transform_edt(~t.occupancy).astype(np.float32) * t.resolution
        j = np.clip(((rl.xy[:, 0] - t.origin[0]) / t.resolution).astype(int), 0, edt.shape[1] - 1)
        i = np.clip(((rl.xy[:, 1] - t.origin[1]) / t.resolution).astype(int), 0, edt.shape[0] - 1)
        return float(edt[i, j].min())

    c = track.centerline
    wl, wr = track_widths(track, c)
    half = np.minimum(wl, wr)
    length = float(np.linalg.norm(np.roll(c, -1, 0) - c, axis=1).sum())
    problems = []
    lap = None
    rl_clear = float("nan")
    if build_raceline:
        try:
            rl = Raceline.build(track)
            lap = float(rl.lap_time)
            rl_clear = float(raceline_clearance(track, rl))
            if rl_clear < CAR_WIDTH / 2 + MIN_MARGIN:
                problems.append(f"the optimised raceline passes within {rl_clear:.2f} m of a wall, "
                                f"less than the car's half-width plus {MIN_MARGIN:.2f} m: it would be "
                                f"dropped from a training set (learn.common.load_tracks)")
        except Exception as exc:                     # a raceline is a solve; it can fail on its own
            problems.append(f"the raceline optimiser failed on this map: {exc}")
    tight = float(half.min())
    if tight < CAR_WIDTH / 2 + MIN_MARGIN:
        at = c[int(np.argmin(half))]
        problems.append(f"the lane is {2 * tight:.2f} m wide at ({at[0]:.1f}, {at[1]:.1f}) m, which the "
                        f"{CAR_WIDTH:.2f} m car cannot pass with a margin")
    if length < 10.0:
        problems.append(f"the traced lane is only {length:.1f} m round, which is shorter than any real "
                        f"track: it is probably a loop through scan noise rather than the lane")
    H, W = track.occupancy.shape
    yaml_path = img_path = ""              # the caller knows where it came from; `inspect` only measures
    return MapReport(
        name=track.name, yaml_path=yaml_path, image_path=img_path, resolution_m=track.resolution,
        size_m=(W * track.resolution, H * track.resolution),
        free_area_m2=float((~track.occupancy).sum()) * track.resolution ** 2,
        lane_length_m=length, half_width_min_m=tight, half_width_median_m=float(np.median(half)),
        raceline_clearance_m=rl_clear, raceline_lap_s=lap, fits=not problems,
        problems=tuple(problems))


def preview(track: Track, out_path: str, *, raceline: bool = True, path_xy=None, seed_xy=None,
            title: Optional[str] = None) -> str:
    """A PNG of what was imported: walls (duct hose / tall), the traced lane centre with arrows in
    the driving direction, its start, and optionally the raceline, the car's own driven path and the
    lane seed.

    The one check that cannot be done in numbers. A map that traces a plausible loop through the
    wrong part of the building reports perfectly good widths. `raceline` builds the minimum-time
    line (20-70 min on a new map): off for a quick look.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
        if f in names:
            plt.rcParams["font.family"] = f
            break
    H, W = track.occupancy.shape
    ext = [track.origin[0], track.origin[0] + W * track.resolution,
           track.origin[1], track.origin[1] + H * track.resolution]
    img = np.ones((H, W, 3))
    img[track.occupancy] = (0.82, 0.82, 0.82)          # solid for the car; the LiDAR sees floor there
    duct = getattr(track, "duct", None)
    tall = getattr(track, "tall", None)
    if duct is not None and tall is not None:
        img[tall] = (0.25, 0.25, 0.25)
        img[duct] = (0.95, 0.55, 0.15)
    else:
        img[track.occupancy] = (0.25, 0.25, 0.25)
    fig, ax = plt.subplots(figsize=(8, 8 * H / max(W, 1) * 0.9 + 1.2), dpi=110)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    if path_xy is not None and len(path_xy):
        ax.plot(path_xy[:, 0], path_xy[:, 1], "-", color="#2da44e", lw=0.6, alpha=0.6, label="차가 실제로 달린 길 (bag)")
    if track.centerline is not None:
        c = np.vstack([track.centerline, track.centerline[:1]])
        L = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
        ax.plot(c[:, 0], c[:, 1], "-", color="#2f81f7", lw=1.4, label=f"차선 가운데 ({L:.1f} m 한 바퀴)")
        n = len(track.centerline)
        for i in range(0, n, max(1, n // 14)):
            p, q = track.centerline[i], track.centerline[(i + 3) % n]
            ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="#2f81f7", lw=1.2))
        ax.plot(c[0, 0], c[0, 1], "o", color="#2f81f7", ms=7, label="센터라인 시작점")
    if seed_xy is not None:
        ax.plot(seed_xy[0], seed_xy[1], "x", color="#cf222e", ms=9, mew=2, label="차선 위 기준점")
    if raceline:
        try:
            from .raceline import Raceline
            rl = Raceline.build(track)
            xy = np.vstack([rl.xy, rl.xy[:1]])
            ax.plot(xy[:, 0], xy[:, 1], "-", color="#e5534b", lw=1.6,
                    label=f"raceline ({rl.lap_time:.2f} s)")
        except Exception:
            pass
    from matplotlib.patches import Patch
    h, _ = ax.get_legend_handles_labels()
    h += [Patch(color=(0.95, 0.55, 0.15), label="덕트 호스 (LiDAR가 넘어 봄)"), Patch(color=(0.25, 0.25, 0.25), label="높은 벽 / 밖"),
          Patch(color=(0.82, 0.82, 0.82), label="못 들어감 (LiDAR엔 바닥)")]
    ax.legend(handles=h, loc="lower right", fontsize=7)
    ax.set_title(title or track.name, fontsize=9); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def install(yaml_path: str, track_id: str, *, boundary: str = "duct", min_clearance: float = 0.35, seed_xy=None,
            despeckle_m2: float = 0.05, keep_region: bool = True, direction: str = "ccw", note: str = "",
            display: str = "", source=None, out_dir: Optional[str] = None, overwrite: bool = False) -> str:
    """Register the map as `real:<track_id>` (also `user:<track_id>`): the pair is copied as built
    into the map registry (`map_registry`, one folder per map) with the keywords it was checked with,
    and from then on loads exactly like a `maps.REAL` entry -- no code edit per map.

    The map itself rather than the cleaned grid: `keep_region` and despeckling are re-applied by
    the loader every time, as for the hand-written entries, so what is registered is what was
    measured and the cleaning can be changed in map.json without re-importing."""
    from . import map_registry
    meta = dict(display=display or track_id, note=note or f"{track_id}, from a recording", boundary=boundary,
                min_clearance=float(min_clearance), seed_xy=[float(v) for v in seed_xy] if seed_xy is not None else None,
                despeckle_m2=float(despeckle_m2), direction=direction,
                from_ros_map=dict(keep_region=bool(keep_region)), source=source or {})
    path = map_registry.install(yaml_path, track_id, meta, root=out_dir, overwrite=overwrite)
    from . import maps, tracks
    maps.refresh_registered()
    tracks.refresh_registered(force=True)
    return path


def user_map_yaml(track_id: str, out_dir: Optional[str] = None) -> Optional[str]:
    """The registered map's yaml, or None."""
    from . import map_registry
    e = map_registry.get(track_id, out_dir)
    return e["yaml"] if e else None


def user_map_ids(out_dir: Optional[str] = None) -> Tuple[str, ...]:
    from . import map_registry
    return tuple(e["id"] for e in map_registry.entries(out_dir))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Import a SLAM toolbox map (yaml + pgm/png) as a simulator track.")
    ap.add_argument("path", help="the map yaml, its image, or the directory holding them")
    ap.add_argument("--install", metavar="ID", default=None,
                    help="also register it as real:<ID> (= user:<ID>) in the map registry (F1SIM_MAPS, ~/.f1sim/maps)")
    ap.add_argument("--overwrite", action="store_true", help="replace a registered map of the same id")
    ap.add_argument("--preview", metavar="PNG", default=None, help="write a picture of what was imported")
    ap.add_argument("--no-raceline", action="store_true",
                    help="skip the minimum-time raceline in the report and the picture (20-70 min on a new map)")
    ap.add_argument("--from-bag", metavar="JSON", default=None,
                    help="bag_map's <name>_bag.json: the lane seed, the driving direction, the driven path and the "
                         "provenance come from the recording (a scan-built map also gets --despeckle-m2 0.15)")
    ap.add_argument("--boundary", choices=["duct", "wall"], default="duct",
                    help="what the occupied cells are: a duct hose a beam can see over (default), or tall walls")
    ap.add_argument("--seed-xy", type=float, nargs=2, metavar=("X", "Y"), default=None,
                    help="[m] a point on the lane, when the map has more than one free region")
    ap.add_argument("--direction", choices=["ccw", "cw"], default=None, help="driving direction (default ccw, or the bag's)")
    ap.add_argument("--min-clearance", type=float, default=0.35,
                    help="[m] how far the traced centre stays off anything solid (default 0.35)")
    ap.add_argument("--no-keep-region", action="store_true",
                    help="keep every free region, including scan spray leaking out of the hall")
    ap.add_argument("--despeckle-m2", type=float, default=None,
                    help="[m^2] occupied specks smaller than this are scan noise, not obstacles (default 0.05)")
    ap.add_argument("--name", default=None, help="track name in the report (default: the yaml's stem)")
    ap.add_argument("--display", default=None, help="name in the console's list (default: the id)")
    ap.add_argument("--note", default=None, help="one line on where it came from")
    a = ap.parse_args(argv)

    bag = {}
    if a.from_bag:
        import json
        bag = json.load(open(a.from_bag))
    seed_xy = a.seed_xy if a.seed_xy is not None else bag.get("seed_xy")
    direction = a.direction or bag.get("direction") or "ccw"
    # a map built from scans keeps whatever stood on the track that day (a box, a parked car): blobs up
    # to ~0.4 x 0.4 m standing free in the lane are dropped; the team's /map was cleaned by hand already
    despeckle = a.despeckle_m2 if a.despeckle_m2 is not None else (0.15 if bag.get("source") == "slam" else 0.05)
    track = load_map(a.path, boundary=a.boundary, seed_xy=seed_xy, min_clearance=a.min_clearance,
                     keep_region=not a.no_keep_region, despeckle_m2=despeckle, name=a.name or a.install)
    if track.centerline is not None and direction == "cw":
        track.centerline = track.centerline[::-1].copy()
    yaml_path, img_path = resolve_paths(a.path)
    report = inspect(track, build_raceline=not a.no_raceline)
    report.yaml_path, report.image_path = yaml_path, img_path
    print(report.text())
    if a.preview:
        path_xy = None
        if bag.get("path_csv") and os.path.exists(bag["path_csv"]):
            path_xy = np.loadtxt(bag["path_csv"], delimiter=",").reshape(-1, 2)
        title = (f"{a.install or track.name}: 출처 {bag.get('source', '?')}, 진행 방향 "
                 f"{'반시계' if direction == 'ccw' else '시계'} (화살표), 한 바퀴 {report.lane_length_m:.1f} m, "
                 f"가장 좁은 곳 {2 * report.half_width_min_m:.2f} m") if bag else None
        print(f"\npreview      {preview(track, a.preview, raceline=not a.no_raceline, path_xy=path_xy, seed_xy=seed_xy, title=title)}")
    if a.install:
        src = {k: bag.get(k) for k in ("bag", "source", "window", "bagmap_scale", "compare", "slam") if k in bag}
        path = install(yaml_path, a.install, boundary=a.boundary, min_clearance=a.min_clearance, seed_xy=seed_xy,
                       despeckle_m2=despeckle, keep_region=not a.no_keep_region, direction=direction,
                       note=a.note or (f"{a.install}, from {os.path.basename(str(bag.get('bag', '')))} ({bag.get('source')})" if bag else ""),
                       display=a.display or a.install, source=src, overwrite=a.overwrite)
        print(f"\ninstalled    {path}\n"
              f"             use it as   --tracks real:{a.install}   (= user:{a.install}; "
              f"also real:{a.install}~rev, real:{a.install}+warp3)")
    elif report.fits:
        print(f"\n             use it as   --tracks {yaml_path}   "
              f"(or --install <id> to catalogue it)")
    raise SystemExit(0 if report.fits else 1)


if __name__ == "__main__":
    main()
