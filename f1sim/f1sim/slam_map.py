"""Turn a SLAM toolbox map (yaml + pgm/png) into a track this simulator can train on.

`Track.from_ros_map` has always read that pair, and `maps.load("/path/to/map.yaml")` has always
worked -- but only as far as the occupancy grid. What it came back without is a **centerline**, and
without one there is no raceline, so no teacher, so no DAgger and no lap-time reward: the map loaded
and then failed at the first thing anyone wanted to do with it. This module is the missing step, and
the checks that say whether the map is fit to drive before an hour of training finds out for you.

    python -m f1sim.slam_map ~/maps/venue.yaml                     # report only
    python -m f1sim.slam_map ~/maps/venue.yaml --install venue     # + add it to the catalogue
    python -m f1sim.slam_map ~/maps/venue.png --seed-xy 1.2 0.4 --preview out.png

`--install <id>` writes a cleaned copy into the user map directory, after which the map is an
ordinary catalogue entry -- `--tracks user:venue`, the console's map list, `real:`-style modifiers
(`user:venue~rev`), obstacles (`user:venue#line:3`) -- and nothing downstream knows it came from
SLAM. Without it, the track is still usable by path (`--tracks /path/to/venue.yaml`), which is the
quicker way to try one.

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

#: Where `--install` puts maps. Outside the repository on purpose: a venue map is a recording, it
#: belongs to whoever made it, and a training run that depends on one should say where it came from.
USER_MAPS = os.path.join(os.path.expanduser("~"), ".f1sim", "maps")

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
            f"grid          {w:.1f} x {h:.1f} m at {100 * self.resolution_m:.0f} mm/cell, "
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


def preview(track: Track, out_path: str, *, raceline: bool = True) -> str:
    """A PNG of what was imported: the grid, the traced lane, and the raceline on it.

    The one check that cannot be done in numbers. A map that traces a plausible loop through the
    wrong part of the building reports perfectly good widths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = track.occupancy.shape
    ext = [track.origin[0], track.origin[0] + W * track.resolution,
           track.origin[1], track.origin[1] + H * track.resolution]
    fig, ax = plt.subplots(figsize=(10, 10 * H / max(W, 1)), dpi=110)
    ax.imshow(~track.occupancy, cmap="gray", origin="lower", extent=ext, interpolation="nearest")
    if track.centerline is not None:
        c = np.vstack([track.centerline, track.centerline[:1]])
        ax.plot(c[:, 0], c[:, 1], "-", color="#2f81f7", lw=1.6, label="lane centre")
        ax.plot(c[0, 0], c[0, 1], "o", color="#2f81f7", ms=6)
    if raceline:
        try:
            from .raceline import Raceline
            rl = Raceline.build(track)
            xy = np.vstack([rl.xy, rl.xy[:1]])
            ax.plot(xy[:, 0], xy[:, 1], "-", color="#e5534b", lw=1.6,
                    label=f"raceline ({rl.lap_time:.2f} s)")
        except Exception:
            pass
    ax.set_title(track.name); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="lower right", fontsize=8); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def install(track: Track, track_id: str, *, out_dir: str = USER_MAPS, overwrite: bool = False) -> str:
    """Write the cleaned map (+ its centerline) into the user map directory as `user:<track_id>`.

    The cleaned grid, not a copy of the original: `keep_region`, despeckling and any downsampling
    are what the track was traced on, so what gets catalogued is what was measured. The centerline
    travels as a CSV beside it, which is also the file to hand-edit if the trace needs a correction.
    """
    d = os.path.join(out_dir, track_id)
    if os.path.exists(d) and not overwrite:
        raise FileExistsError(f"{d} exists; pass --overwrite to replace it")
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    return track.save_ros_map(d, name=track_id)


def user_map_yaml(track_id: str, out_dir: str = USER_MAPS) -> Optional[str]:
    """The installed map's yaml, or None. Used by `maps.load` to resolve `user:<id>`."""
    p = os.path.join(out_dir, track_id, track_id + ".yaml")
    return p if os.path.exists(p) else None


def user_map_ids(out_dir: str = USER_MAPS) -> Tuple[str, ...]:
    if not os.path.isdir(out_dir):
        return ()
    return tuple(sorted(n for n in os.listdir(out_dir) if user_map_yaml(n, out_dir)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Import a SLAM toolbox map (yaml + pgm/png) as a simulator track.")
    ap.add_argument("path", help="the map yaml, its image, or the directory holding them")
    ap.add_argument("--install", metavar="ID", default=None,
                    help="also catalogue it as user:<ID> (usable as --tracks user:<ID>)")
    ap.add_argument("--overwrite", action="store_true", help="replace an installed map of the same id")
    ap.add_argument("--preview", metavar="PNG", default=None, help="write a picture of what was imported")
    ap.add_argument("--boundary", choices=["duct", "wall"], default="duct",
                    help="what the occupied cells are: a duct hose a beam can see over (default), or tall walls")
    ap.add_argument("--seed-xy", type=float, nargs=2, metavar=("X", "Y"), default=None,
                    help="[m] a point on the lane, when the map has more than one free region")
    ap.add_argument("--min-clearance", type=float, default=0.35,
                    help="[m] how far the traced centre stays off anything solid (default 0.35)")
    ap.add_argument("--no-keep-region", action="store_true",
                    help="keep every free region, including scan spray leaking out of the hall")
    ap.add_argument("--despeckle-m2", type=float, default=0.05,
                    help="[m^2] occupied specks smaller than this are scan noise, not obstacles")
    ap.add_argument("--name", default=None, help="track name in the report (default: the yaml's stem)")
    a = ap.parse_args()

    track = load_map(a.path, boundary=a.boundary, seed_xy=a.seed_xy, min_clearance=a.min_clearance,
                     keep_region=not a.no_keep_region, despeckle_m2=a.despeckle_m2, name=a.name)
    yaml_path, img_path = resolve_paths(a.path)
    report = inspect(track)
    report.yaml_path, report.image_path = yaml_path, img_path
    print(report.text())
    if a.preview:
        print(f"\npreview      {preview(track, a.preview)}")
    if a.install:
        path = install(track, a.install, overwrite=a.overwrite)
        print(f"\ninstalled    {path}\n"
              f"             use it as   --tracks user:{a.install}   "
              f"(also user:{a.install}~rev, user:{a.install}#line:3)")
    elif report.fits:
        print(f"\n             use it as   --tracks {yaml_path}   "
              f"(or --install <id> to catalogue it)")
    raise SystemExit(0 if report.fits else 1)


if __name__ == "__main__":
    main()
