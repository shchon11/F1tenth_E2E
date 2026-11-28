"""bag -> car calibration record + place profile + human-readable summary.

    python -m f1sim.bag_profile <bag folder> [--car <profile name or yaml>] --out <folder>
                                [--no-map] [--no-lidar] [--no-tyre] [--name NAME] [--setup setup.yaml]

Outputs in <out>:
  car_calibration.yaml   values that belong to the car setup (servo map + clamp window, ERPM gain, sensor units,
                         delay, trim, steering response, accel / brake limits, current limits, LiDAR mount) --
                         every entry with value, err, unit, source, status, how; loadable by f1sim.profiles.loader
  place_profile.yaml     values that belong to the place (floor grip lower bound, tyre-on-floor stiffness, LiDAR
                         dropout by range, the map and its scale check)
  summary.md             the same in Korean, with figures (figs/), how test segments were found, what was
                         substituted, the setup.yaml comparison, what this bag could not measure (re-record /
                         measure by hand) and the thresholds used
  bag_profile.json       everything, machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import bagio as B, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m f1sim.bag_profile",
                                 description="실차 bag 하나 -> 차 보정 기록 + 장소 프로필 + 요약")
    ap.add_argument("bag", help="bag 폴더 (metadata.yaml 과 .db3 가 있는 곳)")
    ap.add_argument("--car", default=None, help="비교·초기값으로 쓸 차 프로필 (f1sim/f1sim/profiles 의 이름 또는 yaml 경로)")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--setup", default=None, help="차 설정 기록 setup.yaml (기본: bag 폴더 안, 없으면 그 옆 폴더의 setup.yaml)")
    ap.add_argument("--name", default=None, help="출력에 쓸 이름 (기본: bag 폴더 이름 앞 15자)")
    ap.add_argument("--no-map", action="store_true", help="bag_map 지도 추출을 건너뜀")
    ap.add_argument("--no-lidar", action="store_true", help="LiDAR 통계를 건너뜀")
    ap.add_argument("--no-tyre", action="store_true", help="타이어 값 맞춤(가장 느린 단계)을 건너뜀")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    try:
        res = report.run(a.bag, out=a.out, car=a.car, name=a.name, do_map=not a.no_map, do_lidar=not a.no_lidar,
                         do_tyre=not a.no_tyre, setup=a.setup,
                         log=(lambda *x: None) if a.quiet else (lambda *x: print(*x, flush=True)))
    except B.BagFormatError as e:
        print(f"[오류] 읽을 수 없는 bag 형식\n{e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"[오류] {e}", file=sys.stderr)
        return 2
    if not a.quiet:
        print(f"[끝] {res['out']} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
