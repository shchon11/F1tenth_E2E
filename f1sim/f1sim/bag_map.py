"""실차 ROS 2 bag -> 지도 (map_server yaml + pgm) + 출처 기록 json.

    python -m f1sim.bag_map <bag 폴더> --name <지도이름> --out <출력 폴더> [--source auto|map|slam] [--compare]

bag 폴더는 bag 하나(metadata.yaml 이 있는 폴더)여도 되고, bag 여러 개를 담은 폴더여도 된다(그때는 /scan 과
/odom 이 있는 것 중 가장 긴 bag 하나를 쓴다).

지도 출처 (--source):
  map   bag 안의 /map (차의 스택이 위치 추정에 쓰던 지도)을 그대로 꺼낸다.
  slam  /scan + /odom 으로 직접 만든다 (`scan_mapper.build_map`, 스캔-지도 맞춤 + 레이 캐스팅).
  auto  (기본) /map 이 있고 크기 검사를 통과하면 map, 아니면 slam.
        크기 검사: 파티클 필터 위치(/pf/pose/odom)에서 스캔을 /map 벽에 맞추며 거리 배율 하나를 더 맞춘다.
        실제 크기 지도는 1.00, 20 % 작게 그려진 지도는 0.80 (Korea 2025 팀 지도가 이랬음). 1 에서 3 % 넘게
        벗어나면 /map 을 버리고 스캔으로 만든다.
--compare: 둘 다 만들어 맞대 보고 (자유 공간 IoU, 차선 IoU, 벽 일치, 크기 배율) 겹친 그림을 남긴다.

출력 (<출력 폴더>):
  <이름>.yaml / .pgm      고른 지도 (등록·학습에 쓰는 것)
  <이름>_bag.json         어디서 왔는지: bag, 시간 구간, 출처, 크기 검사, 주행 방향, 차선 위 한 점(seed_xy), 걸린 시간
  <이름>_slam.*, <이름>_bagmap.*, <이름>_compare.png   (만든 경우)

ROS 2 Humble 의 rosbag2_py 가 필요하다: bash 에서 `cd ~ && source ~/f1tenth_ws/activate.sh` 뒤에 실행.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import scan_mapper as SM

SCAN_TOPICS = ("/scan",)
ODOM_TOPICS = ("/odom", "/vesc/odom")
POSE_TOPICS = ("/pf/pose/odom", "/amcl_pose", "/car_state/odom")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- which bag
def bag_meta(path: str) -> Optional[dict]:
    """metadata.yaml as {topic: count}, duration [s]; None when `path` is not a bag."""
    import yaml
    p = os.path.join(path, "metadata.yaml")
    if not os.path.isfile(p):
        return None
    info = yaml.safe_load(open(p))["rosbag2_bagfile_information"]
    topics = {t["topic_metadata"]["name"]: int(t["message_count"]) for t in info["topics_with_message_count"]}
    return dict(path=path, topics=topics, duration=info["duration"]["nanoseconds"] / 1e9)


def find_bags(path: str) -> List[dict]:
    path = os.path.abspath(os.path.expanduser(path))
    m = bag_meta(path)
    if m is not None:
        return [m]
    out = []
    for d in sorted(os.listdir(path)):
        m = bag_meta(os.path.join(path, d))
        if m is not None:
            out.append(m)
    return out


def candidate_bags(path: str) -> List[dict]:
    """Bags with a LiDAR and wheel odometry, longest first (a /map in one is a bonus, not a requirement)."""
    bags = find_bags(path)
    if not bags:
        raise SystemExit(f"bag 이 없습니다: {path} (metadata.yaml 이 있는 폴더, 또는 그런 폴더들을 담은 폴더)")

    def first(m, names):
        return next((t for t in names if m["topics"].get(t, 0) > 0), None)

    ok = [m for m in bags if first(m, SCAN_TOPICS) and first(m, ODOM_TOPICS)]
    if not ok:
        raise SystemExit(f"{path}: /scan 과 /odom 이 둘 다 있는 bag 이 없습니다")
    for m in ok:
        m["scan_topic"], m["odom_topic"] = first(m, SCAN_TOPICS), first(m, ODOM_TOPICS)
        m["pose_topic"] = first(m, POSE_TOPICS)
        m["has_map"] = m["topics"].get("/map", 0) > 0
        m["n_bags"] = len(bags)
    return sorted(ok, key=lambda m: -m["duration"])


def pick_bag(path: str) -> dict:
    return candidate_bags(path)[0]


# ---------------------------------------------------------------- reading
def _reader(path: str, topics):
    import rosbag2_py
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    return r, types


def _stamp(msg, bag_ns: int) -> float:
    s = msg.header.stamp
    t = s.sec + 1e-9 * s.nanosec
    return t if t > 1.0 else bag_ns * 1e-9


def _yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def read_light(bag: dict):
    """First pass: odometry, the map-frame pose, the static transforms and /map (all small)."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    topics = [bag["odom_topic"], "/tf_static", "/map"] + ([bag["pose_topic"]] if bag.get("pose_topic") else [])
    r, types = _reader(bag["path"], topics)
    cls = {}
    odom, pose, statics, the_map = [], [], [], None
    while r.has_next():
        topic, data, ns = r.read_next()
        if topic == "/map" and the_map is not None:
            continue
        if topic not in cls:
            cls[topic] = get_message(types[topic])
        m = deserialize_message(data, cls[topic])
        if topic == bag["odom_topic"]:
            p, q, tw = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist
            odom.append((_stamp(m, ns), p.x, p.y, _yaw(q), tw.linear.x, tw.angular.z))
        elif topic == bag.get("pose_topic"):
            p, q = m.pose.pose.position, m.pose.pose.orientation
            pose.append((_stamp(m, ns), p.x, p.y, _yaw(q)))
        elif topic == "/tf_static":
            for tr in m.transforms:
                t, q = tr.transform.translation, tr.transform.rotation
                statics.append((tr.header.frame_id, tr.child_frame_id, t.x, t.y, _yaw(q)))
        elif topic == "/map":
            the_map = m
    odom = np.array(odom, float).reshape(-1, 6)
    odom = odom[np.argsort(odom[:, 0], kind="stable")]
    pose = np.array(pose, float).reshape(-1, 4)
    if len(pose):
        pose = pose[np.argsort(pose[:, 0], kind="stable")]
    return odom, pose, statics, the_map


def laser_offset(statics, child: str, base: str = "base_link") -> Tuple[Tuple[float, float, float], str]:
    """The scanner's pose on the car from /tf_static (a chain of static transforms is followed)."""
    parent = {c: (p, np.array([x, y, yaw])) for p, c, x, y, yaw in statics}
    T = np.zeros(3)
    node, hops = child, 0
    while node != base and node in parent and hops < 10:
        p, rel = parent[node]
        T = SM.compose(rel, T)
        node, hops = p, hops + 1
    if node != base:
        return (0.0, 0.0, 0.0), f"/tf_static 에 {base} -> {child} 가 없어 스캐너를 차 기준점에 둠"
    return (float(T[0]), float(T[1]), float(T[2])), f"/tf_static {base} -> {child}: ({T[0]:.3f}, {T[1]:.3f}) m, {math.degrees(T[2]):.1f} deg"


def read_scans(bag: dict, t0: float, t1: float) -> Tuple[SM.Scans, str]:
    """Second pass: the scans whose stamp falls in [t0, t1]."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r, types = _reader(bag["path"], [bag["scan_topic"]])
    cls = get_message(types[bag["scan_topic"]])
    ts, rows, meta = [], [], None
    while r.has_next():
        topic, data, ns = r.read_next()
        if ns * 1e-9 < t0 - 5.0:          # the bag clock and the header clock agree to well under 5 s
            continue
        if ns * 1e-9 > t1 + 5.0:
            break
        m = deserialize_message(data, cls)
        t = _stamp(m, ns)
        if not (t0 <= t <= t1):
            continue
        if meta is None:
            meta = (m.header.frame_id, m.angle_min, m.angle_increment, m.time_increment, m.range_min, m.range_max, len(m.ranges))
        if len(m.ranges) != meta[6]:
            continue
        ts.append(t)
        rows.append(np.asarray(m.ranges, np.float32))
    if meta is None:
        raise SystemExit(f"{bag['path']}: {t0:.1f}-{t1:.1f} s 에 스캔이 없습니다")
    frame, amin, ainc, tinc, rmin, rmax, _ = meta
    order = np.argsort(ts)
    sc = SM.Scans(t=np.asarray(ts)[order], ranges=np.stack(rows)[order], angle_min=float(amin), angle_inc=float(ainc),
                  time_inc=float(max(tinc, 0.0)), range_min=float(rmin), range_max=float(rmax))
    return sc, frame


def map_msg_to_grid(m) -> SM.GridMap:
    g = np.asarray(m.data, np.int8).reshape(m.info.height, m.info.width)
    o = m.info.origin.position
    return SM.GridMap(free=(g == 0), occ=(g >= 50), x0=float(o.x), y0=float(o.y), res=float(m.info.resolution))


def grid_image(gm: SM.GridMap) -> np.ndarray:
    img = np.full(gm.free.shape, 205, np.uint8)
    img[gm.free] = 254
    img[gm.occ] = 0
    return np.flipud(img)


def seed_on_lane(gm: SM.GridMap, path_xy: np.ndarray) -> Tuple[float, float]:
    """The point of the driven path with the most room around it, in the map's frame: a seed
    for `keep_region` that is on the lane by construction."""
    from scipy import ndimage
    edt = ndimage.distance_transform_edt(gm.free) * gm.res
    j = np.clip(((path_xy[:, 0] - gm.x0) / gm.res).astype(int), 0, edt.shape[1] - 1)
    i = np.clip(((path_xy[:, 1] - gm.y0) / gm.res).astype(int), 0, edt.shape[0] - 1)
    k = int(np.argmax(edt[i, j]))
    return float(path_xy[k, 0]), float(path_xy[k, 1])


def overlay_png(ours: SM.GridMap, ref: SM.GridMap, T, scale, path_ref: Optional[np.ndarray], out: str, title: str):
    """Both maps on the reference grid: reference walls blue, ours red, where they agree purple."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _use_korean_font(plt)
    f_o, o_o = SM.sample_classes(ours, ref, T, scale)
    on = SM.GridMap(free=f_o, occ=o_o, x0=ref.x0, y0=ref.y0, res=ref.res)
    from scipy import ndimage
    so, sr = on.surface(), ref.surface()
    so_d = ndimage.binary_dilation(so, iterations=1); sr_d = ndimage.binary_dilation(sr, iterations=1)
    H, W = ref.free.shape
    img = np.ones((H, W, 3))
    img[ref.free] = (0.93, 0.93, 0.93)
    img[f_o & ~ref.free] = (1.0, 0.85, 0.85)                  # free only in ours
    img[ref.free & ~f_o] = (0.85, 0.88, 1.0)                  # free only in the reference
    img[sr_d] = (0.15, 0.35, 0.95)
    img[so_d] = (0.9, 0.15, 0.15)
    img[so_d & sr_d] = (0.55, 0.1, 0.65)
    ext = [ref.x0, ref.x0 + W * ref.res, ref.y0, ref.y0 + H * ref.res]
    fig, ax = plt.subplots(figsize=(7, 7 * H / max(W, 1) * 0.75 + 1.5), dpi=120)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    if path_ref is not None and len(path_ref):
        ax.plot(path_ref[:, 0], path_ref[:, 1], "-", color="#2da44e", lw=0.6, alpha=0.7)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=(0.15, 0.35, 0.95), label="bag /map 벽"), Patch(color=(0.9, 0.15, 0.15), label="스캔으로 만든 벽"),
                       Patch(color=(0.55, 0.1, 0.65), label="둘이 겹침 (±5 cm)"), Patch(color="#2da44e", label="주행 경로")],
              loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def _use_korean_font(plt):
    """Hangul in titles and legends: the first Korean-capable font matplotlib knows."""
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans CJK SC"):
        if f in names:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="실차 ROS 2 bag -> 지도 (yaml + pgm) + 출처 json")
    ap.add_argument("bag", help="bag 폴더 (metadata.yaml 이 있는 곳) 또는 bag 들을 담은 폴더")
    ap.add_argument("--name", required=True, help="지도 이름 (파일 이름이 됨)")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--source", choices=["auto", "map", "slam"], default="auto")
    ap.add_argument("--compare", action="store_true", help="/map 이 있으면 스캔으로도 만들어 맞대 봄")
    ap.add_argument("--max-s", type=float, default=150.0, help="읽을 주행 시간 [s] (기본 150; 스캔 지도는 깨끗한 3바퀴에서 멈춤)")
    ap.add_argument("--start-s", type=float, default=None, help="bag 시작부터 몇 초 뒤부터 쓸지 (기본: 차가 움직이기 시작한 때)")
    ap.add_argument("--scale-tol", type=float, default=0.03, help="/map 크기 검사 허용 오차 (기본 0.03 = 3 %%)")
    a = ap.parse_args(argv)
    t_all = time.time()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for bag in candidate_bags(a.bag):
        # the longest bag that actually drove: a recording of the car standing still is long too
        odom_a, pose_a, statics, map_msg = read_light(bag)
        path_m = float(np.linalg.norm(np.diff(odom_a[:, 1:3], axis=0), axis=1).sum()) if len(odom_a) > 1 else 0.0
        if path_m >= 30.0:
            break
        log(f"[bag] {os.path.basename(bag['path'])}: 오도메트리로 {path_m:.0f} m 만 움직여 건너뜀")
    else:
        raise SystemExit(f"{a.bag}: 30 m 넘게 달린 bag 이 없습니다")
    log(f"[bag] {bag['path']} ({bag['duration']:.0f} s, 폴더 안 bag {bag['n_bags']}개 중 가장 긴 주행) "
        f"스캔 {bag['scan_topic']} / 오도메트리 {bag['odom_topic']} / 지도 위치 {bag.get('pose_topic') or '없음'} / "
        f"/map {'있음' if bag['has_map'] else '없음'}")
    rec: Dict = dict(bag=bag["path"], bag_seconds=bag["duration"], bags_in_folder=bag["n_bags"],
                     topics=dict(scan=bag["scan_topic"], odom=bag["odom_topic"], pose=bag.get("pose_topic")))
    rec["read_light_s"] = round(time.time() - t0, 1)
    odom = SM.Odom(t=odom_a[:, 0], x=odom_a[:, 1], y=odom_a[:, 2], yaw=odom_a[:, 3], v=odom_a[:, 4], w=odom_a[:, 5])
    moving = np.nonzero(np.abs(odom.v) > 0.5)[0]
    if a.start_s is not None:
        ts = odom.t[0] + a.start_s
    else:
        ts = (odom.t[moving[0]] - 0.5) if len(moving) else odom.t[0]
    te = min(ts + a.max_s, odom.t[-1])
    rec["window"] = dict(start_s=round(ts - odom.t[0], 2), end_s=round(te - odom.t[0], 2))
    want_map = bag["has_map"] and (a.source in ("auto", "map") or a.compare)
    want_slam = a.source == "slam" or (a.source == "auto" and not bag["has_map"]) or a.compare
    if a.source == "map" and not bag["has_map"]:
        raise SystemExit("--source map: 이 bag 에 /map 이 없습니다")
    t0 = time.time()
    scans, frame = read_scans(bag, ts, te)
    lx, why = laser_offset(statics, frame)
    scans.laser = lx
    rec["read_scans_s"] = round(time.time() - t0, 1)
    rec["scans"] = int(len(scans.t)); rec["laser"] = why
    log(f"[읽기] 스캔 {len(scans.t)}개 ({te - ts:.0f} s 구간), {why}, 한 줄 {scans.ranges.shape[1]}개 빔, "
        f"빔 간격 {scans.time_inc * 1e6:.1f} us (왜곡 보정 {'함' if scans.time_inc > 0 else '안 함'}) | {rec['read_light_s'] + rec['read_scans_s']:.0f} s")

    chosen, chosen_path = None, None
    bagmap = None
    if want_map:
        bagmap = map_msg_to_grid(map_msg)
        SM.write_ros_map(grid_image(bagmap), bagmap.x0, bagmap.y0, bagmap.res, a.out, f"{a.name}_bagmap",
                         comment="bag /map as published by the car's stack")
        rec["bagmap"] = dict(size_m=[bagmap.free.shape[1] * bagmap.res, bagmap.free.shape[0] * bagmap.res],
                             res=bagmap.res, free_cells=int(bagmap.free.sum()), unknown_cells=int((~bagmap.free & ~bagmap.occ).sum()))
        if len(pose_a):
            t0 = time.time()
            pz = np.stack([np.interp(scans.t, pose_a[:, 0], pose_a[:, 1]), np.interp(scans.t, pose_a[:, 0], pose_a[:, 2]),
                           SM.wrap(np.interp(scans.t, pose_a[:, 0], np.unwrap(pose_a[:, 3])))], 1)
            sc = SM.scale_from_scans(bagmap, scans, pz, odom)
            sc["seconds"] = round(time.time() - t0, 1)
            rec["bagmap_scale"] = sc
            ok = abs(sc["scale"] - 1.0) <= a.scale_tol if np.isfinite(sc["scale"]) else False
            log(f"[크기 검사] bag /map 대 LiDAR 거리 배율 {sc['scale']:.3f} (사분위 폭 {sc['scale_iqr']:.3f}, 스캔 {sc['scans_used']}개) "
                f"-> {'통과' if ok else '실패'} (허용 ±{a.scale_tol:.0%}) | {sc['seconds']:.0f} s")
            in_win = (pose_a[:, 0] >= ts) & (pose_a[:, 0] <= te)
            path_map = pose_a[in_win, 1:3] if in_win.sum() > 10 else pose_a[:, 1:3]
        else:
            ok = False
            path_map = None
            rec["bagmap_scale"] = dict(scale=float("nan"), note="지도 좌표 위치 토픽이 없어 크기 검사를 못 함")
            log("[크기 검사] 지도 좌표 위치 토픽(/pf/pose/odom 등)이 없어 /map 크기를 검사할 수 없음 -> 스캔으로 만듦")
        if a.source == "map" or (a.source == "auto" and ok):
            chosen, chosen_path = bagmap, path_map
            rec["source"] = "map"
        elif a.source == "auto":
            want_slam = True
    if want_slam:
        t0 = time.time()
        res = SM.build_map(scans, odom, log=log)
        st = res.stats
        img = res.image()
        SM.write_ros_map(img, res.x0, res.y0, res.res, a.out, f"{a.name}_slam",
                         comment=f"built from {os.path.basename(bag['path'])} /scan + /odom by f1sim.scan_mapper")
        slam = SM.GridMap.load(os.path.join(a.out, f"{a.name}_slam.yaml"))
        np.savetxt(os.path.join(a.out, f"{a.name}_slam_path.csv"), np.c_[res.t - odom.t[0], res.poses],
                   delimiter=",", header="t_s (from the bag's first odometry),x_m,y_m,yaw_rad", comments="# ", fmt="%.4f")
        rec["slam"] = dict(st, seconds=round(time.time() - t0, 1), stretches=res.stretches,
                           size_m=[img.shape[1] * res.res, img.shape[0] * res.res])
        log(f"[스캔 지도] 읽은 스캔 {st['scans_read']}개 중 {st['stretch_s'][0]:.0f}-{st['stretch_s'][1]:.0f} s 구간 {st['scans']}개 "
            f"-> 키프레임 {st['keyframes']}개, 표면에 닿은 점 중앙값 {st['inliers_median']:.0%} (하위 5 % {st['inliers_p05']:.0%}), "
            f"다시 찾기 {st['fallbacks']}번, 경로 {st['path_m']:.1f} m (바퀴 오도메트리 {st['odom_path_m']:.1f} m, 비 {st['odom_over_map']:.3f}) "
            f"| {time.time() - t0:.0f} s")
        if chosen is None:
            chosen, chosen_path = slam, res.poses[:, :2]
            rec["source"] = "slam"
        if bagmap is not None:
            t0 = time.time()
            guess = None
            if len(pose_a):
                pz = np.stack([np.interp(res.t, pose_a[:, 0], pose_a[:, 1]), np.interp(res.t, pose_a[:, 0], pose_a[:, 2]),
                               SM.wrap(np.interp(res.t, pose_a[:, 0], np.unwrap(pose_a[:, 3])))], 1)
                Ts = SM.compose(pz, SM.inverse(res.poses))
                guess = np.array([np.median(Ts[:, 0]), np.median(Ts[:, 1]),
                                  math.atan2(np.median(np.sin(Ts[:, 2])), np.median(np.cos(Ts[:, 2])))])
            T, s, inl = SM.align(slam, bagmap, guess=guess)
            T0, s0, inl0 = SM.align(slam, bagmap, guess=guess, fit_scale=False)
            cmp_ = SM.compare(slam, bagmap, T0, 1.0, seed_xy_ref=None)
            path_ref = SM.transform(T0, res.poses[:, :2])
            cmp_.update(scale_similarity=float(s), align_inliers=float(inl0), align_seconds=round(time.time() - t0, 1),
                        align_from="파티클 필터 위치" if guess is not None else "전 방향 탐색")
            rec["compare"] = cmp_
            title = (f"{a.name}: 스캔 지도 vs bag /map — 자유공간 IoU {cmp_['free_iou']:.3f}, 차선 IoU {cmp_['lane_iou']:.3f}, "
                     f"벽 일치(10 cm) {cmp_['wall_f1_10cm']:.3f}, 벽 거리 평균 {cmp_['wall_mean_cm']:.1f} cm, 크기 배율 {s:.3f}")
            png = overlay_png(slam, bagmap, T0, 1.0, path_ref, os.path.join(a.out, f"{a.name}_compare.png"), title)
            rec["compare"]["png"] = png
            log(f"[비교] {title.split(': ', 1)[1]} | 그림 {png}")
    # the chosen map, its seed on the lane and the direction the car drove it
    SM.write_ros_map(grid_image(chosen), chosen.x0, chosen.y0, chosen.res, a.out, a.name,
                     comment=f"{rec['source']} from {os.path.basename(bag['path'])}")
    if chosen_path is None or len(chosen_path) < 10:
        chosen_path = chosen.cells_xy(chosen.free)[:1]
        rec["direction"] = "ccw"
    else:
        rec["direction"] = SM.driving_direction(chosen_path)
    rec["seed_xy"] = list(seed_on_lane(chosen, chosen_path))
    rec["path_csv"] = os.path.join(os.path.abspath(a.out), f"{a.name}_path.csv")
    np.savetxt(rec["path_csv"], chosen_path[:: max(1, len(chosen_path) // 4000)], delimiter=",", fmt="%.3f",
               header="x_m,y_m: the driven path in this map's frame", comments="# ")
    rec["yaml"] = os.path.join(os.path.abspath(a.out), f"{a.name}.yaml")
    rec["seconds"] = round(time.time() - t_all, 1)
    json.dump(rec, open(os.path.join(a.out, f"{a.name}_bag.json"), "w"), indent=1, ensure_ascii=False,
              default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else str(o))
    log(f"[끝] 출처 {rec['source']}, 주행 방향 {'반시계' if rec['direction'] == 'ccw' else '시계'}, 차선 위 점 "
        f"({rec['seed_xy'][0]:.2f}, {rec['seed_xy'][1]:.2f}) -> {rec['yaml']} | 전체 {rec['seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
