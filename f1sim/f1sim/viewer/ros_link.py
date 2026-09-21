"""ROS 2 link for the console's simulation worker.

The console renders one environment in its own window; this makes that same environment visible
to ROS 2 -- the sensors of one car go out as the topics the real car publishes, `/drive` can take
that car over, and the whole scene (every car, the props, the racing line, the plan) goes out as
visualisation topics so rviz shows what the console shows.

Two halves, split on purpose:

* **Message builders** (module-level functions) are numpy in, plain Python structures out. They
  know nothing about rclpy and are what the unit tests exercise.
* **`RosLink`** owns the node: publishers, the `/drive` subscription, the reset service, a spin
  thread. It imports rclpy lazily, in `__init__`, so the worker can be imported -- and every other
  session mode can run -- without ROS on the path.

Threads: the worker's *simulation* thread calls `publish()` and `command()`; the link's own
executor thread runs the subscription and service callbacks. The shared state between them is two
scalars behind a lock. rclpy publishers may be called from a thread other than the executor's.

What the ROS side sees (car `car`, env index, default 0):

    /scan               sensor_msgs/LaserScan       frame `laser`   noisy scan
    /odom               nav_msgs/Odometry           odom -> base_link, VESC dead-reckoning (drifts)
    /ego_racecar/odom   nav_msgs/Odometry           map frame, ground truth (simulation only)
    /sensors/imu        sensor_msgs/Imu             VESC IMU: attitude estimate + latest sample
    /sensors/imu/raw    sensor_msgs/Imu             every raw sample with its own stamp
    /map                nav_msgs/OccupancyGrid      latched
    /f1sim/collision    std_msgs/Bool
    /f1sim/raceline     nav_msgs/Path               latched; /f1sim/raceline_speed is the matching
                                                    std_msgs/Float32MultiArray of target speeds
    /f1sim/centerline   nav_msgs/Path               latched
    /f1sim/viz/cars     visualization_msgs/MarkerArray   every car: the ROS car, its rivals, the rest
    /f1sim/viz/props    visualization_msgs/MarkerArray   latched; placed obstacles as wireframe prisms
    /f1sim/viz/raceline visualization_msgs/Marker   latched; the line coloured by target speed
    /f1sim/viz/plan     visualization_msgs/Marker   the plan tracker's reference, when the policy
                                                    drives in plan mode
    TF                  map -> odom (ground-truth closure), odom -> base_link, base_link -> laser,
                        base_link -> imu
    /drive              ackermann_msgs/AckermannDriveStamped   (in) steering [rad], speed [m/s]
    /f1sim/reset        std_srvs/Empty              (in) reset every car, like the console's button
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

MODES = ("off", "publish", "drive")

#: How long a `/drive` command stays in force. Past it the car coasts to a stop with its last
#: steering angle -- what the bridge does, and roughly what the real car's mux does when a source
#: goes quiet.
CMD_TIMEOUT_S = 0.5

#: Marker colours (r, g, b, a). The ROS car is the one whose sensors are on the wire.
COLOR_ROS_CAR = (0.10, 0.85, 0.30, 0.95)
COLOR_RIVAL = (0.95, 0.35, 0.20, 0.85)
COLOR_OTHER = (0.55, 0.60, 0.70, 0.55)
COLOR_PROP = (0.95, 0.75, 0.15, 0.9)
COLOR_PLAN = (0.20, 0.60, 1.00, 0.9)


# ================================================================ pure builders (no rclpy)
def yaw_quat(yaw: float) -> Tuple[float, float, float, float]:
    """(x, y, z, w) for a rotation about +z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def rpy_quat(r: float, p: float, y: float) -> Tuple[float, float, float, float]:
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2),
                              math.sin(p / 2), math.cos(y / 2), math.sin(y / 2))
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def map_to_odom(st: Sequence[float], od: Sequence[float]) -> Tuple[float, float, float]:
    """The map -> odom transform (x, y, yaw) such that map->odom * odom->base_link is the ground
    truth pose `st`, given the dead-reckoned pose `od`. Every value is a Python float: rclpy's
    message fields assert on numpy scalars."""
    dyaw = float(st[2]) - float(od[2])
    c, s = math.cos(dyaw), math.sin(dyaw)
    ox, oy = float(od[0]), float(od[1])
    return (float(st[0]) - (c * ox - s * oy), float(st[1]) - (s * ox + c * oy), dyaw)


def car_boxes(xy_yaw: np.ndarray, dims: np.ndarray, rear_depth: np.ndarray,
              ros_car: int, rivals: Sequence[int]) -> List[Dict[str, Any]]:
    """One box per car. `xy_yaw` (B,3) ground-truth pose of the CoG, `dims` (B,3) length, width,
    height of the LiDAR silhouette, `rear_depth` (B,) how far the silhouette extends behind the
    CoG (the simulator's `car_rear[:, 0]`). The box is centred where the silhouette actually sits:
    the CoG is not at the middle of the body."""
    out = []
    riv = set(int(i) for i in rivals)
    for i in range(int(xy_yaw.shape[0])):
        x, y, yaw = (float(v) for v in xy_yaw[i, :3])
        L, W, H = (float(v) for v in dims[i, :3])
        back = float(rear_depth[i])
        # silhouette spans [-back, L - back] along the body axis -> centre at (L/2 - back)
        off = L / 2.0 - back
        cx, cy = x + off * math.cos(yaw), y + off * math.sin(yaw)
        color = COLOR_ROS_CAR if i == ros_car else (COLOR_RIVAL if i in riv else COLOR_OTHER)
        out.append({"id": i, "x": cx, "y": cy, "z": H / 2.0, "yaw": yaw,
                    "sx": L, "sy": W, "sz": H, "color": color})
    return out


def prop_wireframe(footprint: np.ndarray, height: float, x: float, y: float, yaw: float
                   ) -> List[Tuple[float, float, float]]:
    """A LINE_LIST (pairs of points) drawing the prop's envelope prism in the map frame: the
    bottom ring, the top ring and the vertical edges."""
    c, s = math.cos(yaw), math.sin(yaw)
    fp = np.asarray(footprint, dtype=np.float64)
    wx = x + fp[:, 0] * c - fp[:, 1] * s
    wy = y + fp[:, 0] * s + fp[:, 1] * c
    K = int(fp.shape[0])
    pts: List[Tuple[float, float, float]] = []
    for k in range(K):
        j = (k + 1) % K
        pts += [(float(wx[k]), float(wy[k]), 0.0), (float(wx[j]), float(wy[j]), 0.0)]
        pts += [(float(wx[k]), float(wy[k]), float(height)), (float(wx[j]), float(wy[j]), float(height))]
        pts += [(float(wx[k]), float(wy[k]), 0.0), (float(wx[k]), float(wy[k]), float(height))]
    return pts


def speed_colors(v: np.ndarray, v_lo: Optional[float] = None, v_hi: Optional[float] = None
                 ) -> List[Tuple[float, float, float, float]]:
    """Blue (slow) -> green -> red (fast), one colour per point."""
    v = np.asarray(v, dtype=np.float64)
    lo = float(v.min()) if v_lo is None else float(v_lo)
    hi = float(v.max()) if v_hi is None else float(v_hi)
    t = np.zeros_like(v) if hi - lo < 1e-6 else np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    out = []
    for u in t:
        u = float(u)
        r = min(1.0, max(0.0, 2.0 * u - 0.5))
        g = 1.0 - abs(2.0 * u - 1.0)
        b = min(1.0, max(0.0, 1.5 - 2.0 * u))
        out.append((r, g, b, 0.95))
    return out


def closed_path(xy: np.ndarray) -> np.ndarray:
    """A polyline with its first point repeated at the end, so a LINE_STRIP closes."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] == 0:
        return xy
    if np.linalg.norm(xy[0] - xy[-1]) > 1e-9:
        xy = np.vstack([xy, xy[:1]])
    return xy


def plan_world(ref: np.ndarray, pose: Sequence[float]) -> np.ndarray:
    """The plan tracker's body-frame reference (K, >=2) -> world XY (K, 2)."""
    ps = [float(v) for v in pose[:3]]
    cs, sn = math.cos(ps[2]), math.sin(ps[2])
    ref = np.asarray(ref, dtype=np.float64)
    return np.stack([ps[0] + ref[:, 0] * cs - ref[:, 1] * sn,
                     ps[1] + ref[:, 0] * sn + ref[:, 1] * cs], 1)


# ================================================================ the node
class RosLink:
    """See the module docstring. Construct on the control thread (it creates the node and starts
    the spin thread); call `publish` and `command` from the simulation thread; `close` from
    whichever thread releases the session."""

    def __init__(self, sim, track, raceline=None, *, mode: str = "publish", car: int = 0,
                 node_name: str = "f1sim_console", base_frame: str = "base_link",
                 laser_frame: str = "laser", odom_frame: str = "odom", map_frame: str = "map",
                 imu_frame: str = "imu", drive_topic: str = "/drive", cmd_timeout: float = CMD_TIMEOUT_S,
                 viz_hz: float = 20.0):
        if mode not in MODES:
            raise ValueError(f"ros2 mode {mode!r}: expected one of {MODES}")
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile
        from ackermann_msgs.msg import AckermannDriveStamped
        from nav_msgs.msg import OccupancyGrid, Odometry, Path
        from sensor_msgs.msg import Imu, LaserScan
        from std_msgs.msg import Bool, Float32MultiArray
        from std_srvs.srv import Empty
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
        from visualization_msgs.msg import Marker, MarkerArray

        self.mode, self.car = mode, int(car)
        self.sim, self.track, self.raceline = sim, track, raceline
        self.frames = dict(base=base_frame, laser=laser_frame, odom=odom_frame, map=map_frame, imu=imu_frame)
        self.drive_topic = drive_topic
        self.cmd_timeout = float(cmd_timeout)
        self.viz_dt = 1.0 / float(viz_hz) if viz_hz > 0 else 0.0
        self._last_viz = 0.0
        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0)
        self._cmd_time: Optional[float] = None
        self._cmd_count = 0
        self._reset_requested = False
        self._alive = True
        self.published = 0

        if not rclpy.ok():
            rclpy.init(args=None)
        self._rclpy = rclpy
        self.node = rclpy.create_node(node_name)
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        n = self.node
        self.pub_scan = n.create_publisher(LaserScan, "/scan", 1)
        self.pub_odom = n.create_publisher(Odometry, "/odom", 1)
        self.pub_gt = n.create_publisher(Odometry, "/ego_racecar/odom", 1)
        self.pub_coll = n.create_publisher(Bool, "/f1sim/collision", 1)
        self.pub_imu = n.create_publisher(Imu, "/sensors/imu", 1)
        self.pub_imu_raw = n.create_publisher(Imu, "/sensors/imu/raw", 10)
        self.pub_map = n.create_publisher(OccupancyGrid, "/map", latched)
        self.pub_raceline = n.create_publisher(Path, "/f1sim/raceline", latched)
        self.pub_raceline_v = n.create_publisher(Float32MultiArray, "/f1sim/raceline_speed", latched)
        self.pub_centerline = n.create_publisher(Path, "/f1sim/centerline", latched)
        self.pub_cars = n.create_publisher(MarkerArray, "/f1sim/viz/cars", 1)
        self.pub_props = n.create_publisher(MarkerArray, "/f1sim/viz/props", latched)
        self.pub_raceline_viz = n.create_publisher(Marker, "/f1sim/viz/raceline", latched)
        self.pub_plan = n.create_publisher(Marker, "/f1sim/viz/plan", 1)
        self.tf = TransformBroadcaster(n)
        self.tf_static = StaticTransformBroadcaster(n)
        self.sub_drive = n.create_subscription(AckermannDriveStamped, drive_topic, self._on_drive, 1)
        self.srv_reset = n.create_service(Empty, "/f1sim/reset", self._on_reset)
        self._M = dict(Marker=Marker, MarkerArray=MarkerArray, Path=Path, Imu=Imu, LaserScan=LaserScan,
                       Odometry=Odometry, Bool=Bool, OccupancyGrid=OccupancyGrid,
                       Float32MultiArray=Float32MultiArray)

        self._publish_static()
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._spin, name="ros-link-spin", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- facts
    def facts(self) -> Dict[str, Any]:
        return {"mode": self.mode, "car": self.car, "node": self.node.get_name(),
                "drive_topic": self.drive_topic,
                "topics": ["/scan", "/odom", "/ego_racecar/odom", "/sensors/imu", "/sensors/imu/raw",
                           "/map", "/f1sim/collision", "/f1sim/raceline", "/f1sim/centerline",
                           "/f1sim/viz/cars", "/f1sim/viz/props", "/f1sim/viz/raceline", "/f1sim/viz/plan"]}

    # ---------------------------------------------------------------- callbacks (executor thread)
    def _spin(self) -> None:
        while self._alive:
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except Exception:
                if self._alive:
                    time.sleep(0.05)

    def _on_drive(self, msg) -> None:
        with self._lock:
            self._cmd = (float(msg.drive.steering_angle), float(msg.drive.speed))
            self._cmd_time = time.monotonic()
            self._cmd_count += 1

    def _on_reset(self, _req, resp):
        with self._lock:
            self._reset_requested = True
        return resp

    # ---------------------------------------------------------------- simulation-thread API
    def command(self) -> Tuple[float, float, bool]:
        """(steer, speed, fresh). When no `/drive` has arrived within `cmd_timeout`, the speed is
        0 and `fresh` is False: the car coasts to a stop on its last steering angle."""
        with self._lock:
            steer, speed = self._cmd
            t = self._cmd_time
        if t is None or time.monotonic() - t > self.cmd_timeout:
            return steer, 0.0, False
        return steer, speed, True

    def drive_count(self) -> int:
        with self._lock:
            return self._cmd_count

    def take_reset(self) -> bool:
        with self._lock:
            r, self._reset_requested = self._reset_requested, False
        return r

    def publish(self, r, env, plan_ref: Optional[np.ndarray] = None) -> None:
        """Publish one control step. `r` is the simulator's StepResult, `env` the gym env (for the
        car dims / rivals). One device->host transfer for everything; the marker array for all
        cars is rate-limited to `viz_hz`."""
        import torch
        sim = self.sim
        c = self.car
        now = self.node.get_clock().now()
        B = int(sim.B)
        want_viz = self.viz_dt <= 0 or (time.monotonic() - self._last_viz) >= self.viz_dt
        pieces = [r.state[c], r.odom[c], r.attitude[c], r.collision[c:c + 1].float()]
        k_imu = int(r.imu.shape[1]) if getattr(r, "imu", None) is not None else 0
        if k_imu > 0:
            pieces += [r.imu[c].reshape(-1), r.imu_att[c], r.imu_offsets.reshape(-1).to(r.state.dtype)]
        pieces.append(r.scan[c])
        if want_viz:
            pieces += [sim.state[:, :3].reshape(-1), sim.car_dims.reshape(-1), sim.car_rear[:, 0]]
        if plan_ref is not None:
            pieces += [plan_ref.reshape(-1).to(r.state.dtype)]
        flat = torch.cat([p.reshape(-1) for p in pieces]).cpu().numpy()

        # Every width is read off the tensor it came from. These used to be written in -- state 7,
        # odom 5, attitude 2, IMU attitude 3 -- and the day the wheel model added a column to the
        # state (`dyn.IOMEGA`, STATE_DIM 7 -> 8) every field after it shifted by one float and was
        # published that way: `/odom` led with the wheel speed, each IMU row came out as
        # [previous a_z, g_x, g_y, g_z, a_x, a_y] -- gravity in `angular_velocity.x` on the second
        # sample of every two-sample step, which is one message in five at 50 Hz against 40 Hz, and
        # no gravity in `linear_acceleration.z` -- and every LiDAR beam sat one index late. At a
        # standstill most of it reads zero, which is why it survived until someone looked.
        i = 0
        n_st, n_od, n_att = int(r.state.shape[1]), int(r.odom.shape[1]), int(r.attitude.shape[1])
        st = flat[i:i + 7]; i += n_st           # x, y, yaw, vx, vy, r, steer -- the rest is not published
        od = flat[i:i + 5]; i += n_od
        i += n_att                              # body roll/pitch: on the frame, not on any topic
        coll = bool(flat[i] > 0.5); i += 1
        imu = imu_att = offsets = None
        if k_imu > 0:
            n_ia = int(r.imu_att.shape[1])
            imu = flat[i:i + k_imu * 6].reshape(k_imu, 6); i += k_imu * 6
            imu_att = flat[i:i + 3]; i += n_ia
            offsets = flat[i:i + k_imu]; i += k_imu
        nb = int(r.scan.shape[1])
        scan = flat[i:i + nb]; i += nb
        cars = dims = rear = None
        if want_viz:
            cars = flat[i:i + 3 * B].reshape(B, 3); i += 3 * B
            dims = flat[i:i + 3 * B].reshape(B, 3); i += 3 * B
            rear = flat[i:i + B]; i += B
        plan_w = None
        if plan_ref is not None:
            k_, c_ = plan_ref.shape
            ref = flat[i:i + k_ * c_].reshape(k_, c_); i += k_ * c_
            plan_w = plan_world(ref, st)

        stamp = now.to_msg()
        self._publish_scan(scan, now)
        self._publish_odom(od, st, stamp)
        self.pub_coll.publish(self._M["Bool"](data=coll))
        if imu is not None:
            self._publish_imu(imu, imu_att, now, offsets)
        if want_viz:
            self._last_viz = time.monotonic()
            rivals = sim.other_idx[c].tolist() if getattr(sim, "other_idx", None) is not None else []
            self._publish_cars(cars, dims, rear, rivals, stamp)
        if plan_w is not None:
            self._publish_plan(plan_w, stamp)
        self.published += 1

    def close(self) -> None:
        """Stop the spin thread and destroy the node. The rclpy context stays up for the process:
        the next session makes a new node on it."""
        self._alive = False
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._executor.remove_node(self.node)
            self._executor.shutdown(timeout_sec=0.5)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass

    # ---------------------------------------------------------------- message assembly
    def _hdr(self, msg, frame: str, stamp):
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        return msg

    def _publish_static(self) -> None:
        from geometry_msgs.msg import TransformStamped
        cfg = self.sim.cfg
        stamp = self.node.get_clock().now().to_msg()
        t = TransformStamped(); t.header.stamp = stamp
        t.header.frame_id = self.frames["base"]; t.child_frame_id = self.frames["laser"]
        t.transform.translation.x = float(cfg.lidar.mount_x)
        t.transform.translation.y = float(cfg.lidar.mount_y)
        t.transform.translation.z = float(cfg.lidar.mount_z)
        q = yaw_quat(float(cfg.lidar.mount_yaw))
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        ti = TransformStamped(); ti.header.stamp = stamp
        ti.header.frame_id = self.frames["base"]; ti.child_frame_id = self.frames["imu"]
        ti.transform.translation.x = float(cfg.imu.imu_x)
        ti.transform.translation.y = float(cfg.imu.imu_y)
        ti.transform.translation.z = float(cfg.imu.imu_z)
        ti.transform.rotation.w = 1.0
        self.tf_static.sendTransform([t, ti])

        self._publish_map(stamp)
        self._publish_lines(stamp)
        self._publish_props(stamp)

    def _publish_map(self, stamp) -> None:
        tr = self.track
        m = self._M["OccupancyGrid"]()
        self._hdr(m, self.frames["map"], stamp)
        m.info.resolution = float(tr.resolution)
        m.info.height, m.info.width = (int(v) for v in tr.occupancy.shape)
        m.info.origin.position.x, m.info.origin.position.y = float(tr.origin[0]), float(tr.origin[1])
        m.info.origin.orientation.w = 1.0
        m.data = np.where(tr.occupancy, 100, 0).astype(np.int8).reshape(-1).tolist()
        self.pub_map.publish(m)

    def _path(self, xy: np.ndarray, stamp):
        from geometry_msgs.msg import PoseStamped
        p = self._M["Path"]()
        self._hdr(p, self.frames["map"], stamp)
        xy = np.asarray(xy, dtype=np.float64)
        n = xy.shape[0]
        for k in range(n):
            ps = PoseStamped()
            ps.header.frame_id = self.frames["map"]; ps.header.stamp = stamp
            ps.pose.position.x, ps.pose.position.y = float(xy[k, 0]), float(xy[k, 1])
            nxt = xy[(k + 1) % n]
            yaw = math.atan2(float(nxt[1] - xy[k, 1]), float(nxt[0] - xy[k, 0]))
            q = yaw_quat(yaw)
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
            p.poses.append(ps)
        return p

    def _publish_lines(self, stamp) -> None:
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        tr = self.track
        if tr.centerline is not None and len(tr.centerline) > 1:
            self.pub_centerline.publish(self._path(tr.centerline, stamp))
        rl = self.raceline
        if rl is None or getattr(rl, "xy", None) is None or len(rl.xy) < 2:
            return
        self.pub_raceline.publish(self._path(rl.xy, stamp))
        arr = self._M["Float32MultiArray"]()
        arr.data = [float(v) for v in np.asarray(rl.v, dtype=np.float32)]
        self.pub_raceline_v.publish(arr)
        m = self._M["Marker"]()
        self._hdr(m, self.frames["map"], stamp)
        m.ns, m.id, m.type, m.action = "raceline", 0, m.LINE_STRIP, m.ADD
        m.scale.x = 0.05
        m.pose.orientation.w = 1.0
        xy = closed_path(rl.xy)
        v = np.concatenate([rl.v, rl.v[:1]]) if xy.shape[0] == len(rl.v) + 1 else rl.v
        for (x, y), col in zip(xy, speed_colors(v)):
            m.points.append(Point(x=float(x), y=float(y), z=0.02))
            m.colors.append(ColorRGBA(r=col[0], g=col[1], b=col[2], a=col[3]))
        self.pub_raceline_viz.publish(m)

    def _publish_props(self, stamp) -> None:
        from geometry_msgs.msg import Point
        props = tuple(getattr(self.track, "props", ()) or ())
        arr = self._M["MarkerArray"]()
        for k, sp in enumerate(props):
            try:
                env = sp.build().envelope
            except Exception:
                continue
            m = self._M["Marker"]()
            self._hdr(m, self.frames["map"], stamp)
            m.ns, m.id, m.type, m.action = "props", k, m.LINE_LIST, m.ADD
            m.scale.x = 0.03
            m.pose.orientation.w = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = COLOR_PROP
            for (x, y, z) in prop_wireframe(env.footprint, env.height, sp.x, sp.y, sp.yaw):
                m.points.append(Point(x=x, y=y, z=z))
            arr.markers.append(m)
        self.pub_props.publish(arr)          # an empty array is the honest answer on a map without props

    def _publish_scan(self, ranges: np.ndarray, now) -> None:
        from rclpy.duration import Duration
        meta = self.sim.scan_meta()
        sweep = float(meta["time_increment"]) * (len(ranges) - 1)
        msg = self._M["LaserScan"]()
        self._hdr(msg, self.frames["laser"], (now - Duration(seconds=sweep)).to_msg())
        msg.angle_min, msg.angle_max = float(meta["angle_min"]), float(meta["angle_max"])
        msg.angle_increment = float(meta["angle_increment"])
        msg.time_increment, msg.scan_time = float(meta["time_increment"]), float(meta["scan_time"])
        msg.range_min, msg.range_max = float(meta["range_min"]), float(meta["range_max"])
        msg.ranges = np.asarray(ranges, dtype=np.float32).tolist()
        self.pub_scan.publish(msg)

    def _publish_odom(self, od: np.ndarray, st: np.ndarray, stamp) -> None:
        from geometry_msgs.msg import TransformStamped
        Odometry = self._M["Odometry"]
        f = self.frames
        o = Odometry(); self._hdr(o, f["odom"], stamp); o.child_frame_id = f["base"]
        o.pose.pose.position.x, o.pose.pose.position.y = float(od[0]), float(od[1])
        q = yaw_quat(float(od[2]))
        o.pose.pose.orientation.x, o.pose.pose.orientation.y, o.pose.pose.orientation.z, o.pose.pose.orientation.w = q
        o.twist.twist.linear.x, o.twist.twist.angular.z = float(od[3]), float(od[4])
        self.pub_odom.publish(o)
        t = TransformStamped(); t.header.stamp = stamp; t.header.frame_id = f["odom"]; t.child_frame_id = f["base"]
        t.transform.translation.x, t.transform.translation.y = float(od[0]), float(od[1])
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        g = Odometry(); self._hdr(g, f["map"], stamp); g.child_frame_id = f["base"]
        g.pose.pose.position.x, g.pose.pose.position.y = float(st[0]), float(st[1])
        qg = yaw_quat(float(st[2]))
        g.pose.pose.orientation.x, g.pose.pose.orientation.y, g.pose.pose.orientation.z, g.pose.pose.orientation.w = qg
        g.twist.twist.linear.x, g.twist.twist.linear.y, g.twist.twist.angular.z = float(st[3]), float(st[4]), float(st[5])
        self.pub_gt.publish(g)
        mx, my, dyaw = map_to_odom(st, od)
        mo = TransformStamped(); mo.header.stamp = stamp; mo.header.frame_id = f["map"]; mo.child_frame_id = f["odom"]
        mo.transform.translation.x, mo.transform.translation.y = mx, my
        qm = yaw_quat(dyaw)
        mo.transform.rotation.x, mo.transform.rotation.y, mo.transform.rotation.z, mo.transform.rotation.w = qm
        self.tf.sendTransform([t, mo])

    def _publish_imu(self, samples: np.ndarray, att: np.ndarray, now, offsets: np.ndarray) -> None:
        from rclpy.duration import Duration
        Imu = self._M["Imu"]
        K = int(samples.shape[0])
        for k in range(K):
            m = Imu()
            self._hdr(m, self.frames["imu"], (now - Duration(seconds=float(offsets[k]))).to_msg())
            g, a = samples[k, :3], samples[k, 3:]
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = float(g[0]), float(g[1]), float(g[2])
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = float(a[0]), float(a[1]), float(a[2])
            if k == K - 1:
                q = rpy_quat(float(att[0]), float(att[1]), float(att[2]))
                m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = q
            else:
                m.orientation_covariance[0] = -1.0
            self.pub_imu_raw.publish(m)
        m = Imu()
        self._hdr(m, self.frames["imu"], (now - Duration(seconds=float(offsets[-1]))).to_msg())
        g, a = samples[-1, :3], samples[-1, 3:]
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = float(g[0]), float(g[1]), float(g[2])
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = float(a[0]), float(a[1]), float(a[2])
        q = rpy_quat(float(att[0]), float(att[1]), float(att[2]))
        m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = q
        self.pub_imu.publish(m)

    def _publish_cars(self, cars: np.ndarray, dims: np.ndarray, rear: np.ndarray,
                      rivals: Sequence[int], stamp) -> None:
        arr = self._M["MarkerArray"]()
        for box in car_boxes(cars, dims, rear, self.car, rivals):
            m = self._M["Marker"]()
            self._hdr(m, self.frames["map"], stamp)
            m.ns, m.id, m.type, m.action = "cars", int(box["id"]), m.CUBE, m.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = box["x"], box["y"], box["z"]
            q = yaw_quat(box["yaw"])
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
            m.scale.x, m.scale.y, m.scale.z = box["sx"], box["sy"], box["sz"]
            m.color.r, m.color.g, m.color.b, m.color.a = box["color"]
            arr.markers.append(m)
        self.pub_cars.publish(arr)

    def _publish_plan(self, xy: np.ndarray, stamp) -> None:
        from geometry_msgs.msg import Point
        m = self._M["Marker"]()
        self._hdr(m, self.frames["map"], stamp)
        m.ns, m.id, m.type, m.action = "plan", 0, m.LINE_STRIP, m.ADD
        m.scale.x = 0.04
        m.pose.orientation.w = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = COLOR_PLAN
        for x, y in xy:
            m.points.append(Point(x=float(x), y=float(y), z=0.05))
        self.pub_plan.publish(m)


def ros2_available() -> Optional[str]:
    """None when rclpy and the message packages import; else the reason they do not."""
    try:
        import rclpy  # noqa: F401
        import ackermann_msgs.msg  # noqa: F401
        import visualization_msgs.msg  # noqa: F401
        import tf2_ros  # noqa: F401
    except Exception as exc:  # ImportError, but also a broken ROS install
        return f"{type(exc).__name__}: {exc}"
    return None
