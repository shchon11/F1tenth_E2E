"""실측 bag -> npz (11-09, 사용자: "실측백 있잖아. 기존 측정했던거 믿지 말고 네가 새로 직접 다시 확인해").
bag 하나에서 차의 명령, 모터 상태, IMU, 위치, LiDAR를 시간과 함께 뽑아 /data/sim2real_1109/<bag 이름>.npz로 저장.
시간: t_* = bag 기록 시각 [s] (bag 시작 기준), s_* = 메시지 header 시각 [s] (같은 기준; header가 없거나 0이면 nan).
실행: bash -c "cd ~ && source ~/f1tenth_ws/activate.sh && python3 bag_extract.py <bag 폴더> [출력 폴더]"
"""
import os, sys, math
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag = sys.argv[1].rstrip("/")
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/data/sim2real_1109"
os.makedirs(out_dir, exist_ok=True)
name = os.path.basename(bag)
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
WANT = ["/drive", "/commands/motor/speed", "/commands/servo/position", "/sensors/servo_position_command", "/sensors/core",
        "/sensors/imu/raw", "/sensors/imu", "/imu/filtered_angular_velocity", "/odom", "/pf/pose/odom", "/car_state/odom",
        "/scan", "/state", "/map", "/tf_static", "/teleop", "/joy"]
cls = {t: get_message(types[t]) for t in WANT if t in types}
D = {k: [] for k in WANT}
t0 = None


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def hs(h):
    s = h.stamp.sec + h.stamp.nanosec * 1e-9
    return s if s > 0 else float("nan")


scan_meta = None
map_msg = None
tf_static = []
while r.has_next():
    topic, data, tns = r.read_next()
    if topic not in cls:
        continue
    t = tns * 1e-9
    if t0 is None:
        t0 = t
    m = deserialize_message(data, cls[topic])
    if topic in ("/drive", "/teleop"):
        D[topic].append((t, hs(m.header), m.drive.speed, m.drive.steering_angle, m.drive.acceleration, m.drive.steering_angle_velocity))
    elif topic in ("/commands/motor/speed", "/commands/servo/position", "/sensors/servo_position_command", "/imu/filtered_angular_velocity"):
        D[topic].append((t, m.data))
    elif topic == "/sensors/core":
        s = m.state
        D[topic].append((t, hs(m.header), s.speed, s.current_motor, s.current_input, s.duty_cycle, s.voltage_input,
                         float(s.displacement), float(s.distance_traveled), s.avg_iq, s.avg_id))
    elif topic == "/sensors/imu/raw":
        q = m.orientation
        D[topic].append((t, hs(m.header), m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
                         m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z, yaw_of(q), q.x, q.y, q.z, q.w))
    elif topic == "/sensors/imu":
        i = m.imu
        D[topic].append((t, hs(m.header), i.ypr.x, i.ypr.y, i.ypr.z, i.linear_acceleration.x, i.linear_acceleration.y,
                         i.linear_acceleration.z, i.angular_velocity.x, i.angular_velocity.y, i.angular_velocity.z))
    elif topic in ("/odom", "/pf/pose/odom", "/car_state/odom"):
        p, tw = m.pose.pose, m.twist.twist
        D[topic].append((t, hs(m.header), p.position.x, p.position.y, yaw_of(p.orientation), tw.linear.x, tw.linear.y, tw.angular.z))
    elif topic == "/scan":
        if scan_meta is None:
            scan_meta = (m.angle_min, m.angle_max, m.angle_increment, m.time_increment, m.scan_time, m.range_min, m.range_max, len(m.ranges))
        D[topic].append((t, hs(m.header), np.asarray(m.ranges, dtype=np.float32), np.asarray(m.intensities, dtype=np.float32)))
    elif topic == "/state":
        D[topic].append((t, m.data))
    elif topic == "/joy":
        D[topic].append((t, list(m.axes)[:8] + [0.0] * max(0, 8 - len(m.axes)), list(m.buttons)[:12] + [0] * max(0, 12 - len(m.buttons))))
    elif topic == "/map" and map_msg is None:
        map_msg = m
    elif topic == "/tf_static":
        for tr in m.transforms:
            tf_static.append((tr.header.frame_id, tr.child_frame_id, tr.transform.translation.x, tr.transform.translation.y,
                              tr.transform.translation.z, yaw_of(tr.transform.rotation), tr.transform.rotation.x,
                              tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w))
out = {"t0": t0, "name": name}
num = {"/drive": "drive", "/teleop": "teleop", "/commands/motor/speed": "erpm_cmd", "/commands/servo/position": "servo_cmd",
       "/sensors/servo_position_command": "servo_pos_cmd", "/imu/filtered_angular_velocity": "imu_filt", "/sensors/core": "core",
       "/sensors/imu/raw": "imu_raw", "/sensors/imu": "vesc_imu", "/odom": "odom", "/pf/pose/odom": "pf", "/car_state/odom": "cs_odom"}
for k, nm in num.items():
    if D[k]:
        a = np.array(D[k], dtype=np.float64)
        a[:, 0] -= t0
        if a.shape[1] > 1 and nm in ("drive", "teleop", "core", "imu_raw", "vesc_imu", "odom", "pf", "cs_odom"):
            a[:, 1] -= t0
        out[nm] = a
if D["/scan"]:
    out["scan_t"] = np.array([x[0] - t0 for x in D["/scan"]]); out["scan_s"] = np.array([x[1] - t0 for x in D["/scan"]])
    out["scan"] = np.stack([x[2] for x in D["/scan"]]).astype(np.float16)
    if D["/scan"][0][3].size:
        out["scan_int"] = np.stack([x[3] for x in D["/scan"]]).astype(np.float16)
    out["scan_meta"] = np.array(scan_meta, dtype=np.float64)
if D["/state"]:
    out["state_t"] = np.array([x[0] - t0 for x in D["/state"]]); out["state"] = np.array([x[1] for x in D["/state"]])
if D["/joy"]:
    out["joy_t"] = np.array([x[0] - t0 for x in D["/joy"]]); out["joy_axes"] = np.array([x[1] for x in D["/joy"]])
    out["joy_buttons"] = np.array([x[2] for x in D["/joy"]])
if map_msg is not None:
    i = map_msg.info
    out["map"] = np.array(map_msg.data, dtype=np.int8).reshape(i.height, i.width)
    out["map_info"] = np.array([i.resolution, i.origin.position.x, i.origin.position.y, yaw_of(i.origin.orientation)])
if tf_static:
    out["tf_static"] = np.array([str(x) for x in tf_static])
np.savez_compressed(os.path.join(out_dir, name + ".npz"), **out)
print(name, {k: (v.shape if hasattr(v, "shape") else v) for k, v in out.items() if k not in ("name",)}, flush=True)
