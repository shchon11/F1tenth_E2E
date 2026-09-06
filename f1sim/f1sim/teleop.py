"""Manual driving with racing-game feel, on top of the car's real command interface.

The car (real or simulated) takes (steering angle, target speed) like the VESC speed loop.
Games give you throttle and brake pedals, not a speed target, so DriveModel integrates a
target speed from pedal inputs: throttle accelerates, brake decelerates hard, no pedals
coasts down (engine braking), brake at standstill engages reverse, and the target never
runs away from the measured speed when the car is blocked. Steering has a speed-sensitive
lock (less lock at speed, like game assists / real arm effort) and slew rates; keyboard
keys are ramped into analog values so digital input feels like a stick.
Everything is pure python: used by the ROS teleop node, the vesc_sim window and demo_viewer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


@dataclass
class TeleopConfig:
    v_max: float = 5.0            # [m/s] manual top speed (boost lifts it to v_boost)
    v_boost: float = 8.0
    v_reverse: float = 1.2
    a_throttle: float = 2.0       # [m/s^2] target-speed ramp at full throttle (gentle; boost doubles it)
    a_brake: float = 5.0
    a_coast: float = 0.8          # engine braking with no pedals
    a_handbrake: float = 10.0
    throttle_gamma: float = 1.8   # pedal curve: the first half of the pedal is soft
    steer_max: float = 0.4189     # [rad] full lock
    steer_rate: float = 4.0       # [rad/s] toward the stick target
    center_rate: float = 6.0      # [rad/s] back toward center / reversing direction
    lock_v_ref: float = 2.5       # [m/s] above this the usable lock shrinks ~ v_ref / v
    lock_min: float = 0.35        # fraction of full lock kept at high speed
    key_press_time: float = 0.35  # [s] keyboard 0 -> 1 ramp (steer, throttle, brake)
    key_release_time: float = 0.10
    stick_deadzone: float = 0.08
    stick_gamma: float = 1.7      # >1: finer control around center
    trigger_deadzone: float = 0.03
    reverse_hold: float = 0.35    # [s] brake held at standstill before reverse engages
    max_lead: float = 2.5         # [m/s] cap on target - measured speed (blocked / spinning wheels)


@dataclass
class Inputs:
    steer: float = 0.0            # -1 (right) .. +1 (left), analog
    throttle: float = 0.0         # 0..1
    brake: float = 0.0            # 0..1
    boost: bool = False
    handbrake: bool = False
    deadman: bool = True          # False -> the driver is not holding the enable button: stop
    autonomy: bool = False        # True -> hand over to the autonomous topic (stop publishing)


class KeyRamp:
    """Turns held keys into ramped analog axes (what a keyboard racing game does)."""

    def __init__(self, cfg: TeleopConfig):
        self.cfg = cfg
        self.steer = 0.0; self.throttle = 0.0; self.brake = 0.0

    @staticmethod
    def _ramp(x, target, dt, t_up, t_down):
        toward_center = abs(target) < abs(x) or (x * target < 0)
        rate = 1.0 / (t_down if toward_center else t_up)
        step = rate * dt
        return x + max(-step, min(step, target - x))

    def update(self, left: bool, right: bool, up: bool, down: bool, dt: float, boost=False, handbrake=False) -> Inputs:
        c = self.cfg
        self.steer = self._ramp(self.steer, float(left) - float(right), dt, c.key_press_time, c.key_release_time)
        self.throttle = self._ramp(self.throttle, float(up), dt, c.key_press_time, c.key_release_time)
        self.brake = self._ramp(self.brake, float(down), dt, c.key_press_time * 0.6, c.key_release_time)
        return Inputs(self.steer, self.throttle, self.brake, boost, handbrake)


@dataclass
class JoyMapping:
    """Indices into sensor_msgs/Joy. ROS joy: sticks are +1 left / +1 up, triggers rest at +1 and go to -1."""
    steer_axis: int = 0
    throttle_axis: Optional[int] = 5     # analog trigger (rest +1 -> pressed -1); None: use speed_axis
    brake_axis: Optional[int] = 2
    speed_axis: Optional[int] = None     # stick axis: up = throttle, down = brake
    boost_button: Optional[int] = 4
    handbrake_button: Optional[int] = 0
    deadman_button: Optional[int] = None  # must be held to drive (F1TENTH stack semantics)
    autonomy_button: Optional[int] = None  # held -> hand over to /drive


PRESETS: Dict[str, JoyMapping] = {
    # Xbox / F710 in X mode: left stick steers, RT throttle, LT brake, LB boost, A handbrake
    "xbox": JoyMapping(),
    # f1tenth_stack joy_teleop.yaml (F710 D mode): left stick Y speed, right stick X steer,
    # hold LB (4) to drive, hold RB (5) for autonomous, nothing held -> stop
    "f1tenth": JoyMapping(steer_axis=2, throttle_axis=None, brake_axis=None, speed_axis=1,
                          boost_button=None, handbrake_button=None, deadman_button=4, autonomy_button=5),
}


class JoyDecoder:
    def __init__(self, cfg: TeleopConfig, mapping: JoyMapping):
        self.cfg, self.m = cfg, mapping
        self._trigger_seen: Dict[int, bool] = {}

    def _stick(self, v: float) -> float:
        dz = self.cfg.stick_deadzone
        if abs(v) < dz:
            return 0.0
        x = (abs(v) - dz) / (1 - dz)
        return math.copysign(x ** self.cfg.stick_gamma, v)

    def _trigger(self, axes, idx) -> float:
        if idx is None or idx >= len(axes):
            return 0.0
        v = axes[idx]
        if v != 0.0:
            self._trigger_seen[idx] = True
        if not self._trigger_seen.get(idx):          # driver reports 0 until first touch: not pressed
            return 0.0
        t = (1.0 - v) / 2.0
        return 0.0 if t < self.cfg.trigger_deadzone else min(1.0, t)

    def decode(self, axes: Sequence[float], buttons: Sequence[int]) -> Inputs:
        m = self.m
        btn = lambda i: bool(i is not None and i < len(buttons) and buttons[i])
        steer = self._stick(axes[m.steer_axis]) if m.steer_axis < len(axes) else 0.0
        if m.speed_axis is not None and m.speed_axis < len(axes):
            s = self._stick(axes[m.speed_axis]); throttle, brake = max(0.0, s), max(0.0, -s)
        else:
            throttle, brake = self._trigger(axes, m.throttle_axis), self._trigger(axes, m.brake_axis)
        return Inputs(steer, throttle, brake, btn(m.boost_button), btn(m.handbrake_button),
                      deadman=(m.deadman_button is None) or btn(m.deadman_button), autonomy=btn(m.autonomy_button))


class DriveModel:
    """Pedals + stick -> (steering angle [rad], target speed [m/s]) with game-like dynamics."""

    def __init__(self, cfg: Optional[TeleopConfig] = None):
        self.cfg = cfg or TeleopConfig()
        self.v_cmd = 0.0; self.steer = 0.0; self.reverse = False; self._hold = 0.0

    def reset(self):
        self.v_cmd = 0.0; self.steer = 0.0; self.reverse = False; self._hold = 0.0

    def lock(self, v: float) -> float:
        return max(self.cfg.lock_min, min(1.0, self.cfg.lock_v_ref / max(abs(v), 1e-3)))

    def update(self, inp: Inputs, v_meas: float, dt: float):
        c = self.cfg
        if not inp.deadman:                                   # enable button released: stop, straighten
            self.v_cmd = 0.0; self.steer = self._slew(self.steer, 0.0, c.center_rate * dt); self.reverse = False
            return self.steer, 0.0
        # --- steering with speed-sensitive lock and slew
        tgt = max(-1.0, min(1.0, inp.steer)) * c.steer_max * self.lock(v_meas)
        toward_center = abs(tgt) < abs(self.steer) or tgt * self.steer < 0
        self.steer = self._slew(self.steer, tgt, (c.center_rate if toward_center else c.steer_rate) * dt)
        # --- longitudinal: integrate a target speed from the pedals
        v = self.v_cmd
        vmax = c.v_boost if inp.boost else c.v_max
        if inp.handbrake:
            v = self._slew(v, 0.0, c.a_handbrake * dt); self.reverse = self.reverse and v < 0
        elif not self.reverse:
            thr = inp.throttle ** c.throttle_gamma * (2.0 if inp.boost else 1.0)
            if inp.throttle > 0: v += thr * c.a_throttle * dt
            if inp.brake > 0: v -= inp.brake * c.a_brake * dt
            if inp.throttle == 0 and inp.brake == 0: v = max(0.0, v - c.a_coast * dt)
            v = max(0.0, min(vmax, v))
            if inp.brake > 0.5 and abs(v_meas) < 0.15 and v <= 0.0:
                self._hold += dt
                if self._hold > c.reverse_hold:
                    self.reverse = True; self._hold = 0.0
            else:
                self._hold = 0.0
        else:
            if inp.brake > 0: v -= inp.brake * c.a_throttle * 0.6 * dt          # brake pedal = reverse throttle
            if inp.throttle > 0: v += inp.throttle * c.a_brake * dt             # throttle = brake while reversing
            if inp.throttle == 0 and inp.brake == 0: v = min(0.0, v + c.a_coast * dt)
            v = max(-c.v_reverse, min(0.0, v))
            if v >= 0.0 and inp.brake == 0:
                self.reverse = False
        # --- never let the target run far ahead of the car (wall contact, wheel spin)
        v = max(v_meas - c.max_lead, min(v_meas + c.max_lead, v))
        self.v_cmd = v
        return self.steer, v

    @staticmethod
    def _slew(x, target, step):
        return x + max(-step, min(step, target - x))

    def hud(self, inp: Inputs, v_meas: float) -> list:
        bar = lambda x, n=12: "#" * int(round(abs(x) * n)) + "." * (n - int(round(abs(x) * n)))
        st = self.steer / self.cfg.steer_max
        left, right = bar(max(0.0, st), 8)[::-1], bar(max(0.0, -st), 8)
        return [f"MANUAL  throttle [{bar(inp.throttle)}]  brake [{bar(inp.brake)}]" + ("  BOOST" if inp.boost else "") + ("  HANDBRAKE" if inp.handbrake else "") + ("  REVERSE" if self.reverse else ""),
                f"        steer   {left}|{right}  lock {self.lock(v_meas) * 100:3.0f}%   target {self.v_cmd:5.2f} m/s"]


# keyboard layout shared by the viewer window and the terminal node
KEYS = {"left": ("A", "LEFT"), "right": ("D", "RIGHT"), "up": ("W", "UP"), "down": ("S", "DOWN"),
        "boost": ("LEFT_SHIFT", "RIGHT_SHIFT"), "handbrake": ("SPACE",), "reset": ("R",)}
