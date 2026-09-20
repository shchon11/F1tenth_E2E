"""A learned sensor front-end: raw scan + IMU -> per-beam class, denoised range, attitude.

The user's idea (2026-09-15): *"the simulator has the labels anyway, so why not a model that does a
first pass -- LiDAR in, IMU in, filtered LiDAR and IMU out? The IMU is such noisy data..."*

Three outputs, all from labels the simulator gives away and none of which the car has:

  * **per-beam class** {solid, floor, no-return} against `scan_type` -- the same question
    `learn/floor.py` answers with geometry, asked of a network that does not need the attitude to
    be right;
  * **a denoised range** against `StepResult.scan_true`, the same rays with the noise, the spike,
    the dropout and the grazing fade switched off. (The contract asks for a `clean_ranges` flag on
    the LiDAR; the simulator already returns exactly that tensor beside the noisy one, and
    `capture.py` stores it, so no LiDAR change was needed. Written down rather than silently
    skipped.)
  * **roll and pitch** against the true scan-plane tilt. This is the output the geometric channel
    most needs: measured, the VESC quaternion is wrong by 0.14 rad rms in roll while driving and
    `floor.AttitudeTracker` -- gyro integration with every trick this project knows -- still leaves
    0.05, while the band the floor geometry needs is ~0.015. A network that sees the *scan* can do
    what no IMU-only estimator can: the floor, when it is visible, tells the sensor how it is
    tilted, which is precisely the LiDAR-vs-map method of `real_data_calibration.md` §2.9 without
    the map.

What it must not become
-----------------------
**The actor keeps the raw stack.** The front-end's outputs are handed to the policy *beside* the
raw scan, never in place of it, so a hallucinated clean range cannot hide a real wall. That is a
property of how it is wired (`obs.SCAN_CHANNELS` gains rows; it removes none) and it is the reason
the denoised range is a channel rather than a replacement.

The risk, stated
----------------
A front-end trained on simulated noise can be confidently wrong on real noise it never saw. The
argument that it has seen the right noise is the calibration: LiDAR sigma 7.4 mm + 1 mm/m (§2.4),
the dropout structure (§2.4), the grazing-incidence fade (§2.8), the vibration floor and slope
(§2.6) and the impact shocks (§2.12) are all fitted to the 22 recordings. The check is the real
bags, and it is a check, not a proof.

Budget
------
<= 150 k parameters and <= 1 ms on this desktop CPU at batch 1, single thread, measured by
`learn/budget.py`'s protocol -- so it fits inside the 25 ms control step beside the actor. The
shape that buys it: almost no work at full beam resolution. One stride-2 convolution takes 1081
beams down to 541 immediately, the encoder does its work at 1/8 and 1/16 of the beam axis, and only
a single 1x1 convolution ever runs on all 1081 columns.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#: The per-beam classes, in the order the logits are laid out.
#:
#: **Four, not three,** and the fourth is the point. A floor hit at grazing incidence usually does
#: not come back: `lidar.floor_dropout` removes 41-46 % of them in simulation (measured), and root's
#: reading of `real_data_calibration.md` §2.9 is that on a real floor the share is higher still. A
#: three-class label calls every one of those `none`, so the network is told nothing about the
#: commonest thing the floor does -- and the distinction matters to whoever reads the channel,
#: because "no return because the beam went to the floor" means the bearing is CLEAR out to at
#: least the floor range, while "no return" on its own means nothing at all.
#:
#: Both labels are free: `scan_type == HIT_GROUND` says the beam reached the floor whether or not
#: the return survived the dropout model.
CLASSES = ("solid", "floor", "none_floor", "none_other")
SOLID, FLOOR, NONE_FLOOR, NONE_OTHER = 0, 1, 2, 3
#: Beams with no return at all, either way -- what the range loss must not be scored on.
NONE = NONE_FLOOR


def beam_labels(scan_type, returned):
    """`(B, N)` class indices into `CLASSES`, from the simulator's per-beam hit type and whether the
    beam returned anything.

    Here rather than in the training script because the class list is here: a label derived
    somewhere else can drift from the order the logits are laid out in, and nothing would notice.

        scan_type == HIT_GROUND (3) and returned -> floor
        scan_type == HIT_GROUND     and not      -> none_floor    (the dropout the floor caused)
        anything else and returned               -> solid
        anything else and not                    -> none_other

    `returned` is the observation's own test (`scan < 1 - range_eps`), not `scan_type != HIT_NONE`:
    a beam can have reached something and still be removed by the noise model's dropout, and which
    of the two happened is exactly what the last two classes separate.
    """
    import torch as _t
    st = _t.as_tensor(scan_type)
    ret = _t.as_tensor(returned).to(_t.bool)
    ground = st == 3
    return _t.where(ret, _t.where(ground, _t.full_like(st, FLOOR), _t.full_like(st, SOLID)),
                    _t.where(ground, _t.full_like(st, NONE_FLOOR),
                             _t.full_like(st, NONE_OTHER))).long()


def frontend_spec(width: int = 20, depth_width: int = 56, imu_width: int = 48,
                  n_beams: int = 1081, imu_dim: int = 0) -> dict:
    """The validated configuration, recorded with any checkpoint this produces."""
    out = {"width": int(width), "depth_width": int(depth_width), "imu_width": int(imu_width),
           "n_beams": int(n_beams), "imu_dim": int(imu_dim)}
    for k in ("width", "depth_width", "imu_width"):
        if out[k] < 4:
            raise ValueError(f"frontend {k} {out[k]} must be at least 4")
    if out["n_beams"] < 32:
        raise ValueError(f"frontend n_beams {out['n_beams']} is too few to stride four times")
    if out["imu_dim"] < 1:
        raise ValueError("frontend imu_dim must be the width of the IMU vector it is fed")
    return out


class FrontEnd(nn.Module):
    """`(scan stack, IMU vector) -> (per-beam class logits, denoised range, roll/pitch)`.

    `scan` is (B, K, N) in the observation's own units (range / range_max, no return = 1.0), K the
    stacked frames -- the same tensor the actor's stem is fed, minus any extra channels. `imu` is
    (B, D), whatever the caller declares: `frontend_data.imu_vector` builds it from the proprio
    vector the observation already carries (speed, this step's gyro and accelerometer, and the
    history rows).

    The range output is a **residual** on the newest frame, not a free prediction: the identity is
    what a perfect denoiser mostly is, and a residual head starts there instead of having to learn
    it. It is bounded by `tanh`, so the front-end cannot move a return further than `RANGE_SPAN`
    -- a wall cannot be denoised into open space.
    """

    #: [normalised range] the most the denoiser may move a return. 0.05 x 10 m = 0.5 m, well past
    #: the 7.4 mm + 1 mm/m of sensor noise and past a grazing floor return's error, and far short of
    #: being able to invent or erase a wall.
    RANGE_SPAN = 0.05

    def __init__(self, k_stack: int, spec: dict):
        super().__init__()
        self.spec = dict(spec)
        w, dw, iw = spec["width"], spec["depth_width"], spec["imu_width"]
        n = spec["n_beams"]
        self.k_stack = int(k_stack)
        cin = self.k_stack + 1                      # frames + the beam-angle ramp
        self.register_buffer("beam_angle", torch.linspace(-1.0, 1.0, n)[None, None, :],
                             persistent=False)
        # Down: 1081 -> 541 -> 271 -> 136 -> 68. The only full-resolution operator is the head's
        # 1x1 at the very end.
        self.d1 = nn.Sequential(nn.Conv1d(cin, w, 7, stride=2, padding=3), nn.GELU())
        self.d2 = nn.Sequential(nn.Conv1d(w, w * 2, 5, stride=2, padding=2), nn.GELU())
        self.d3 = nn.Sequential(nn.Conv1d(w * 2, dw, 5, stride=2, padding=2), nn.GELU())
        self.d4 = nn.Sequential(nn.Conv1d(dw, dw, 5, stride=2, padding=2, dilation=1), nn.GELU())
        self.imu = nn.Sequential(nn.Linear(spec["imu_dim"], iw), nn.GELU(), nn.Linear(iw, iw),
                                 nn.GELU())
        # The IMU enters the bottleneck as a per-channel bias, which is where "the car is braking"
        # belongs: it conditions what the scan means, it is not a beam.
        self.imu_to_bottleneck = nn.Linear(iw, dw)
        # Attitude: the bottleneck pooled over bearings, plus the IMU embedding. Both halves are
        # needed -- the scan says where the floor is and the IMU says how the body just moved.
        self.att = nn.Sequential(nn.Linear(2 * dw + iw, iw), nn.GELU(), nn.Linear(iw, 2))
        # Up: 68 -> 136 -> 271 -> 541, each stage concatenating the encoder map of that length.
        self.u3 = nn.Sequential(nn.Conv1d(dw + dw, dw, 3, padding=1), nn.GELU())
        self.u2 = nn.Sequential(nn.Conv1d(dw + w * 2, w * 2, 3, padding=1), nn.GELU())
        # Kernel 1 at the finest stage, 3 deeper down: at 541 columns a 3-tap over 72 channels is
        # 2.8 MFLOP and the whole budget is ~2, while the receptive field it would add is two beams
        # that the encoder has already seen. Measured, not assumed -- see `budget.py`'s row.
        self.u1 = nn.Sequential(nn.Conv1d(w * 2 + w, w, 1), nn.GELU())
        #: A local, full-resolution branch on the raw frames, and the reason it exists is measured.
        #: Without it the head's only full-resolution operator is a 1x1 convolution on a **linearly
        #: upsampled** feature map, so the residual it can express is smooth across beams -- and a
        #: smooth residual cannot cancel per-beam white noise. Trained that way the denoiser sat at
        #: 0.82 cm MAE against the identity's 0.821 for six epochs, and weighting the term ten times
        #: harder did not move it: it was not an optimisation problem. Three taps over the six raw
        #: frames is 117 k multiply-accumulates of a ~1.5 M network and gives the head the
        #: neighbours an average needs.
        self.local = nn.Sequential(nn.Conv1d(self.k_stack, 6, 3, padding=1), nn.GELU())
        self.head = nn.Conv1d(w + 6, len(CLASSES) + 1, 1)   # class logits + 1 range residual
        #: Zero, so an untrained front-end classifies everything at the uniform prior and denoises
        #: to the identity -- the same discipline every other addition in this project keeps.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.att[-1].weight)
        nn.init.zeros_(self.att[-1].bias)

    def forward(self, scan: torch.Tensor, imu: torch.Tensor):
        """`(logits (B, 3, N), range (B, N), attitude (B, 2))`."""
        if scan.dim() != 3:
            raise ValueError(f"scan must be (B, K, N), got {tuple(scan.shape)}")
        if scan.shape[1] != self.k_stack:
            raise ValueError(f"this front-end was built for {self.k_stack} stacked frames and was "
                             f"given {scan.shape[1]}")
        n = scan.shape[2]
        x = torch.cat([scan, self.beam_angle[..., :n].expand(scan.shape[0], -1, -1)], 1)
        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        b = self.d4(e3)
        g = self.imu(imu)
        b = b + self.imu_to_bottleneck(g)[:, :, None]
        pooled = torch.cat([b.mean(2), b.amax(2), g], 1)
        att = self.att(pooled)
        u = self.u3(torch.cat([_up(b, e3.shape[-1]), e3], 1))
        u = self.u2(torch.cat([_up(u, e2.shape[-1]), e2], 1))
        u = self.u1(torch.cat([_up(u, e1.shape[-1]), e1], 1))
        out = self.head(torch.cat([_up(u, n), self.local(scan)], 1))
        c = len(CLASSES)
        rng = (scan[:, 0] + self.RANGE_SPAN * torch.tanh(out[:, c])).clamp(0.0, 1.0)
        return out[:, :c], rng, att


def _up(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="linear", align_corners=False)


def imu_index_spec(spec, ego: bool = True) -> dict:
    """Which proprio columns the front-end's IMU input is made of, and how wide that makes it.

    **Sensors only.** The proprio vector also carries the previous actions and the speed cap; those
    are the policy's own state, and a *sensor* model conditioned on what the policy did last is a
    different object -- it could learn "this driver brakes here, so expect floor" instead of
    learning what a floor return looks like. So the vector is: this step's speed, gyro,
    accelerometer and the VESC roll/pitch, and the same four quantities from each history row.

    The layout is `flatten_obs`' concatenation of `PROPRIO_KEYS` (see `obs.att_index_spec`); a
    history row is `[speed, imu (6), att (2), action (act_dim)]` and only its first nine columns
    are taken.

    `ego` says whether `floor.EgoStateAttitude`'s six columns -- filtered wheel speed, filtered yaw
    rate, longitudinal and lateral acceleration, and the roll/pitch those imply -- are appended.
    They are the physically meaningful form of what the raw rows already contain, and the point of
    handing them over is that the network should not have to rediscover the suspension's calibrated
    map from a history of wheel speeds.
    """
    i_imu = 1 + spec.act_dim * spec.action_history + 1
    head = [0] + list(range(i_imu, i_imu + 8))            # speed, gyro xyz, accel xyz, roll, pitch
    base = i_imu + 8
    row = 1 + 6 + 2 + spec.act_dim
    cols = list(head)
    for r in range(spec.hist_len):
        start = base + r * row
        cols += list(range(start, start + 9))
    return {"proprio_dim": int(spec.proprio_dim), "cols": cols, "ego": bool(ego),
            "ego_dim": 6 if ego else 0, "dim": len(cols) + (6 if ego else 0),
            "v_max": float(spec.v_max), "gyro_scale": float(spec.gyro_scale),
            "accel_scale": float(spec.accel_scale), "att_scale": float(spec.att_scale)}


def imu_vector(proprio: torch.Tensor, idx: dict, ego: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(B, D) the front-end's IMU input, sliced out of the proprio vector by the index spec.

    Left in the observation's own normalisation: these are network inputs, not geometry, and the
    scales are already O(1). The geometry that needs radians is `learn/floor.py`, which multiplies
    them back itself.
    """
    if proprio.dim() != 2:
        raise ValueError(f"proprio must be (B, P), got {tuple(proprio.shape)}")
    if proprio.shape[1] != int(idx["proprio_dim"]):
        raise ValueError(
            f"the front-end was built for a {idx['proprio_dim']}-wide proprio vector and this one "
            f"is {proprio.shape[1]} wide; its columns are read by index. Rebuild it for this "
            f"observation, or feed the observation it was built for.")
    cols = torch.as_tensor(idx["cols"], device=proprio.device, dtype=torch.long)
    out = proprio.index_select(1, cols)
    if not idx.get("ego"):
        return out
    if ego is None:
        raise ValueError(
            "this front-end was built with the ego-state columns and none were passed. They come "
            "from `floor.EgoStateAttitude` -- `state_vector()` and the attitude it returns -- and "
            "substituting zeros would hand the network a parked car.")
    if ego.shape[1] != int(idx["ego_dim"]):
        raise ValueError(f"ego block must be (B, {idx['ego_dim']}), got {tuple(ego.shape)}")
    # The six are SI (m/s, rad/s, m/s^2, rad) and the rest of the vector is O(1)-normalised, so
    # they are put on the same scale here rather than left to the first Linear to discover.
    scale = torch.tensor([1.0 / float(idx["v_max"]), 1.0 / float(idx["gyro_scale"]),
                          1.0 / float(idx["accel_scale"]), 1.0 / float(idx["accel_scale"]),
                          1.0 / float(idx["att_scale"]), 1.0 / float(idx["att_scale"])],
                         device=ego.device, dtype=ego.dtype)
    return torch.cat([out, ego * scale], 1)


def frontend_losses(logits, rng, att, label, clean, att_true, weights=(1.0, 1.0, 1.0),
                    class_weight: Optional[torch.Tensor] = None):
    """`(total, parts)` -- cross-entropy on the class, L1 on the range, L2 on the attitude.

    `label` (B, N) int indexing `CLASSES`; `clean` (B, N) the noise-free normalised range; `att_true`
    (B, 2) rad. The range term is scored only where the beam **returned something** -- a no-return
    beam has no range to denoise and its clean value is `range_max`, which would train the head to
    reach for the ceiling.

    L1 on the range rather than L2 because the error distribution has the spike outliers in it
    (`lidar.spike_prob` replaces a range with a uniform draw) and a squared loss would spend the
    head's capacity on them.
    """
    ce = F.cross_entropy(logits.float(), label.long(), weight=class_weight, reduction="mean")
    ret = (label < NONE_FLOOR).float()                   # solid or floor: the beams that returned
    l1 = ((rng.float() - clean.float()).abs() * ret).sum() / ret.sum().clamp_min(1.0)
    la = ((att.float() - att_true.float()) ** 2).mean()
    total = weights[0] * ce + weights[1] * l1 + weights[2] * la
    with torch.no_grad():
        pred = logits.argmax(1)
        parts = {"ce": ce, "range_l1": l1, "att_mse": la}
        for i, name in enumerate(CLASSES):
            t = label == i
            p = pred == i
            parts[f"recall_{name}"] = (p & t).sum() / t.sum().clamp_min(1)
            parts[f"precision_{name}"] = (p & t).sum() / p.sum().clamp_min(1)
        # The two no-return classes, scored together as well: telling them apart is the new ask,
        # and "did it at least know there was no return" is the older, easier one.
        none_t = label >= NONE_FLOOR
        none_p = pred >= NONE_FLOOR
        parts["recall_none_any"] = (none_p & none_t).sum() / none_t.sum().clamp_min(1)
        parts["precision_none_any"] = (none_p & none_t).sum() / none_p.sum().clamp_min(1)
        parts["att_rmse_roll"] = ((att[:, 0] - att_true[:, 0]) ** 2).mean().sqrt()
        parts["att_rmse_pitch"] = ((att[:, 1] - att_true[:, 1]) ** 2).mean().sqrt()
    return total, parts


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def describe(spec: Optional[dict]) -> str:
    if not spec:
        return "no front-end"
    return (f"front-end: width {spec['width']}/{spec['depth_width']}, imu {spec['imu_dim']} -> "
            f"{spec['imu_width']}, {spec['n_beams']} beams")


# ====================================================================== deployment
def save_frontend(path, model: FrontEnd, spec: dict, imu_index: dict, k_stack: int,
                  extra: Optional[dict] = None) -> None:
    """One file with everything a consumer needs: the weights, the architecture, and the proprio
    column map its IMU input was sliced with."""
    torch.save({"state_dict": model.state_dict(), "spec": dict(spec),
                "imu_index": dict(imu_index), "k_stack": int(k_stack), "extra": extra or {}}, path)


def load_frontend(path, device="cpu"):
    """`(model in eval mode, spec, imu_index, k_stack)`.

    The column map travels with the weights and is not re-derived: a front-end whose IMU input was
    sliced from one proprio layout must not be fed another, and `imu_vector` raises rather than
    reading a prev-action where it expects a gyro.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    spec, idx, k = dict(ck["spec"]), dict(ck["imu_index"]), int(ck["k_stack"])
    m = FrontEnd(k, spec).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    for prm in m.parameters():
        prm.requires_grad_(False)
    return m, spec, idx, k


class FrontEndRuntime:
    """The front-end on one inference path: the scan stack and the ego state in, the three
    per-beam outputs and the attitude out, once per control step.

    Holds no state of its own -- `EgoStateAttitude` is owned by the caller (`obs.ScanAugment`), so
    a path that uses both the geometric channel and the front-end advances one estimator, not two
    that can disagree.
    """

    def __init__(self, path, device="cpu"):
        self.model, self.spec, self.idx, self.k_stack = load_frontend(path, device)
        self.device = torch.device(device)
        self.path = str(path)
        #: The last step's outputs, so a consumer that wants only the attitude (the ROS node's
        #: fallback) does not have to run the network a second time.
        self.last_att: Optional[torch.Tensor] = None
        self.last_probs: Optional[torch.Tensor] = None
        self.last_range: Optional[torch.Tensor] = None

    @torch.no_grad()
    def __call__(self, scan_stack: torch.Tensor, proprio: torch.Tensor,
                 ego: Optional[torch.Tensor] = None):
        """`(class probabilities (B, 3, N), denoised range (B, N), attitude (B, 2))`."""
        k = self.k_stack
        if scan_stack.shape[1] < k:
            raise ValueError(f"this front-end reads {k} stacked frames and the observation has "
                             f"{scan_stack.shape[1]}")
        logits, rng, att = self.model(scan_stack[:, :k].to(self.device),
                                      imu_vector(proprio.to(self.device), self.idx, ego))
        probs = torch.softmax(logits, 1)
        self.last_probs, self.last_range, self.last_att = probs, rng, att
        return probs, rng, att

    def describe(self) -> str:
        return f"{describe(self.spec)} from {self.path}"
