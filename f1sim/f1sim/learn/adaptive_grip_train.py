"""Bounded CPU-only identifiability pilot for adaptive_grip_v2 (never auto-promotes).

Example: python -m f1sim.learn.adaptive_grip_train --out /path/to/new/pilot
The collector uses current Simulator physics/IMU/wheel odometry with only unused LiDAR
raycasting skipped. Truth is isolated to targets, never appended to sensor features.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .. import dynamics as dyn
from ..params import Config
from ..sim import Simulator
from ..track import Track
from .adaptive_grip import (AdaptiveConsumption, AdaptiveGripEstimator, AdaptiveGripNet,
                           adaptive_feature_spec, save_adaptive_grip_estimator)
from .grip_estimator import SensorHistory, file_sha256, pinball_loss
from .obs import norm_att, norm_imu, norm_speed

GATES = {"mae_reduction_vs_prior": 0.20, "q10_exceedance_max": 0.15,
         "confidence_auroc_min": 0.75, "low_excitation_fpr_max": 0.15,
         "minimum_informative_windows_per_split": 80,
         "minimum_informative_episodes_per_split": 8,
         "minimum_admitted_windows": 100, "minimum_admitted_episodes": 8,
         "admitted_q10_exceedance_max": 0.15}


def sensor_observation(result):
    """Same three channels/scales as gym_env._obs; wheel odometry, not result.state."""
    if result.imu is None or result.imu.shape[1] == 0 or result.imu_att is None:
        raise ValueError("collector requires actual simulated IMU samples")
    imu = result.imu.mean(1)
    return {"speed": norm_speed(result.odom[:, 3:4], 10.0),
            "imu": norm_imu(imu, 5.0, 10.0),
            "imu_att": norm_att(result.imu_att[:, :2], 0.35)}


def privileged_targets(sim):
    """End-of-step axle utilization targets; small slip alone is never informative.

    Reuses current tire formula and load-transfer equations. These are sampled endpoint
    targets, not a force sensor, and do not resolve substep-only saturation events.
    """
    state, p = sim.state, sim.P
    vx, vy, yaw_rate, steer = [state[:, i] for i in (dyn.IVX, dyn.IVY, dyn.IR, dyn.ISTEER)]
    vx_safe = torch.where(vx.abs() < 0.3, torch.full_like(vx, 0.3) * torch.sign(vx + 1e-9), vx)
    af = steer - torch.atan2(vy + p["lf"] * yaw_rate, vx_safe)
    ar = -torch.atan2(vy - p["lr"] * yaw_rate, vx_safe)
    kappa = (state[:, dyn.IOMEGA] * p["r_w"] - vx) / vx.abs().clamp_min(p["v_slip_eps"])
    uf = dyn.pacejka(af, p["B_f"], p["C_f"], p["E_f"]).abs()
    ux = dyn.pacejka(kappa, p["B_x"], p["C_x"], p["E_x"])
    uy = dyn.pacejka(ar, p["B_r"], p["C_r"], p["E_r"])
    ur = torch.sqrt(ux.square() + uy.square()).clamp_max(1.0)
    utilization = torch.maximum(uf, ur)
    # Endpoint near the tire-force peak, at a speed where dynamic tire equations dominate.
    informative = (utilization >= 0.92) & (vx.abs() >= p["v_blend_max"])
    low_excitation = (utilization <= 0.50)
    # Realized acceleration is a lower bound, not a point label for available friction.
    # Simulator may permit a front/rear approximate force model: cap at its known true limit.
    lower_bound = (0.90 * torch.sqrt(sim.ax.square() + sim.ay.square()) / dyn.G).clamp_min(0)
    lower_bound = torch.minimum(lower_bound, p["mu"])
    return p["mu"].clone(), informative, low_excitation, lower_bound, utilization


def _empty_track():
    occ = np.zeros((512, 512), dtype=bool)
    occ[[0, -1], :] = True
    occ[:, [0, -1]] = True
    return Track.from_occupancy(occ, 0.5, origin=(-128.0, -128.0), name="adaptive_grip_empty_256m")


def collect_split(episodes, steps, seed, spec, stride=4):
    """Split owns a disjoint simulator seed before any windows are extracted."""
    cfg = Config()
    cfg.sim.device, cfg.sim.compile, cfg.sim.seed = "cpu", False, seed
    cfg.vehicle.wheel_model, cfg.imu.enabled = True, True
    cfg.lidar.n_beams = 8
    cfg.rand.ranges["vehicle.mu"] = (spec.mu_min / cfg.vehicle.mu, spec.mu_max / cfg.vehicle.mu)
    sim = Simulator(_empty_track(), cfg, episodes, "cpu")
    # No estimator feature uses LiDAR. Skip raycasting only; retain step physics, collision,
    # sensor scheduling, VESC odometry, and IMU generation without altering their models.
    empty_scan = torch.full((episodes, cfg.lidar.n_beams), float("inf"))
    sim.lidar.scan = lambda *args, **kwargs: (empty_scan, empty_scan, torch.zeros_like(empty_scan, dtype=torch.int32))
    generator = torch.Generator().manual_seed(seed + 100003)
    phase = torch.rand(episodes, generator=generator) * (2 * math.pi)
    modes = torch.arange(episodes) % 4
    amp = 0.08 + torch.rand(episodes, generator=generator) * 0.28
    cruise = 4.5 + torch.rand(episodes, generator=generator) * 5.0
    sim.reset(poses=torch.zeros(episodes, 3), speed=torch.full((episodes,), 1.0))
    histories = SensorHistory(episodes, "cpu", spec)
    positive_history = torch.zeros(episodes, 8, dtype=torch.bool)
    low_history = torch.zeros_like(positive_history)
    data = {k: [] for k in ("features", "valid", "mu", "informative", "low_excitation", "lower_bound", "episode", "step", "utilization")}
    collisions = 0
    mu_per_episode = sim.P["mu"].tolist()
    params = {key: val.tolist() for key, val in sim.P.items() if key in (
        "mu", "m", "Iz", "cmd_delay", "servo_tau", "motor_tau", "speed_gain", "steer_gain",
        "B_x", "B_f", "B_r", "I_w", "drive_split_r", "road_tilt", "roll_per_g", "pitch_per_g")}
    for step in range(steps):
        t = step * spec.control_dt
        # Gentle episodes provide actual low-excitation negatives under every friction draw.
        gentle_speed = 1.0 + 0.25 * t
        ramp_speed = min(1.0 + 2.8 * t, 10.5)
        requested_speed = torch.minimum(cruise, torch.full_like(cruise, ramp_speed))
        requested_speed = torch.where(modes == 0, torch.full_like(cruise, gentle_speed), requested_speed)
        requested_speed = torch.where((modes == 3) & (step > steps * 0.72), torch.ones_like(cruise), requested_speed)
        ramp = min(1.0, max(0.0, (t - 0.65) / 0.7))
        steer = amp * ramp * torch.sin(phase + 0.7 * t + 0.30 * t * t)
        steer = torch.where(modes == 0, 0.015 * torch.sin(phase + t), steer)
        steer = torch.where(modes == 1, 0.15 * steer, steer)
        cmd = torch.stack([steer, requested_speed], 1)
        result = sim.step(cmd)
        obs = sensor_observation(result)
        history_reset = result.collision
        if bool(history_reset.any()):
            collisions += int(history_reset.sum())
            # End-of-episode samples are discarded; never mix new mu with old sensor rows.
            histories.valid[history_reset] = False
            positive_history[history_reset] = False
            low_history[history_reset] = False
        histories.push(obs, cmd)
        mu, positive, low, lower, utilization = privileged_targets(sim)
        positive_history = positive_history.roll(1, 1)
        low_history = low_history.roll(1, 1)
        positive_history[:, 0], low_history[:, 0] = positive, low
        valid_episode = ~result.collision
        if step >= 7 and step % stride == 0:
            mask = valid_episode & (histories.valid_count() >= spec.warm_frames)
            values = {"features": histories.features, "valid": histories.valid,
                      "mu": mu, "informative": positive_history.sum(1) >= 3,
                      "low_excitation": low_history.sum(1) >= 6, "lower_bound": lower,
                      "episode": torch.arange(episodes) + seed * 1000,
                      "step": torch.full((episodes,), step), "utilization": utilization}
            for key, value in values.items():
                data[key].append(value[mask].clone())
        # A 256 m empty square should not collide in 5 s; fail instead of changing the dataset.
        if collisions:
            raise RuntimeError("collector collision: episode boundary invalidated; no pilot claim")
    data = {key: torch.cat(value) for key, value in data.items()}
    manifest = {"seed": seed, "episode_count": episodes, "steps_per_episode": steps,
                "windows": len(data["mu"]), "collisions": collisions, "mu_per_episode": mu_per_episode,
                "sampled_parameters": params, "config": asdict(cfg), "raycasting_skipped": True,
                "truth_only_targets": ["mu", "informative", "low_excitation", "lower_bound", "utilization"],
                "informative_windows": int(data["informative"].sum()),
                "informative_episodes": len(data["episode"][data["informative"]].unique()),
                "low_excitation_windows": int(data["low_excitation"].sum())}
    return data, manifest


def adaptive_loss(q, logits, data):
    positive = data["informative"].bool()
    quantile = pinball_loss(q[positive], data["mu"][positive]) if bool(positive.any()) else q.sum() * 0
    negative = ~positive
    # Censoring only: larger predicted friction above utilized force is not penalized.
    censored = (nn.functional.relu(data["lower_bound"][negative, None] - q[negative]).mean()
                if bool(negative.any()) else q.sum() * 0)
    confidence = nn.functional.binary_cross_entropy_with_logits(logits, positive.float())
    return quantile + 0.15 * censored + 0.35 * confidence


@torch.no_grad()
def predictions(net, data, batch_size=256):
    qs, cs = [], []
    net.eval()
    for offset in range(0, len(data["mu"]), batch_size):
        q, logit = net.distribution(data["features"][offset:offset + batch_size], data["valid"][offset:offset + batch_size])
        qs.append(q)
        cs.append(logit.sigmoid())
    return torch.cat(qs), torch.cat(cs)


def auroc(scores, positive):
    a, b = scores[positive], scores[~positive]
    if not len(a) or not len(b):
        return None
    # Exact tie-aware Mann-Whitney statistic without the O(N_positive*N_negative) matrix.
    sorted_negative = b.sort().values
    lower = torch.searchsorted(sorted_negative, a, right=False)
    upper = torch.searchsorted(sorted_negative, a, right=True)
    return float(((lower + upper).double() * .5).sum() / (len(a) * len(b)))


def evaluate(q, confidence, data, prior, delta, threshold=0.65):
    positive, low = data["informative"], data["low_excitation"]
    y = data["mu"]
    prior_mae = float((y[positive] - prior).abs().mean())
    model_mae = float((y[positive] - q[positive, 1]).abs().mean())
    bound = (q[:, 0] - delta).clamp(0.05, adaptive_feature_spec().mu_max)
    selected = confidence >= threshold
    return {"informative_windows": int(positive.sum()), "informative_mu_mae": model_mae,
            "eligible_windows": len(y), "low_excitation_windows": int(low.sum()),
            "nominal_mu_mae": float((y[positive] - AdaptiveConsumption().nominal_mu).abs().mean()),
            "training_prior_mu": prior, "training_prior_mu_mae": prior_mae,
            "mae_reduction_vs_prior": 1 - model_mae / max(prior_mae, 1e-9),
            "q10_exceedance": float((bound[positive] > y[positive]).float().mean()),
            "q10_coverage": float((bound[positive] <= y[positive]).float().mean()),
            "q10_mean_slack": float((y[positive] - bound[positive]).mean()),
            "confidence_auroc": auroc(confidence, positive),
            "low_excitation_fpr": float(selected[low].float().mean()) if bool(low.any()) else None,
            "confidence_true_positive_rate": float(selected[positive].float().mean()),
            "confidence_selection_rate": float(selected.float().mean()),
            "selected_q10_exceedance": float((bound[selected] > y[selected]).float().mean()) if bool(selected.any()) else None,
            "selected_windows": int(selected.sum()),
            "selected_episodes": len(data["episode"][selected].unique()),
            "informative_q50_std": float(q[positive, 1].std()),
            "informative_true_mu_std": float(y[positive].std())}


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ["learn/adaptive_grip.py", "learn/adaptive_grip_train.py", "learn/grip_estimator.py",
             "sim.py", "dynamics.py", "actuators.py", "imu.py", "odom.py", "params.py", "randomization.py"]
    return {name: file_sha256(root / name) for name in names}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=128)
    parser.add_argument("--cal-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args(argv)
    if min(args.train_episodes, args.cal_episodes, args.test_episodes) < 8 or not 40 <= args.steps <= 240 or not 1 <= args.epochs <= 10:
        parser.error("bounded pilot needs >=8 episodes/split, 40..240 steps, 1..10 epochs")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError("use a new output directory; never overwrite a pilot")
    torch.set_num_threads(1)
    torch.manual_seed(170923)
    start = time.monotonic()
    spec = adaptive_feature_spec()
    prereg = {"format": "adaptive_grip_v2_cpu_pilot", "gates": GATES, "training_seed": 170923,
              "split_seeds": {"train": 17101, "cal": 17201, "test": 17301},
              "training": {"epochs": args.epochs, "optimizer": "AdamW", "lr": 0.002,
                           "batch_size": 128, "hidden": 64, "window_stride": 4},
              "feature_spec": spec.to_meta(), "consumption": asdict(AdaptiveConsumption()),
              "source_sha256_before": _source_hashes(),
              "selection": "final epoch only; no TEST tuning; no model promotion",
              "mae_baseline": "TRAIN informative-mu median fixed before TEST, same TEST informative subset",
              "informative_target": "axle tire-force utilization >=0.92 at dynamic speed in >=3 of last8 frames",
              "low_excitation_target": "axle tire-force utilization <=0.50 in >=6 of last8 frames",
              "calibration": "max(0, CAL informative q90 of raw_q10 - mu), higher interpolation",
              "limitations": ["simulation only", "endpoint tire-utilization targets are approximate",
                              "uniform surface per episode", "no real-car or race closed-loop validation"]}
    (args.out / "predeclared.json").write_text(json.dumps(prereg, indent=2))
    datasets, manifests = {}, {}
    for name, count in (("train", args.train_episodes), ("cal", args.cal_episodes), ("test", args.test_episodes)):
        data, manifest = collect_split(count, args.steps, prereg["split_seeds"][name], spec)
        datasets[name], manifests[name] = data, manifest
        torch.save(data, args.out / f"{name}.pt")
        (args.out / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"collected {name}: {manifest['windows']} windows; {manifest['informative_windows']} informative; {manifest['low_excitation_windows']} low-excitation", flush=True)
    coverage_ok = all(m["informative_windows"] >= GATES["minimum_informative_windows_per_split"]
                      and m["informative_episodes"] >= GATES["minimum_informative_episodes_per_split"]
                      and m["low_excitation_windows"] >= 80 for m in manifests.values())
    if not coverage_ok:
        report = {"status": "collector_coverage_failure", "promoted": False, "manifests": manifests}
        (args.out / "report.json").write_text(json.dumps(report, indent=2))
        print("collector coverage failed; stopped before training", flush=True)
        return 2
    train = datasets["train"]
    # Re-seed after collection, since Simulator owns/reset its own global sensor RNG.
    torch.manual_seed(prereg["training_seed"])
    net = AdaptiveGripNet(spec)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.002)
    curve = []
    for epoch in range(args.epochs):
        net.train()
        order = torch.randperm(len(train["mu"]))
        losses = []
        for ix in order.split(128):
            batch = {k: v[ix] for k, v in train.items()}
            q, logits = net.distribution(batch["features"], batch["valid"])
            loss = adaptive_loss(q, logits, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        curve.append(sum(losses) / len(losses))
        print(f"epoch {epoch + 1}/{args.epochs}: loss={curve[-1]:.5f}", flush=True)
    cal, test = datasets["cal"], datasets["test"]
    cal_q, cal_confidence = predictions(net, cal)
    delta = max(0.0, float(torch.quantile(cal_q[cal["informative"], 0] - cal["mu"][cal["informative"]], 0.90, interpolation="higher")))
    calibration = {"cal_delta": delta, "target_exceedance": 0.10, "split_seed": prereg["split_seeds"]["cal"],
                   "informative_windows": int(cal["informative"].sum()), "method": "downward_only_empirical_cal_quantile"}
    prior = float(train["mu"][train["informative"]].median())
    test_q, test_confidence = predictions(net, test)
    metrics = evaluate(test_q, test_confidence, test, prior, delta)
    checks = {"mae": metrics["mae_reduction_vs_prior"] >= GATES["mae_reduction_vs_prior"],
              "coverage": metrics["q10_exceedance"] <= GATES["q10_exceedance_max"],
              "confidence": metrics["confidence_auroc"] is not None and metrics["confidence_auroc"] >= GATES["confidence_auroc_min"],
              "false_positive": metrics["low_excitation_fpr"] is not None and metrics["low_excitation_fpr"] <= GATES["low_excitation_fpr_max"]}
    checks["admission_count"] = (metrics["selected_windows"] >= GATES["minimum_admitted_windows"]
                                 and metrics["selected_episodes"] >= GATES["minimum_admitted_episodes"])
    checks["admitted_coverage"] = (metrics["selected_q10_exceedance"] is not None
                                   and metrics["selected_q10_exceedance"] <= GATES["admitted_q10_exceedance_max"])
    after = _source_hashes()
    source_stable = prereg["source_sha256_before"] == after
    meta = {"pilot_only": True, "promoted": False, "deployment_approved": False,
            "pilot_gates_passed": all(checks.values()) and source_stable,
            "training": prereg, "test_metrics": metrics,
            "gates": checks, "source_stable_during_pilot": source_stable}
    sha = save_adaptive_grip_estimator(args.out / "candidate.pt", net, spec, calibration, meta)
    report = {"status": "pilot_gates_pass" if all(checks.values()) and source_stable else "pilot_not_advanced",
              "promoted": False, "test": metrics, "cal": evaluate(cal_q, cal_confidence, cal, prior, delta),
              "gates": checks, "calibration": calibration, "training_loss": curve,
              "elapsed_seconds": time.monotonic() - start, "candidate_sha256": sha,
              "source_sha256_after": after, "source_stable_during_pilot": source_stable,
              "split_manifests": {name: f"{name}_manifest.json" for name in manifests},
              "limitations": prereg["limitations"]}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({"quantiles": test_q, "confidence": test_confidence, "mu": test["mu"],
                "informative": test["informative"], "episode": test["episode"]}, args.out / "test_predictions.pt")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
