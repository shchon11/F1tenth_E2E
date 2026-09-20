"""Small, explicit training-only guards for adapting an 8D recurrent plan actor."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch

from .memory import reset_hidden


def reference_identity(path: str) -> dict:
    p = Path(path).expanduser().resolve(strict=True)
    payload = p.read_bytes()
    ck = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    return {"path": str(p), "sha256": hashlib.sha256(payload).hexdigest(),
            "meta": dict(ck.get("meta") or {}),
            "source": {k: (ck.get("extra") or {}).get(k) for k in
                       ("phase", "run", "update", "total_steps")}}


def verified_reference_stream(identity: dict) -> io.BytesIO:
    """Hash and load the SAME bytes, refusing a swapped reference checkpoint."""
    payload = Path(identity["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != identity["sha256"]:
        raise ValueError("the original reference checkpoint hash changed before actor load")
    return io.BytesIO(payload)


def resolve_reference(path: str, previous: dict | None) -> dict:
    """Keep the original reference on resume and detect an altered checkpoint file."""
    if previous:
        chosen = path or previous["path"]
        identity = reference_identity(chosen)
        if identity["sha256"] != previous["sha256"]:
            raise ValueError("the original reference checkpoint hash changed on resume")
        return dict(previous)
    if not path:
        raise ValueError("adaptive training needs --reference pointing to the original D3 checkpoint")
    return reference_identity(path)


def require_exact_actor(model, checkpoint: dict) -> None:
    """A critic may migrate; the D3 actor must load every named tensor exactly."""
    saved = {name: value for name, value in checkpoint["state_dict"].items()
             if name.startswith("actor.")}
    current = {name: value for name, value in model.state_dict().items()
               if name.startswith("actor.")}
    if saved.keys() != current.keys() or any(
            saved[name].shape != value.shape or
            not torch.equal(saved[name].to(value.device), value)
            for name, value in current.items()):
        raise RuntimeError("adaptive initialization did not load every actor tensor exactly")


def stage_schedule(stage: str, total_steps: float, step_size: int, origin_total_steps: int,
                   hyperparameters: dict, previous: dict | None = None,
                   checkpoint_total_steps: int | None = None,
                   checkpoint_stage_steps: int | None = None) -> tuple[dict, int]:
    """Seal a stage budget and recover its exact update offset from a checkpoint.

    ``total_steps`` is the complete stage budget, not an additional resume-leg budget.
    The environment/RNG state is intentionally not restored; resumption starts at the
    next PPO update boundary with the original schedule and fresh episode states.
    """
    budget = int(total_steps)
    if budget != total_steps or budget <= 0 or step_size <= 0 or budget % step_size:
        raise ValueError("adaptive --total must be a positive multiple of horizon * learner envs")
    if hyperparameters["kl_decay"] <= 0 or hyperparameters["cap_steps"] <= 0:
        raise ValueError("adaptive KL decay and cap steps must be positive")
    schedule = {"stage": stage, "total_steps": budget, "step_size": int(step_size),
                "origin_total_steps": int(origin_total_steps),
                "hyperparameters": dict(hyperparameters)}
    if previous is None:
        return schedule, 0
    if schedule != previous:
        raise ValueError("adaptive stage schedule or training hyperparameters changed on resume")
    if checkpoint_total_steps is None:
        raise ValueError("adaptive resume checkpoint has no global total_steps")
    completed = int(checkpoint_total_steps) - schedule["origin_total_steps"]
    if completed < 0 or completed > budget or completed % step_size:
        raise ValueError("adaptive checkpoint stage progress is outside its declared update budget")
    if checkpoint_stage_steps is None or int(checkpoint_stage_steps) != completed:
        raise ValueError("adaptive checkpoint stage_steps disagrees with global total_steps")
    return schedule, completed


def stage_lr_kl(schedule: dict, completed_steps: int) -> tuple[float, float]:
    """Learning rate and KL coefficient for the next update of this stage."""
    p = schedule["hyperparameters"]
    fraction = completed_steps / schedule["total_steps"]
    lr = p["lr"] + (p["lr_end"] - p["lr"]) * fraction
    kl = p["kl_coef"] * max(0.0, 1.0 - completed_steps / p["kl_decay"])
    return lr, kl


class RecurrentReference:
    """A frozen actor whose hidden state belongs only to the reference rollout."""

    def __init__(self, actor, batch: int, device):
        self.actor = actor.eval()
        for param in self.actor.parameters():
            param.requires_grad_(False)
        self.hidden = actor.initial_hidden(batch, device=device)

    @torch.no_grad()
    def step(self, scan, proprio, cond=None):
        dist, self.hidden = self.actor.step_dist(scan, proprio, cond, self.hidden)
        if self.hidden is not None:
            self.hidden = self.hidden.float()
        return dist.mean.float().detach(), dist.stddev.float().detach()

    def reset(self, done):
        self.hidden = reset_hidden(self.hidden, done)


class SpeedRowFreeze:
    """Train only the final two action rows while preserving exact frozen actor weights."""

    def __init__(self, model):
        actor = model.actor
        if actor.mu.out_features != 8 or actor.log_std.numel() != 8:
            raise ValueError("speed adaptation requires the 8D plan actor")
        for param in actor.parameters():
            param.requires_grad_(False)
        for param in (actor.mu.weight, actor.mu.bias, actor.log_std):
            param.requires_grad_(True)
        self.params = (actor.mu.weight, actor.mu.bias, actor.log_std)
        self.frozen_rows = tuple(param[:6].detach().clone() for param in self.params)

    def before_step(self):
        for param in self.params:
            if param.grad is not None:
                param.grad[:6].zero_()

    @torch.no_grad()
    def after_step(self):
        for param, frozen in zip(self.params, self.frozen_rows):
            param[:6].copy_(frozen)
