"""Conditioning vector for the Stage-1 friction utility experiment.

The question this exists to answer is narrow: does a policy *trained* with the true current friction
as an input drive better than an identical policy trained with that input held at zero? Both arms
share one architecture, one checkpoint and one budget; only the contents of `c` differ.

    A0  cond_source="zero"      c = 0
    A1  cond_source="true_mu"   c = (mu - 1.0) / 0.25

`mu` here is the per-car friction multiplier `Simulator.P["mu"]`, which is constant within an episode
in today's environment. The normalization is fixed rather than learned so that the two arms and any
later checkpoint mean the same thing by the same number: `P["mu"]` is drawn over roughly
[0.734, 1.154] (`params.py` `vehicle.mu` scaled by its nominal), so `(mu - 1) / 0.25` lands in about
[-1.06, +0.62]. The offset and scale are recorded in the checkpoint; nothing infers them.

**A1 is a laboratory arm.** Its input is privileged, it has no deployment path, and the checkpoint
says so (`lab_oracle=True`) so that a viewer, export or ROS load refuses it rather than running a
policy whose input the car cannot produce.

What this deliberately does NOT contain: remaining-margin features, spatial fields, per-axle values,
or anything the car has not already driven over. Those belong to Stage 2 and would confound the
question Stage 1 asks.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import torch

#: Fixed affine map from `P["mu"]` to the conditioning scalar. Changing either number changes what a
#: checkpoint's `c` means, so both are stored in the checkpoint and compared on load.
MU_OFFSET = 1.0
MU_SCALE = 0.25

#: How `c` is filled at rollout time.
SOURCES = ("zero", "true_mu")

#: What a conditional checkpoint claims about itself.
KIND_MU = "current_mu"


@dataclass(frozen=True)
class CondSpec:
    """What a conditional model's `c` is, recorded in the checkpoint and checked on load."""
    dim: int = 0                        # 0 = legacy unconditional model
    kind: str = ""                      # "" when dim == 0, else KIND_MU
    offset: float = MU_OFFSET
    scale: float = MU_SCALE
    source: str = ""                    # the arm this checkpoint was trained as: "zero" | "true_mu"
    lab_oracle: bool = False            # True when `c` is privileged and cannot be produced on the car

    def to_meta(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_meta(m: Optional[dict]) -> "CondSpec":
        """Parse and **validate**. Unknown keys are an error, not something to drop quietly.

        A checkpoint's conditioning metadata is the only record of what its numbers meant. Silently
        ignoring a field written by a future version, or accepting a spec whose parts disagree, is
        how an arm ends up evaluated against a normalization it was not trained with.
        """
        if not m:
            return CondSpec()
        unknown = set(m) - set(CondSpec.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unsupported conditioning metadata keys {sorted(unknown)}; this build "
                             f"knows {sorted(CondSpec.__dataclass_fields__)}")
        spec = CondSpec(**m)
        spec.validate()
        return spec

    def validate(self) -> "CondSpec":
        """Reject anything this build cannot interpret, rather than guessing."""
        if self.dim == 0:
            if self.kind or self.source or self.lab_oracle:
                raise ValueError(f"cond_dim 0 must carry no kind/source/lab_oracle, got {self}")
            return self
        if self.dim != 1 or self.kind != KIND_MU:
            raise ValueError(f"this build supports only dim=1 kind={KIND_MU!r}, got "
                             f"dim={self.dim} kind={self.kind!r}")
        if self.source not in SOURCES:
            raise ValueError(f"cond source must be one of {SOURCES}, got {self.source!r}")
        if self.lab_oracle != (self.source == "true_mu"):
            raise ValueError(f"lab_oracle={self.lab_oracle} disagrees with source={self.source!r}: "
                             f"only the true-friction arm is a lab oracle")
        if float(self.offset) != MU_OFFSET or float(self.scale) != MU_SCALE:
            raise ValueError(f"checkpoint normalization (mu - {self.offset}) / {self.scale} differs "
                             f"from this build's (mu - {MU_OFFSET}) / {MU_SCALE}; the same number "
                             f"would mean a different friction")
        if float(self.scale) == 0.0:
            raise ValueError("cond scale must be nonzero")
        return self

    def describe(self) -> str:
        if self.dim == 0:
            return "unconditional"
        return (f"{self.kind} dim={self.dim} (mu - {self.offset}) / {self.scale}, "
                f"trained as source={self.source}" + (", LAB ORACLE" if self.lab_oracle else ""))


def spec_for(source: str) -> CondSpec:
    """The spec for a Stage-1 arm. Both arms are dim 1; only `source` and `lab_oracle` differ."""
    if source not in SOURCES:
        raise ValueError(f"cond source must be one of {SOURCES}, got {source!r}")
    return CondSpec(dim=1, kind=KIND_MU, offset=MU_OFFSET, scale=MU_SCALE, source=source,
                    lab_oracle=(source == "true_mu")).validate()


def mu_to_c(mu: torch.Tensor, spec: CondSpec) -> torch.Tensor:
    """(B,) friction -> (B,1) conditioning scalar."""
    return ((mu.reshape(-1) - spec.offset) / spec.scale)[:, None]


def make_condition(source: str, spec: CondSpec, priv: torch.Tensor, mu_index: int) -> torch.Tensor:
    """Build `c` for one rollout step from the RAW privileged vector.

    Raw, not adapted: `mu_index` indexes `env.privileged()` as the env produces it. The critic input
    adapter below changes the width the critic sees and would move this column.
    """
    if spec.dim == 0:
        raise ValueError("make_condition called on an unconditional spec")
    if source == "zero":
        return torch.zeros(priv.shape[0], spec.dim, device=priv.device, dtype=priv.dtype)
    if source == "true_mu":
        return mu_to_c(priv[:, mu_index], spec).to(priv.dtype)
    raise ValueError(f"unknown cond source {source!r}")


# ---------------------------------------------------------------- critic input adapter
#: Where the nearest-opponent block sits in `env.privileged()`: after the 8 dynamic-state values
#: (`gym_env.privileged` / `_priv`, and `gym_env.priv_mu_index` is `8 + (4 if M > 1 else 0)`).
OPP_BLOCK_AT = 8
OPP_BLOCK_WIDTH = 4


def insert_absent_opponent_columns(priv: torch.Tensor) -> torch.Tensor:
    """Solo privileged (17) -> race-shaped privileged (21), with the opponent block zeroed.

    The frozen checkpoint's critic was trained in a race environment, so its privileged input is 21
    wide: 8 dynamic + 4 nearest-opponent + 8 randomized params + 1 speed cap. A solo environment
    produces 17 -- the same vector without the opponent block. Feeding the 17 directly would either
    fail on shape or, worse, silently reinterpret the parameter columns as opponent state.

    Inserting four zeros at the block's own position preserves every trained weight. It is a
    **compatibility padding convention, not a semantic encoding of "no opponent"** -- in the race
    vector those columns are offsets and a scaled distance, so a zero there reads as an opponent at
    zero distance, which is the opposite of absent. Nothing here claims the critic interprets the
    padding correctly; the claim is narrower and is the one the experiment needs: **both arms and
    the critic warm-up use the identical convention**, so whatever bias it introduces is common to
    the comparison. It is experiment-only and recorded in the run metadata.

    Only this one layout is handled. Any other width mismatch is a different problem and is rejected.
    """
    if priv.shape[1] != 17:
        raise ValueError(
            f"the absent-opponent adapter maps 17 -> 21 only; got {priv.shape[1]}. Any other "
            f"privileged width is an unexplained layout mismatch and must be diagnosed, not padded.")
    zeros = priv.new_zeros(priv.shape[0], OPP_BLOCK_WIDTH)
    return torch.cat([priv[:, :OPP_BLOCK_AT], zeros, priv[:, OPP_BLOCK_AT:]], 1)


#: Name recorded in the checkpoint meta and applied inside `Critic.forward`. Putting it there rather
#: than at the call sites means every value path -- rollout, minibatch, truncation bootstrap and the
#: final bootstrap -- is covered by construction, and raw privileged storage and the `mu` / aux label
#: indices stay exactly as the env produced them. `None` is strict legacy: no adaptation, any width
#: mismatch raises from the linear layer as before.
ADAPTER_ABSENT_OPPONENT = "absent_opponent_17_to_21"

PRIV_ADAPTERS = {
    ADAPTER_ABSENT_OPPONENT: insert_absent_opponent_columns,
}


def get_priv_adapter(name: Optional[str]):
    if not name:
        return None
    if name not in PRIV_ADAPTERS:
        raise ValueError(f"unknown privileged adapter {name!r}; known: {sorted(PRIV_ADAPTERS)}")
    return PRIV_ADAPTERS[name]
