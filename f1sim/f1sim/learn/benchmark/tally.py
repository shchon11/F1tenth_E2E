"""Denominators that cannot silently shrink, and N/A that is never a zero.

The rule throughout: a trial that was pre-validated and then failed early is a *failure*, not an
exclusion. Only a scenario that was never valid to begin with leaves the denominator, and that is a
setup failure reported against the scenario, not against the system.
"""
from __future__ import annotations
from dataclasses import dataclass, field

#: Outcomes that keep a trial in the primary denominator without crediting it.
FAILURE_REASONS = ("approach_collision", "approach_timeout", "contact", "opponent_respawn",
                   "learner_reset", "terminated", "no_pass", "repassed", "hit")

#: The only way out of a denominator: the scenario itself was invalid before any policy ran.
SETUP_REASONS = ("lead_start", "invalid_geometry")


def na(reason: str) -> dict:
    """A missing measurement. Never rendered as 0."""
    return {"value": None, "reason": reason}


@dataclass
class Tally:
    """Counts for one (system, suite, cell-group) with reasons preserved.

    `expected_n` is the declared trial count for the group. After the suite freeze a group that does
    not deliver exactly that many pre-validated trials cannot be ranked: a shrunken denominator is a
    different measurement, not a worse score.

    `frozen` marks post-freeze scoring. Setup failures are a *geometry-phase* diagnostic; once the
    suite is frozen they are a refusal, because the scenario was supposed to be proven valid before
    any system ran.
    """
    successes: int = 0
    failures: dict = field(default_factory=dict)
    setup_failures: dict = field(default_factory=dict)
    expected_n: int | None = None
    frozen: bool = False

    def record(self, success: bool, reason: str | None = None) -> None:
        if success:
            self.successes += 1
            return
        key = reason or "unspecified"
        if key in SETUP_REASONS:
            if self.frozen:
                raise ValueError(
                    f"{key} after suite freeze: the scenario was validated in the geometry phase, so "
                    f"this cell must be refused rather than scored with a shrunken denominator")
            self.setup_failures[key] = self.setup_failures.get(key, 0) + 1
            return
        self.failures[key] = self.failures.get(key, 0) + 1

    @property
    def denominator(self) -> int:
        """Pre-validated trials: successes + every failure. Setup failures are not in it."""
        return self.successes + sum(self.failures.values())

    @property
    def excluded(self) -> int:
        return sum(self.setup_failures.values())

    def rate(self) -> dict:
        """Primary rate, or N/A with a reason -- never 0/0 rendered as zero, never a short count."""
        n = self.denominator
        if n == 0:
            return na("no pre-validated trials")
        if self.expected_n is not None and n != self.expected_n:
            return na(f"incomplete: {n} of {self.expected_n} declared trials")
        return {"value": self.successes / n, "reason": None}

    def conditional_rate(self, encountered: int) -> dict:
        """Secondary, explicitly conditional (e.g. cleared/encountered).

        Enforces successes <= encountered <= pre-validated n. A success that never encountered the
        obstacle, or more encounters than trials, is a bookkeeping error, not a high score.
        """
        if not 0 <= encountered <= self.denominator:
            raise ValueError(f"encountered {encountered} outside [0, {self.denominator}]")
        if self.successes > encountered:
            raise ValueError(f"{self.successes} successes but only {encountered} encounters")
        if encountered == 0:
            return na("no encounters")
        return {"value": self.successes / encountered, "reason": None}

    def as_dict(self) -> dict:
        return {"successes": self.successes,
                "expected_n": self.expected_n,
                "complete": self.expected_n is None or self.denominator == self.expected_n,
                "denominator": self.denominator,
                "rate": self.rate(),
                "failures": dict(sorted(self.failures.items())),
                "setup_failures": dict(sorted(self.setup_failures.items())),
                "excluded_from_denominator": self.excluded}
