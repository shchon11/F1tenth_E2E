"""Prioritized level replay (Jiang, Grefenstette & Rocktaschel 2021), per race seat.

A *level* is everything a race reset draws -- the venue variant, the obstacle layout, how many opponents
and who drives them at what pace, the grid, the physics randomisation -- and here it is named by the seed
the simulator's generator is set to for that one race's reset (`F1VecEnv._reset_envs` with a
`level_replay`). Every draw of a reset is sized by the number of races and indexed by the race, so a seed
reproduces its level exactly in the race seat it was drawn in; each seat therefore keeps its own buffer.

What gets replayed is not the lowest-return level but the one with the most left to learn: the score is
the mean |GAE advantage| of the learner's steps on it (the paper's "positive value loss" family). A level
nobody can finish has a low return but, once the critic knows that, a low score too -- which is what keeps
the replay off impossible levels. New levels keep arriving (`p_replay`), and a staleness term brings back
levels whose score is old.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


class LevelReplay:
    def __init__(self, n_seats: int, p_replay: float = 0.5, size: int = 24, beta: float = 0.3,
                 rho: float = 0.1, min_fill: int = 4, seed: int = 0):
        if not 0.0 <= p_replay < 1.0:
            raise ValueError(f"p_replay must be in [0, 1), got {p_replay}")
        self.n, self.p, self.size, self.beta, self.rho, self.min_fill = int(n_seats), float(p_replay), int(size), \
            float(beta), float(rho), int(min_fill)
        self.rng = np.random.default_rng(seed)
        #: per seat: seed -> [score, tick last played]
        self.buf: List[Dict[int, list]] = [dict() for _ in range(self.n)]
        #: per seat: the level being played and its running |adv| sum / count
        self.cur = [-1] * self.n
        self.play_sum = [0.0] * self.n
        self.play_cnt = [0] * self.n
        self.tick = 0
        self.n_new = 0
        self.n_replay = 0

    # ------------------------------------------------------------------ choosing
    def choose(self, seat: int) -> int:
        """The seed of the next level for this seat: a replayed one with probability `p` (once the
        buffer holds `min_fill`), else a fresh one."""
        self._commit(seat)
        self.tick += 1
        b = self.buf[seat]
        if len(b) >= self.min_fill and self.rng.random() < self.p:
            seeds = list(b)
            scores = np.array([b[s][0] for s in seeds], dtype=float)
            stale = np.array([self.tick - b[s][1] for s in seeds], dtype=float)
            # ties broken at random: a stable sort would hand rank 1 to whichever tied level came first
            order = np.lexsort((self.rng.random(len(seeds)), -scores))
            rank = np.empty(len(seeds)); rank[order] = np.arange(1, len(seeds) + 1)
            h = (1.0 / rank) ** (1.0 / self.beta); h /= h.sum()
            c = stale / stale.sum() if stale.sum() > 0 else np.full(len(seeds), 1.0 / len(seeds))
            pr = (1.0 - self.rho) * h + self.rho * c
            s = seeds[int(self.rng.choice(len(seeds), p=pr / pr.sum()))]
            self.n_replay += 1
        else:
            s = int(self.rng.integers(1, 2 ** 62))
            self.n_new += 1
        self.cur[seat] = s
        self.play_sum[seat] = 0.0
        self.play_cnt[seat] = 0
        return s

    # ------------------------------------------------------------------ scoring
    def accumulate(self, seat: int, seed: int, abs_adv_sum: float, count: int) -> None:
        """|advantage| of `count` learner steps that were played on `seed` in this seat."""
        if count <= 0:
            return
        if seed == self.cur[seat]:
            self.play_sum[seat] += float(abs_adv_sum); self.play_cnt[seat] += int(count)
        elif seed in self.buf[seat]:
            # the tail of a play that ended inside this rollout: fold it into the stored score
            b = self.buf[seat][seed]
            b[0] = 0.5 * b[0] + 0.5 * float(abs_adv_sum) / count

    def _commit(self, seat: int) -> None:
        """The play that just ended goes into the buffer with its score; the buffer keeps its best `size`."""
        s = self.cur[seat]
        if s < 0 or self.play_cnt[seat] == 0:
            return
        b = self.buf[seat]
        b[s] = [self.play_sum[seat] / self.play_cnt[seat], self.tick]
        if len(b) > self.size:
            worst = min(b, key=lambda k: b[k][0])
            del b[worst]

    def stats(self) -> dict:
        filled = [len(b) for b in self.buf]
        scores = [v[0] for b in self.buf for v in b.values()]
        return {"new": self.n_new, "replayed": self.n_replay, "buffer_mean": float(np.mean(filled)) if filled else 0.0,
                "score_mean": float(np.mean(scores)) if scores else 0.0}
