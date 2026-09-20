# Run `cl_future_s701` (2026-09-14 03:14): the long run with the future head, no leash

Launched from main `c143fd8` after `cl_long_s701` was stopped at update ~330 (64-trial proxy flat:
u32 211/58, u64 206/48, u96 207/44 against the original's 188/38; `kl_ref` past 1.0). User:
"이번 건 끄고 예측 헤드 넣은 걸로 가자", "KL은 풀어라".

Same as `run-recipe-long-2026-09-13.md` except:

| flag | long run | this run |
|---|---|---|
| `--kl-coef` | 0.01 | **0.0** (no leash to the original) |
| `--lr` / `--lr-end` | 1e-4 / 3e-5 | 5e-5 / 2e-5 |
| `--aux-future` | — | **1.0**: predict the nearest opponent's relative position/velocity, presence, and the ego's speed/yaw rate 0.5 s ahead from the GRU state (`future-head-2026-09-14.md`) |

What the future head's smoke said before this run: the head trains and costs the PPO objective
nothing, but at smoke scale the hidden state carries the ego's own future well (R² 0.82–0.87) and the
opponent's hardly at all (R² 0.0–0.3, seed-dependent), and an offline fit memorises rather than
generalises. This run is the test of whether 8.4M steps change that: run `python -m
f1sim.learn.probe_hidden` on the selected checkpoint against the frozen original.

Selection: `select_mid.sh` (64-trial proxy on CPU, every saved checkpoint), then the 128-trial proxy
on the top three → `~/f1sim_runs/cl_future_s701/ppo_best.pt`,
`work/learning-next/future-20260914/select/selection.md`. Reference rows on the 64-trial proxy:
original 188/512 · 38/96, mem_u8 233 · 47.
