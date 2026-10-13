"""Which column of a plan is which: `mpc.plan_layout`, and the guard that keeps it the only answer.

"linear" (six curvature knots, two speeds) and "wp4" (four waypoint curvatures, four speeds) are both
8 numbers wide, so a slice written for one -- the last two as the speeds, the first six as the path --
reads the other as a plausible-looking wrong plan and no shape ever complains. Before 2026-10-14 the
layers that touch a plan did exactly that: `[..., -2:]` in the env, the teachers and the benchmark
experts, `[:, :6]` in the PPO KL, `SPEED_COLS = slice(N_KNOTS, ACT_DIM)` in DAgger, `== 8` in the
controller and the bag reader. They now read the layout, and `test_no_positional_plan_slicing` fails
the moment one of those patterns comes back in the source.
"""
from __future__ import annotations

import io
import os
import re
import tokenize

import pytest
import torch

from f1sim import mpc

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: Source trees the guard reads. Tests are not in it: they build linear plans by position on
#: purpose, which is what a test of the linear layout is.
SOURCE_DIRS = ("f1sim/f1sim", "f1sim_ros/f1sim_ros", "f1sim/scripts", "scripts")

#: Code (comments and string literals removed) that addresses a plan action by position.
_LEAD = r"\[\s*(?:(?:\.\.\.|:)\s*,\s*)?"                     # `[`, `[:, ` or `[..., `
_WIDTH = r"(?:\w+\.)?(?:N_KNOTS|ACT_DIM|N_ACC|PLAN_DIM)\b"
FORBIDDEN = {
    "last two columns": re.compile(r"\[[^\[\]]*(?:-\s*2\s*:|:\s*-\s*2)[^\[\]]*\]"),
    "first six": re.compile(_LEAD + r":\s*6\s*\]"),
    "from six": re.compile(r"\[\s*(?:\.\.\.|:)\s*,\s*6\s*:\s*8?\s*\]"),
    "column 6 or 7": re.compile(r"\[\s*(?:\.\.\.|:)\s*,\s*[67]\s*\]"),
    "a width constant as an index": re.compile(
        r"\[[^\[\]]*(?:[:,]\s*" + _WIDTH + r"|" + _WIDTH + r"(?:\s*[-+]\s*\d+)?\s*:)[^\[\]]*\]"),
    "a width of eight": re.compile(r"(?:==|!=)\s*8\b(?!\.)|\bin\s*\(\s*2\s*,\s*8\s*\)"),
}

#: Hits that are not plans, by file and the exact code on the line. Exact, so an edit to one of these
#: lines has to come back here and say again why it is not a plan.
ALLOWED = {
    # the simulator's state vector (x, y, yaw, vx, vy, yaw_rate, steer, omega_r): column 6 is steer
    ("f1sim/f1sim/viewer/native.py",
     'return {"x": pack[:, 0], "y": pack[:, 1], "yaw": pack[:, 2], "vx": pack[:, 3], "steer": pack[:, 6],'),
    ("f1sim/f1sim/viewer/server.py",
     "cars = torch.stack([st[:, 0], st[:, 1], st[:, 2], st[:, 6], st[:, 3], r.lap[:n].float(),"),
    ("f1sim/f1sim/viewer/sim_worker.py", '"steer": _copy(p[:, 6]), "roll": _copy(p[:, 7]), "pitch": _copy(p[:, 8]),'),
    # the adaptive grip estimator's feature vector, column 6 of its own layout
    ("f1sim/f1sim/learn/adaptive_grip.py", "x[..., 6] - 0.981,"),
    # the first six names of a list, in an error message
    ("f1sim_ros/f1sim_ros/eval_node.py", '"\\n  ".join(sorted(cells)[:6]) + f"\\n  ... ({len(cells)} in all)")'),
    # an occupancy grid's border, and an encoder benchmark's own label table (column 6: visible)
    ("f1sim/scripts/replay_dynamics.py",
     "occ = np.zeros((n, n), bool); occ[:2] = occ[-2:] = True; occ[:, :2] = occ[:, -2:] = True"),
    ("f1sim/scripts/encoder_bench.py", "return (lab[:, 6] > 0.5) & ((car.cpu() & near).sum(1) >= VIS_BEAMS)"),
    ("f1sim/scripts/encoder_bench.py", "y[:, 6] = visible(tr, tt, ll, angles).float()"),
    ("f1sim/scripts/encoder_bench.py", "pres = y[:, 6]"),
}


def _code_lines(path: str):
    """(line number, code, line) for every line of a Python file: the code with comments and string
    literals blanked out -- f-strings included, which on this Python are single STRING tokens -- and
    the line as written, which is what ALLOWED names."""
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    blank = [list(line) for line in lines]
    for tok in tokens:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (r0, c0), (r1, c1) = tok.start, tok.end
        for r in range(r0, r1 + 1):
            if r - 1 >= len(blank):
                break
            row = blank[r - 1]
            a = c0 if r == r0 else 0
            b = c1 if r == r1 else len(row)
            for c in range(a, min(b, len(row))):
                row[c] = " "
    return [(i + 1, "".join(row), line) for i, (row, line) in enumerate(zip(blank, lines))]


def _scan():
    hits = []
    for d in SOURCE_DIRS:
        root = os.path.join(REPO, d)
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                rel = os.path.relpath(path, REPO)
                for n, code, line in _code_lines(path):
                    for name, rx in FORBIDDEN.items():
                        if rx.search(code):
                            hits.append((rel, n, name, line.strip()))
                            break
    return hits


def test_no_positional_plan_slicing():
    """Every plan column is addressed through `mpc.plan_layout`, never by position.

    A hit here is either a plan read by position -- use `plan_layout(spec).path` / `.speed` -- or
    something that is not a plan, which then belongs in ALLOWED with the reason."""
    hits = [h for h in _scan() if (h[0], " ".join(h[3].split())) not in
            {(f, " ".join(c.split())) for f, c in ALLOWED}]
    assert not hits, "positional plan slicing:\n" + "\n".join(f"  {f}:{n}  [{name}]  {code}"
                                                             for f, n, name, code in hits)


def test_the_allow_list_has_no_dead_entries():
    """An entry whose line is gone allows nothing and only hides the next mistake with its name."""
    found = {(f, " ".join(code.split())) for f, _, _, code in _scan()}
    dead = [e for e in ALLOWED if (e[0], " ".join(e[1].split())) not in found]
    assert not dead, f"ALLOWED entries that match nothing any more: {dead}"


def test_the_guard_sees_what_it_is_for():
    """The patterns catch the shapes that were in the tree, and not a comment or a string about them."""
    for code in ("an[:, -2:] = x", "a = torch.cat([a[..., :-2], b], 2)", "ref_mean[:, :6]", "param[:6]",
                 "e = act[:, N_KNOTS:ACT_DIM]", "action[:, _mpc.N_KNOTS + 1]", "if out_features != 8:",
                 "spec.act_dim == 8", "stop[:, 6:] = -1", "v = a[:, 7]", "out[..., :N_KNOTS] = 0",
                 "e = lab[..., 6:8]"):
        assert any(rx.search(code) for rx in FORBIDDEN.values()), code
    for code in ("x = a[:, lay.speed]", "speed_cap=8.0", "n = 8", "torch.zeros(B, 8)", "q[:, :3]",
                 "lp[:, 3:6]", "b[6:9]", "at = [min(j / (N_ACC - 1), 1) for j in range(N_ACC)]"):
        assert not any(rx.search(code) for rx in FORBIDDEN.values()), code
    src = 'x = 1  # an[:, -2:] is gone\ny = f"{sorted(s)[:6]}"\n'
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
    try:
        assert not any(rx.search(c) for _, c, _ in _code_lines(fh.name) for rx in FORBIDDEN.values())
    finally:
        os.unlink(fh.name)


# ------------------------------------------------------------------ the layouts
def test_linear_is_six_knots_then_two_speeds():
    lin = mpc.plan_layout("linear")
    assert (lin.act_dim, lin.path, lin.speed, lin.grip) == (8, slice(0, 6), slice(6, 8), None)
    assert lin.act_dim == mpc.ACT_DIM == mpc.PLAN_WIRE_DIM
    assert lin.other == slice(0, 6)


@pytest.mark.parametrize("mode", mpc.SPEED_MODES)
def test_every_mode_s_layout_partitions_its_action(mode):
    lay = mpc.plan_layout(mode)
    assert lay.act_dim == mpc.act_dim(mode)
    cols = list(range(lay.act_dim))
    speed = cols[lay.speed]
    other = cols[lay.other] if isinstance(lay.other, slice) else list(lay.other)
    assert sorted(speed + other) == cols and not set(speed) & set(other)
    assert set(cols[lay.path]) <= set(other)
    if lay.grip is not None:
        assert lay.grip in other or mode == "envelope"          # envelope slows down through its a_hat


def test_the_layout_is_read_from_a_mode_a_spec_or_a_contract_and_never_guessed():
    assert mpc.plan_layout(None) is mpc.plan_layout("linear")
    assert mpc.plan_layout(mpc.PlanSpec(speed_mode="budget")) is mpc.plan_layout("budget")
    assert mpc.plan_layout(mpc.contract_of({"experiment": {"speed_mode": "budget"}})) is mpc.plan_layout("budget")
    assert mpc.plan_layout(mpc.contract_of({})) is mpc.plan_layout("linear")
    with pytest.raises(ValueError, match="contract"):
        mpc.plan_layout({"experiment": {"speed_mode": "budget"}})  # a raw `extra` would read as linear
    with pytest.raises(ValueError, match="speed_mode"):
        mpc.plan_layout("wp6")


def test_the_layout_columns_are_the_ones_the_linear_code_always_used():
    """Bit-identity for every linear checkpoint rests on this: the same elements, in the same order."""
    a = torch.arange(16.0).view(2, 8)
    lay = mpc.plan_layout("linear")
    assert torch.equal(a[:, lay.speed], a[:, -2:]) and torch.equal(a[:, lay.path], a[:, :6])
    assert torch.equal(a[:, lay.other], a[:, :-2])
    b = a.clone(); b[:, lay.speed] = -1.0
    c = a.clone(); c[:, -2:] = -1.0
    assert torch.equal(b, c)
