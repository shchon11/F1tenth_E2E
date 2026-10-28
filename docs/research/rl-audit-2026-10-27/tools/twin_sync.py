"""Make one env an exact copy of its twin (rl-audit 2026-10-28, for paired_branch.py).

Two envs built by the same `make_env` call have the same object structure, so walking them side by side pairs every
piece of state: tensors are copied in place (the simulator rebinds some every step and CUDA graphs hold pointers to
others, so a copy must land in the tensor the twin will read next), tensors of another shape are rebound to a clone,
torch / numpy / Python generators get the source's state, and the numbers of f1sim objects (`sim.t`, the IMU phase,
counters) are assigned. Functions, modules' weights and strings are left alone: the twin keeps its own graphs and
code. `diff` walks the same way and lists the paths whose tensors differ -- the check that nothing was missed."""
from __future__ import annotations

import random

import numpy as np
import torch

_NUM = (bool, int, float)


def _is_f1sim(o) -> bool:
    return type(o).__module__.startswith("f1sim") and hasattr(o, "__dict__")


_MISSING = object()


def _children(a, b):
    """(key, kind, value in a, value in b) for every slot of a (b's value is _MISSING where b has none); kind 'a' =
    attribute, 'k' = item."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [(k, "k", a[k], b.get(k, _MISSING)) for k in list(a.keys())]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        return [(i, "k", x, y) for i, (x, y) in enumerate(zip(a, b))]
    if isinstance(a, torch.nn.Module) and isinstance(b, torch.nn.Module):
        ba, bb = dict(a.named_buffers()), dict(b.named_buffers())
        out = [(k, "buf", ba[k], bb[k]) for k in ba if k in bb]
        skip = {"_parameters", "_buffers", "_modules"}
        out += [(k, "a", v, vars(b)[k]) for k, v in vars(a).items() if k not in skip and k in vars(b)]
        return out
    if _is_f1sim(a) and _is_f1sim(b) and type(a) is type(b):
        vb = vars(b)
        return [(k, "a", v, vb.get(k, _MISSING)) for k, v in list(vars(a).items())]
    return []


def _fresh(x):
    """A copy of a value b lacks (lazily created state): tensors cloned, containers of tensors rebuilt, else None."""
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, dict) and all(torch.is_tensor(v) for v in x.values()):
        return {k: v.clone() for k, v in x.items()}
    if isinstance(x, torch.Generator):
        g = torch.Generator(device=x.device); g.set_state(x.get_state()); return g
    return None


def _assign(parent, key, kind, value) -> bool:
    if kind == "a":
        setattr(parent, key, value); return True
    if kind == "k" and isinstance(parent, (dict, list)):
        parent[key] = value; return True
    return False


def sync(src, dst, max_depth: int = 10) -> dict:
    """Copy every piece of `src`'s state into `dst` (its twin). Returns counts per kind."""
    n = {"copied": 0, "rebound": 0, "generators": 0, "numbers": 0, "arrays": 0, "unassignable": 0}
    seen = set()

    def rec(a, b, pb, depth):
        if a is b or depth > max_depth or id(a) in seen:
            return
        seen.add(id(a))
        for key, kind, x, y in _children(a, b):
            if x is y:
                continue
            if y is _MISSING or (y is None and x is not None):
                f = _fresh(x)
                if f is not None and _assign(b, key, kind, f):
                    n["created"] = n.get("created", 0) + 1
                elif x is not None:
                    n.setdefault("absent", []).append(f"{key}")
                continue
            if torch.is_tensor(x):
                if torch.is_tensor(y) and y.shape == x.shape and y.dtype == x.dtype and y.device == x.device:
                    if y.data_ptr() != x.data_ptr():
                        y.copy_(x); n["copied"] += 1
                elif kind != "buf" and _assign(b, key, kind, x.clone()):
                    n["rebound"] += 1
                else:
                    n["unassignable"] += 1
            elif isinstance(x, torch.Generator):
                if isinstance(y, torch.Generator):
                    y.set_state(x.get_state())
                else:
                    g = torch.Generator(device=x.device); g.set_state(x.get_state()); _assign(b, key, kind, g)
                n["generators"] += 1
            elif isinstance(x, np.random.Generator) and isinstance(y, np.random.Generator):
                y.bit_generator.state = x.bit_generator.state; n["generators"] += 1
            elif isinstance(x, np.random.RandomState) and isinstance(y, np.random.RandomState):
                y.set_state(x.get_state()); n["generators"] += 1
            elif isinstance(x, random.Random) and isinstance(y, random.Random):
                y.setstate(x.getstate()); n["generators"] += 1
            elif isinstance(x, np.ndarray):
                if isinstance(y, np.ndarray) and y.shape == x.shape and y.dtype == x.dtype and y.flags.writeable:
                    np.copyto(y, x)
                else:
                    _assign(b, key, kind, x.copy())
                n["arrays"] += 1
            elif isinstance(x, _NUM) and kind == "a":
                if type(x) is type(y) and x != y:
                    setattr(b, key, x); n["numbers"] += 1
            else:
                rec(x, y, b, depth + 1)

    rec(src, dst, None, 0)
    return n


def diff(a0, b0, max_depth: int = 10, tol: float = 0.0, limit: int = 40) -> list:
    """Paths whose tensors (or numbers) differ between two twins, with the largest difference."""
    out, seen = [], set()

    def rec(a, b, path, depth):
        if a is b or depth > max_depth or id(a) in seen or len(out) >= limit:
            return
        seen.add(id(a))
        for key, kind, x, y in _children(a, b):
            p = f"{path}.{key}"
            if y is _MISSING:
                out.append((p, "missing", None, None)); continue
            if torch.is_tensor(x) and torch.is_tensor(y):
                if x.shape != y.shape:
                    out.append((p, "shape", tuple(x.shape), tuple(y.shape)))
                elif x.numel() and x.data_ptr() != y.data_ptr():
                    d = (x.double() - y.double()).abs().max().item() if x.dtype != torch.bool else float((x != y).any())
                    if d > tol:
                        out.append((p, "value", d, None))
            elif isinstance(x, _NUM) and isinstance(y, _NUM) and kind == "a":
                if x != y:
                    out.append((p, "number", x, y))
            elif isinstance(x, torch.Generator) and isinstance(y, torch.Generator):
                if not torch.equal(x.get_state(), y.get_state()):
                    out.append((p, "generator", None, None))
            else:
                rec(x, y, p, depth + 1)

    rec(a0, b0, "env", 0)
    return out


def reach(root, max_depth: int = 14) -> dict:
    """Every tensor reachable from `root` through ANY object (attributes, containers, closures, bound methods), by id ->
    the first path found -- to see what `sync`'s f1sim-only walk misses."""
    import collections
    import types
    out, seen = {}, set()

    def rec(o, path, depth):
        if depth > max_depth or id(o) in seen:
            return
        seen.add(id(o))
        if torch.is_tensor(o):
            out.setdefault(id(o), (path, tuple(o.shape), str(o.dtype)))
            return
        if isinstance(o, (str, bytes, int, float, bool, type(None), type, types.ModuleType)):
            return
        if isinstance(o, dict):
            for k, v in list(o.items()):
                rec(v, f"{path}[{k!r}]", depth + 1)
        elif isinstance(o, (list, tuple, set, frozenset, collections.deque)):
            for i, v in enumerate(list(o)[:4096]):
                rec(v, f"{path}[{i}]", depth + 1)
        elif isinstance(o, types.MethodType):
            rec(o.__self__, f"{path}.__self__", depth + 1)
        elif isinstance(o, types.FunctionType):
            for i, c in enumerate(o.__closure__ or ()):
                try:
                    rec(c.cell_contents, f"{path}.<closure {o.__code__.co_freevars[i]}>", depth + 1)
                except ValueError:
                    pass
        elif isinstance(o, torch.nn.Module):
            for n_, t in list(o.named_buffers()):
                rec(t, f"{path}.{n_}", depth + 1)
            for k, v in vars(o).items():
                if k not in ("_parameters", "_buffers", "_modules"):
                    rec(v, f"{path}.{k}", depth + 1)
            for n_, m in o.named_children():
                rec(m, f"{path}.{n_}", depth + 1)
        elif hasattr(o, "__dict__"):
            for k, v in list(vars(o).items()):
                rec(v, f"{path}.{k}", depth + 1)

    rec(root, "env", 0)
    return out


def covered(src, dst, max_depth: int = 10) -> set:
    """ids of the tensors `sync` visits in `src`."""
    ids, seen = set(), set()

    def rec(a, b, depth):
        if a is b or depth > max_depth or id(a) in seen:
            return
        seen.add(id(a))
        for key, kind, x, y in _children(a, b):
            if y is _MISSING:
                continue
            if torch.is_tensor(x):
                ids.add(id(x))
            elif not isinstance(x, (torch.Generator, np.ndarray) + _NUM):
                rec(x, y, depth + 1)

    rec(src, dst, 0)
    return ids
