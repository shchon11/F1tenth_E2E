"""Suite and profile files: loading, resolving paths, the content hash every result carries."""
import os
import json
import hashlib

from . import PKG_DIR, REPO, DATA


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def load_profile(name: str) -> dict:
    p = name if name.endswith(".json") else os.path.join(PKG_DIR, "profiles", f"{name}.json")
    d = json.load(open(p))
    d["_sha256"] = hashlib.sha256(_canon({k: v for k, v in d.items() if not k.startswith("_")})).hexdigest()
    d["_path"] = os.path.abspath(p)
    return d


def resolve_path(p: str) -> str:
    """suite paths: '~' expanded; 'banks/..', 'tuned/..' under DATA; other relative paths under the repo"""
    if p is None:
        return None
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return p
    if p.split("/")[0] in ("banks", "tuned", "raw", "results"):
        return os.path.join(DATA, p)
    return os.path.join(REPO, p)


def load_suite(name: str) -> dict:
    p = name if name.endswith(".json") else os.path.join(PKG_DIR, "suites", f"{name}.json")
    s = json.load(open(p))
    profiles = {c: load_profile(v["profile"]) for c, v in s["conditions"].items()}
    body = {k: v for k, v in s.items() if not k.startswith("_")}
    h = hashlib.sha256(_canon(body) + b"".join(_canon({k: v for k, v in pr.items() if not k.startswith("_")})
                                               for _, pr in sorted(profiles.items()))).hexdigest()
    s["_sha256"] = h
    s["_tag"] = f"{s['name']}-{s['version']}+{h[:10]}"
    s["_path"] = os.path.abspath(p)
    s["_profiles"] = profiles
    return s


def check_bank(m: dict, allow_unfrozen: bool = False) -> str:
    """the map's bank directory, refused when its content is not the one the suite froze"""
    from .layouts import bank_sha256
    d = resolve_path(m["bank"])
    if not os.path.isdir(d):
        raise FileNotFoundError(f"suite map {m['id']}: bank {d} missing (build it: see the suite's 'role')")
    got = bank_sha256(d)
    want = m.get("bank_sha256")
    if want is None:
        if not allow_unfrozen:
            raise ValueError(f"suite map {m['id']}: bank sha256 not frozen in the suite file (got {got}); freeze it first")
    elif got != want:
        raise ValueError(f"suite map {m['id']}: bank {d} changed (sha256 {got}, suite says {want})")
    return d


def cell_sha256(s: dict, condition: str, m: dict, seed: int, fmt: str = "qual") -> str:
    """the hash of everything that defines one cell's sessions (map entry, format block, condition, profile, seed, world):
    a record is comparable with another of the same cell exactly when the two carry the same cell hash"""
    prof = {k: v for k, v in s["_profiles"][condition].items() if not k.startswith("_")}
    cond = dict(s["conditions"][condition])
    cond.pop("note", None)
    mm = {k: v for k, v in m.items() if k != "role"}
    blk = {k: v for k, v in s[fmt].items() if k not in ("opponents", "rule", "start", "starts")} if fmt in s else {}
    return hashlib.sha256(_canon(dict(map=mm, fmt=fmt, block=blk, condition=cond, profile=prof, seed=int(seed),
                                      world=s["world"]))).hexdigest()
