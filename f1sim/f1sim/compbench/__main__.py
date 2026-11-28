"""python -m f1sim.compbench <command> ...

  run      --driver D --suite dev --condition real [--format qual|race] [--maps a,b] [--seed S]
           [--opponent ID] [--starts half_lap,side_by_side] [--force]
           one driver on the suite; raw records + logs under /data/compbench/raw/<suite tag>/, scored, table printed
  table    --suite dev --condition real [--format qual|race] [--drivers a,b,...] [--maps ...] [--out file.md]
           the competition table of every driver already run (bootstrap 95 % intervals)
  compare  --suite dev --condition real --a D1 --b D2 [--maps ...]
           paired (same-session) differences D2 - D1 with intervals and the 'same' verdict per metric
  race     --suite lab-dev --condition real    the 1v1 table with the catch-up columns (half a lap apart = representative)
  qualify  --suite lab-dev --maps lab12x16     the opponent qualification table (each kind alone; GT use; settings source)
  freeze   --suite lab-dev                     write the hash of every bank that exists into the suite file
  noise    --suite dev --condition real --driver D [--maps k26scan]
           seed-to-seed spread of one driver (the suite's noise seeds) and the detectable-difference table
  determinism  A.npz B.npz      bit-level comparison of two records of the same job
  suite    --suite dev          print the suite's tag, hash, maps and banks
  tune     --kind ftg ...       see tune.py
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m f1sim.compbench")
    ap.add_argument("command")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--driver"); ap.add_argument("--drivers"); ap.add_argument("--suite", default="dev")
    ap.add_argument("--condition", default="real"); ap.add_argument("--format", default="qual")
    ap.add_argument("--maps"); ap.add_argument("--seed", type=int); ap.add_argument("--opponent")
    ap.add_argument("--starts"); ap.add_argument("--force", action="store_true"); ap.add_argument("--out")
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--kind"); ap.add_argument("--seeds")
    ap.add_argument("--no-score", action="store_true"); ap.add_argument("--maps-b")
    a = ap.parse_args(argv)
    maps = a.maps.split(",") if a.maps else None
    if a.command == "run":
        from .runner import run_qual, run_race
        if a.format == "race":
            outs = run_race(a.driver, a.suite, a.condition, a.opponent, a.starts.split(",") if a.starts else None, a.seed,
                            a.force, score=not a.no_score)
        else:
            outs = run_qual(a.driver, a.suite, a.condition, maps, a.seed, a.force, score=not a.no_score)
        if not a.no_score and outs:
            from .report import table_for, race_catch_table
            from . import DATA
            from . import drivers as D
            md = table_for(a.suite, a.condition, a.format, drivers=[a.driver], maps=maps, seed=a.seed)
            if a.format == "race":
                md += "\n" + race_catch_table(a.suite, a.condition, drivers=[D.resolve(a.driver)["id"]], seed=a.seed)
            print(md)
            od = os.path.join(DATA, "results", a.suite, a.format, a.condition)
            os.makedirs(od, exist_ok=True)
            open(os.path.join(od, f"{D.resolve(a.driver)['id']}_s{a.seed or 'suite'}.md"), "w").write(md)
            json.dump([json.load(open(o[:-4] + ".score.json")) for o in outs],
                      open(os.path.join(od, f"{D.resolve(a.driver)['id']}_s{a.seed or 'suite'}.json"), "w"), default=float)
    elif a.command == "table":
        from .report import table_for
        md = table_for(a.suite, a.condition, a.format, drivers=a.drivers.split(",") if a.drivers else None, maps=maps,
                       seed=a.seed)
        print(md)
        if a.out:
            open(a.out, "w").write(md)
    elif a.command == "race":
        from .report import race_catch_table
        print(race_catch_table(a.suite, a.condition, drivers=a.drivers.split(",") if a.drivers else None, seed=a.seed))
    elif a.command == "qualify":
        from .report import qualification
        print(qualification(a.suite, maps or ["lab12x16"]))
    elif a.command == "compare":
        from .report import compare
        print(compare(a.suite, a.condition, a.a, a.b, maps=maps, fmt=a.format, seed=a.seed,
                      maps_b=a.maps_b.split(",") if a.maps_b else None))
    elif a.command == "noise":
        from .report import noise
        print(noise(a.suite, a.condition, a.driver, maps=maps or ["k26scan"],
                    seeds=[int(x) for x in a.seeds.split(",")] if a.seeds else None))
    elif a.command == "resolution":
        from .report import mde_pairs
        print(mde_pairs(a.suite, a.condition, a.a, a.b, maps=maps, seed=a.seed))
    elif a.command == "determinism":
        from .score import determinism
        print(json.dumps(determinism(a.rest[0], a.rest[1]), indent=1))
    elif a.command == "suite":
        from .suite import load_suite
        s = load_suite(a.suite)
        print(s["_tag"], s["_sha256"])
        for m in s["maps"]:
            print(f"  {m['id']:14s} {m['track']:42s} sessions {m['sessions']:4d} bank {m['bank']} {str(m.get('bank_sha256'))[:12]}")
        for c, pr in s["_profiles"].items():
            print(f"  condition {c}: mode {s['conditions'][c]['mode']}, profile {pr['name']} {pr['_sha256'][:12]} ranges {pr['ranges']}")
    elif a.command == "freeze":
        # fill the bank hash of every map whose bank exists and is not frozen yet (a frozen hash is never overwritten:
        # a changed bank is refused by every run instead)
        from .suite import load_suite, resolve_path
        from .layouts import bank_sha256
        s = load_suite(a.suite)
        raw = json.load(open(s["_path"]))
        for m in raw["maps"]:
            d = resolve_path(m["bank"])
            if m.get("bank_sha256") is None and os.path.isdir(d) and any(f.endswith(".pt") for f in os.listdir(d)):
                m["bank_sha256"] = bank_sha256(d)
                print(f"froze {m['id']}: {m['bank_sha256']}")
        json.dump(raw, open(s["_path"], "w"), indent=1, ensure_ascii=False)
    elif a.command == "tune":
        from .tune import main as tmain
        tmain(a.rest)
    else:
        ap.error(f"unknown command {a.command}")


if __name__ == "__main__":
    main()
