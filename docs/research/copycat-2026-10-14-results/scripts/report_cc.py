"""The copycat study's table: every arm's M1-M5 and the PREREG verdict (docs/research/copycat-2026-10-14.md).
usage: python report_cc.py TEACHER.json ARM_NAME[=label] ... > report.txt ; also writes report.json"""
import json
import os
import sys

D = "/tmp/claude-1000/-home-shchon11-F1tenth-E2E/d98b5699-f98e-4674-9a12-ab5a57bf35f1/scratchpad/v2cc"
RUNS = "/home/shchon11/f1sim_runs"
teacher_path, arms = sys.argv[1], sys.argv[2:]
tj = json.load(open(teacher_path))
t = next(iter(tj.values()))
t_cpk = float(t["collisions_per_km"])
T = 3.0 * t_cpk
out = {"teacher": {k: t.get(k) for k in ("completions", "collisions", "completion_rate", "collisions_per_km",
                                        "distance_m", "mean_speed", "lap_time_s")}, "threshold_cpk": T, "arms": {}}
print(f"teacher ICCAS solo: {t['completions']}/128 completions, {t_cpk:.3f} coll/km over {t['distance_m'] / 1000:.2f} km "
      f"-> T = 3 x = {T:.3f} coll/km")
for spec in arms:
    name, label = (spec.split("=", 1) + [spec])[:2]
    a = {"name": name}
    ic = json.load(open(os.path.join(D, f"iccas_{name}.json")))
    rows = []
    for ck, r in sorted(ic.items()):
        it = int(os.path.basename(ck).split("_it")[1].split(".")[0])
        rows.append({"iter": it, "completions": int(r["completions"]), "collisions": int(r["collisions"]),
                     "cpk": float(r["collisions_per_km"]), "km": float(r["distance_m"]) / 1000,
                     "mean_speed": float(r["mean_speed"]), "lap_s": r.get("lap_time_s")})
    rows.sort(key=lambda x: x["iter"])
    a["iccas"] = rows
    c = [r["completions"] for r in rows]
    crit = {"a_nondecreasing": all(c[i] <= c[i + 1] for i in range(len(c) - 1)),
            "b_it2_ge_103": c[-1] >= 103, "c_it2_cpk_le_T": rows[-1]["cpk"] <= T}
    a["criteria"] = crit
    a["pass"] = all(crit.values())
    # M4 race evaluation and M5 timing from the run's own records
    prog = [json.loads(x) for x in open(os.path.join(RUNS, name, "progress.jsonl"))]
    a["race"] = [{"iter": p["iter"], "beta": p["beta"], "student_cpk": p.get("student_coll_per_km"),
                  "student_prog": p.get("student_prog_mps"), "teacher_cpk": p["teacher_coll_per_km"],
                  "teacher_prog": p["teacher_prog_mps"], "collect_s": p["collect_s"], "train_s": p["train_s"],
                  "eval_s": p["eval_s"], "teacher_eval_s": p["teacher_eval_s"], "loss": p["loss"],
                  "bc_lat": p.get("train/bc_lat"), "bc_speed": p.get("train/bc_speed"),
                  "dropped": p.get("train/action_dropped_frac"),
                  "masked": p.get("collect/bc_masked_frac"), "capped": p.get("collect/speed_capped_frac")}
                 for p in prog]
    H = [json.loads(x) for x in open(os.path.join(D, name, "harness.jsonl"))]
    a["collect"] = [{"iter": h["iter"], "wall_s": h["wall_s"], "label_s_per_step": h["acc"]["label_s"] / h["acc"]["label_calls"],
                     "step_s_per_step": h["acc"]["step_s"] / h["acc"]["step_calls"]}
                    for h in H if h["kind"] == "collect_end"]
    ends = [h for h in H if h["kind"] == "end"]
    a["total_s"] = ends[-1]["total_s"] if ends else None
    # M2 held-out BC
    hp = os.path.join(D, f"heldout_{name}.json")
    if os.path.exists(hp):
        hj = json.load(open(hp))
        a["heldout"] = [{"buffer": s["buffer"], "iter": s["iter"], "mae_v": s.get("mae_v_mps_all"),
                         "mae_lat": s.get("mae_lat_m_all"), "bias_v": s.get("bias_v_mps_all"),
                         "mae_v_car": s.get("mae_v_mps_car_rows"), "n": s.get("n_all")} for s in hj["scores"]]
    # M3 copycat probe
    a["probe"] = {}
    for who in ("last_student_drives", "teacher_drives"):
        cp = os.path.join(D, f"copycat_{name}_{who}.json")
        if os.path.exists(cp):
            cj = json.load(open(cp))
            a["probe"][who] = [{"iter": v["contract"]["iter"], **v["gains"],
                                "recorded_mae_v": v["recorded"]["mae_v"], "recorded_bias_v": v["recorded"]["bias_v"],
                                **({"zero_actions_mae_v": v["zero_actions"]["mae_v"],
                                    "teacher_history_mae_v": v["teacher_history"]["mae_v"]} if "zero_actions" in v else {})}
                               for v in cj.values()]
    out["arms"][label] = a
    print(f"\n== {label} ({name}): {'PASS' if a['pass'] else 'FAIL'} {crit}")
    for r in rows:
        print(f"  it{r['iter']}: ICCAS {r['completions']:3d}/128  {r['cpk']:7.3f} coll/km ({r['collisions']} over "
              f"{r['km']:.2f} km)  speed {r['mean_speed']:.2f} m/s  lap {r['lap_s']}")
    for p in a["race"]:
        print(f"  race it{p['iter']} beta {p['beta']:.2f}: student {p['student_cpk']} coll/km prog {p['student_prog']} | "
              f"teacher {p['teacher_cpk']:.2f} | bc_lat {p['bc_lat']:.4f} bc_speed {p['bc_speed']:.4f} | dropped {p['dropped']} "
              f"| masked {p['masked']:.3f} capped {p['capped']:.3f} | collect {p['collect_s']:.0f} train {p['train_s']:.0f} "
              f"eval {p['eval_s']:.0f} teacher-eval {p['teacher_eval_s']:.0f} s")
    for cl in a["collect"]:
        print(f"  collect it{cl['iter']}: {cl['wall_s']:.0f} s, labels {cl['label_s_per_step']:.4f} s/step, env.step "
              f"{cl['step_s_per_step']:.3f} s/step")
    for h in a.get("heldout", []):
        print(f"  held-out {h['buffer']:20s} it{h['iter']}: speed MAE {h['mae_v']:.3f} m/s (bias {h['bias_v']:+.3f}), "
              f"lateral MAE {h['mae_lat']:.3f} m, car rows speed MAE {h['mae_v_car']}")
    for who, ps in a["probe"].items():
        for p in ps:
            print(f"  probe {who:20s} it{p['iter']}: action-speed gain {p.get('act_speed_hist', float('nan')):.3f}, "
                  f"measured-speed gain {p['meas_speed_hist']:.3f}, recorded speed MAE {p['recorded_mae_v']:.3f} "
                  f"(zero actions {p.get('zero_actions_mae_v', float('nan')):.3f}, teacher history "
                  f"{p.get('teacher_history_mae_v', float('nan')):.3f})")
passing = [(k, v) for k, v in out["arms"].items() if v["pass"]]


def held_v(v):
    h = [x for x in v.get("heldout", []) if x["buffer"] == "last_student_drives" and x["iter"] == 2]
    return h[0]["mae_v"] if h else float("inf")


passing.sort(key=lambda kv: (-kv[1]["iccas"][-1]["completions"], kv[1]["iccas"][-1]["cpk"], held_v(kv[1])))
out["verdict"] = passing[0][0] if passing else None
print("\nverdict:", out["verdict"] or "no arm passes")
json.dump(out, open(os.path.join(D, "report.json"), "w"), indent=1)
