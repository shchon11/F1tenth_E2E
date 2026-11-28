"""단계별 표: 같은 세션 64개(같은 상자 배치·같은 물리 추첨)에서 무접촉 세션 비율, 세게 닿음/세션(벽·상자), 최고 랩, 완주 실패,
그리고 R0·R4와 짝지은 차이. 채점은 감사 C의 comp_metrics.score 그대로(120 s, 60 s에 상자 치움).
구간: 세션 단위 부트스트랩 95 % (2000번; 짝지은 차이는 같은 세션을 함께 뽑음).
인자: 출력json 라벨=기록이름 ...  (기록 폴더 /data/pm-execgap-2026-11-28/rec)"""
import sys, json, os
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/pm/audit-2026-11-28/C_scripts")
import comp_metrics as CM

O = os.environ.get("XG_REC", "/data/pm-execgap-2026-11-28/rec")
RNG = np.random.default_rng(0)


def per_session(fn):
    S = CM.score(fn)["sessions"]
    g = lambda k: np.array([s[k] for s in S], float)
    clean = (g("wall_hard") + g("box_hard")) == 0
    return {"clean": clean.astype(float), "wall_hard": g("wall_hard"), "box_hard": g("box_hard"), "best": g("best"),
            "median": g("median"), "dnf": g("dnf"), "dist": g("dist"), "streak": g("streak_hard"), "stuck": g("stuck_s"),
            "wall_any": g("wall_any"), "box_any": g("box_any")}


def boot(x, stat=np.nanmean, n=2000):
    idx = RNG.integers(0, len(x), (n, len(x)))
    bs = np.array([stat(x[i]) for i in idx])
    return [float(stat(x)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


if __name__ == "__main__":
    outj = sys.argv[1]
    items = [a.split("=", 1) for a in sys.argv[2:]]
    D = {}
    for lab, nm in items:
        fn = f"{O}/{nm}.npz"
        if os.path.exists(fn):
            D[lab] = per_session(fn)
    res = {}
    labs = list(D)
    base0 = labs[0]
    base4 = next((l for l in labs if l.startswith("R4")), None)
    for lab in labs:
        d = D[lab]
        r = {"n": int(len(d["clean"])), "clean": boot(d["clean"]), "wall_hard": boot(d["wall_hard"]), "box_hard": boot(d["box_hard"]),
             "best": boot(d["best"], np.nanmedian), "median_lap": boot(d["median"], np.nanmedian), "dnf": boot(d["dnf"]),
             "dist": boot(d["dist"]), "streak": boot(d["streak"]), "wall_any": boot(d["wall_any"]), "box_any": boot(d["box_any"])}
        for bl in (base0, base4):
            if bl is None or bl == lab or len(D[bl]["clean"]) != len(d["clean"]):
                continue
            for k in ("clean", "wall_hard", "box_hard", "dist"):
                r[f"d_{k}_vs_{bl}"] = boot(d[k] - D[bl][k])
            r[f"d_best_vs_{bl}"] = boot(d["best"] - D[bl]["best"], np.nanmedian)
        res[lab] = r
    json.dump(res, open(outj, "w"), ensure_ascii=False)
    f = lambda v, p=2, s=1.0: f"{v[0] * s:.{p}f} [{v[1] * s:.{p}f}–{v[2] * s:.{p}f}]"
    print("| 단계 | 세션 | 무접촉(세게) 세션 | 세게 닿음/세션 벽 | 상자 | 최고 랩 s (세션 중앙) | 완주 실패 | 120 s 거리 m |")
    print("|---|---|---|---|---|---|---|---|")
    for lab in labs:
        r = res[lab]
        print(f"| {lab} | {r['n']} | {f(r['clean'], 0, 100)} % | {f(r['wall_hard'])} | {f(r['box_hard'])} | {f(r['best'])} | "
              f"{f(r['dnf'], 0, 100)} % | {f(r['dist'], 0)} |")
    print()
    for lab in labs:
        r = res[lab]
        ks = [k for k in r if k.startswith("d_")]
        if ks:
            print(lab, " ".join(f"{k}={f(r[k], 2 if 'dist' not in k else 0, 100 if 'clean' in k else 1)}" for k in ks))
