"""모든 결과를 한 파일로: 단계표, 몫, 기록 분석(사건 목록 제외), 같은 plan 비교 요약 -> summary.json
인자: 단계표json 몫json 분석json 같은plan명목npz 같은plan무작위npz 출력json"""
import sys, json
import numpy as np

tbl, sh, an, s1, s2, out = sys.argv[1:7]
A = json.load(open(an))
S = {"stages": json.load(open(tbl)), "shares": json.load(open(sh)), "bandwidth": A.get("_bandwidth")}
S["records"] = {}
for nm, r in A.items():
    if nm.startswith("_"):
        continue
    S["records"][nm] = {k: v for k, v in r.items() if k not in ("events",) and not (k == "B" and False)}
    S["records"][nm]["B"] = {k: v for k, v in r["B"].items() if not k.startswith("psd") and k != "f"}
for lab, fn in (("same_plan_nominal", s1), ("same_plan_random_cars", s2)):
    Z = np.load(fn)
    names = [k[5:] for k in Z.files if k.startswith("trj__") and k != "trj__plan"]
    d = {}
    for nm in names:
        lat = np.abs(Z[f"m__{nm}__lat"]); pos = Z[f"m__{nm}__pos"]
        d[nm] = {"lat_cm_p50_0.3_0.5_1.0": [float(np.median(lat[:, j]) * 100) for j in (2, 4, 9)],
                 "lat_cm_p90_0.5": float(np.percentile(lat[:, 4], 90) * 100), "pos_cm_p50_0.5": float(np.median(pos[:, 4]) * 100)}
    P = {nm: Z[f"trj__{nm}"] for nm in names}
    d["pairwise_pos_cm_p50_0.5s"] = {a: {b: float(np.median(np.hypot(P[a][:, 20, 0] - P[b][:, 20, 0], P[a][:, 20, 1] - P[b][:, 20, 1])) * 100)
                                         for b in names} for a in names}
    S[lab] = d
json.dump(S, open(out, "w"), ensure_ascii=False, indent=1, default=float)
print("saved", out)
