"""그림: 세게 닿은 상자 사건마다 처음 보인 때(닿기 몇 s 전, 몇 m)와 회피 가능성 분류, 그리고 상자 자리별 첫 만남에서 처음 보인 거리 분포.
인자: events.json memory_test.json out.png"""
import sys, json
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import plt
E = json.load(open(sys.argv[1])); M = json.load(open(sys.argv[2]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cols = {"early": "#2ca02c", "imm": "#ff7f0e", "no": "#d62728"}
done = set()
for lab, r in E.items():
    for h in r["hard"]:
        if h["t"] == 0 or h["rehit_in_enc"]:
            continue
        f0, f3 = h["feas"]["d0.0_brake"][0], h["feas"]["d0.3_brake"][0]
        k = "early" if f3 else ("imm" if f0 else "no")
        nm = {"early": "0.3 s 늦게 반응해도 피할 수 있었음", "imm": "0.3 s 안에 반응해야 피할 수 있었음", "no": "물리적으로 못 피함"}[k]
        ax[0].scatter(h["vis_s"], h["d_vis"], c=cols[k], s=18 + 4 * h["v_vis"], alpha=0.75, label=nm if k not in done else None, edgecolors="k", linewidths=0.3)
        done.add(k)
ax[0].set_xlabel("처음 보인 때부터 닿을 때까지 [s]"); ax[0].set_ylabel("처음 보인 때 상자까지 거리 [m]")
ax[0].set_title("세게 닿은 상자 56건 (출발 결함 7건 뺌, 점 크기 = 속도)", fontsize=9); ax[0].legend(fontsize=8, loc="upper left")
d = []
for lab, rows in M.items():
    seen = set()
    for x in rows:
        if (x["b"], x["j"]) not in seen:
            seen.add((x["b"], x["j"])); d.append(x["dvis"])
ax[1].hist(d, bins=np.arange(0, 10.5, 0.5), color="#1f77b4")
ax[1].set_xlabel("첫 만남에서 처음 보인 거리 [m] (LiDAR 최대 10 m)"); ax[1].set_ylabel("상자 수 (세 기록)")
ax[1].set_title(f"상자 {len(d)}개: 5 m 안에서야 보이는 상자 {100 * np.mean(np.array(d) < 5):.0f} %", fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=110); print("saved", sys.argv[3])
