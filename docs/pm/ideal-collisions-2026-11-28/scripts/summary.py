"""사건 단위 최종 표: events.json(가시성·회피 가능성·계획), reaction.json(상자 1 m 앞 옆 위치로 정한 지나는 쪽), plan_side.json(계획의 쪽 바꿈)을
합쳐 세게 닿은 상자 사건마다 한 줄, 그리고 분류 비율(배치 단위 부트스트랩 95 %). 출발 순간(0.0 s) 사건은 따로 뺌.
인자: events.json reaction.json plan_side.json out.md"""
import sys, json
import numpy as np
RNG = np.random.default_rng(0)
E = json.load(open(sys.argv[1])); RE = json.load(open(sys.argv[2])); PSd = json.load(open(sys.argv[3]))
NM = {"scan31b": "scan31b u475", "hs43b": "hs43b u900", "scan31b_HO": "scan31b u475·새 변형"}


def boot(flags, cl, n=4000):
    f = np.asarray(flags, float); cl = np.asarray(cl); u = np.unique(cl); idx = {c: np.nonzero(cl == c)[0] for c in u}
    o = [f[np.concatenate([idx[c] for c in RNG.choice(u, len(u))])].mean() for _ in range(n)]
    return f.mean(), np.percentile(o, 2.5), np.percentile(o, 97.5)


rows = []
for lab, r in E.items():
    rx = {}
    for o in RE[lab]:
        rx.setdefault((o["b"], o["j"]), []).append(o)
    ps = {}
    for o in PSd[lab]:
        if o["hit"]:
            ps.setdefault((o["b"], o["j"]), []).append(o)
    for h in r["hard"]:
        b, j, t = h["b"], h["j"], h["t"]
        f0, f3 = h["feas"]["d0.0_brake"][0], h["feas"]["d0.3_brake"][0]
        if t == 0:
            vis = "출발 배치 결함(0.0 s에 이미 닿음)"
        elif h["rehit_in_enc"]:
            vis = "끼인 채 다시 닿음"
        elif not f0:
            vis = "늦게 보임: 물리적으로 불가"
        elif not f3:
            vis = "늦게 보임: 0.3 s 안에 반응해야 가능"
        else:
            vis = "일찍 보임: 0.3 s 늦게 반응해도 가능"
        # robust side: car centre vs box centre 1 m before the box, this pass and the previous one
        seq = sorted(rx.get((b, j), []), key=lambda o: o["enc"])
        me = next((o for o in seq if o["hit"] and abs(o["t_end"] - t) <= 2), None)
        prev = [o for o in seq if o["t_end"] < t - 5]
        rel = lambda o: (o["dev_m1"] - o["box_lat"]) if (o is not None and np.isfinite(o["dev_m1"])) else np.nan
        r_me = rel(me)
        if t == 0:
            hist = "-"
        elif not prev:
            hist = "첫 만남"
        elif prev[-1]["hit"] or prev[-1]["any"]:
            hist = "앞 바퀴에도 닿음"
        else:
            r_p = rel(prev[-1])
            if np.isfinite(r_me) and abs(r_me) < 0.10:
                hist = "앞 바퀴와 다른 길: 정면으로"
            elif np.isfinite(r_me) and np.isfinite(r_p) and np.sign(r_me) != np.sign(r_p):
                hist = "앞 바퀴와 다른 길: 반대쪽으로"
            elif prev[-1]["clear"] < 0.10:
                hist = "앞 바퀴와 같은 쪽, 앞 바퀴 여유 <10 cm"
            else:
                hist = "앞 바퀴와 같은 쪽, 앞 바퀴 여유 >=10 cm"
        sw = [o for o in ps.get((b, j), [])]
        nsw = sw[0]["nsw"] if sw else -1
        rows.append(dict(lab=lab, b=b, t=t, cl=f"{lab}:{b % 64}", vis=vis, hist=hist, plan_through=h["plan_clear_last"] <= 0,
                         vis_s=h["vis_s"], d_vis=h["d_vis"], v_vis=h["v_vis"], v_hit=h["v_hit"], enc=h["enc_no"] + 1,
                         prev_clear=prev[-1]["clear"] if prev else np.nan, rel=r_me, nsw=nsw, room=h["wall_free_max"],
                         room_min=h["wall_free"]))
drive = [x for x in rows if x["t"] > 0]
out = []
out.append(f"세게 닿은 상자 사건 {len(rows)}건 (출발 배치 결함 {len(rows) - len(drive)}건 포함). 아래 비율은 주행 중 사건 {len(drive)}건 기준, [ ] = 세션의 상자 배치 단위 부트스트랩 95 %.\n")
for key, title in (("vis", "보였을 때 피할 수 있었나"), ("hist", "같은 상자의 앞 바퀴")):
    out.append(f"\n| {title} | 건수 | 비율 |\n|---|---|---|")
    cats = sorted(set(x[key] for x in drive), key=lambda c: -sum(x[key] == c for x in drive))
    for c in cats:
        m, lo, hi = boot([x[key] == c for x in drive], [x["cl"] for x in drive])
        out.append(f"| {c} | {sum(x[key] == c for x in drive)} | {100 * m:.0f} % [{100 * lo:.0f}-{100 * hi:.0f}] |")
m, lo, hi = boot([x["plan_through"] for x in drive], [x["cl"] for x in drive])
out.append(f"\n닿기 0.1 s 전 계획이 상자를 지나감: {sum(x['plan_through'] for x in drive)}/{len(drive)} = {100 * m:.0f} % [{100 * lo:.0f}-{100 * hi:.0f}]")
k = [x for x in drive if x["nsw"] >= 0]
m, lo, hi = boot([x["nsw"] > 0 for x in k], [x["cl"] for x in k])
out.append(f"\n마지막 1.5 s에 계획이 상자를 지나는 쪽을 한 번 이상 바꿈: {sum(x['nsw'] > 0 for x in k)}/{len(k)} = {100 * m:.0f} % [{100 * lo:.0f}-{100 * hi:.0f}]")
rr = np.array([abs(x["rel"]) for x in drive if np.isfinite(x["rel"])])
out.append(f"\n상자 1 m 앞에서 차 중심과 상자 중심의 옆 거리: 중앙 {100 * np.median(rr):.0f} cm, 10 cm 안(정면으로 다가감) {100 * np.mean(rr < 0.10):.0f} % (n {len(rr)})")
vs = np.array([x["vis_s"] for x in drive]); dv = np.array([x["d_vis"] for x in drive])
out.append(f"\n처음 보인 때(닿을 때까지 이어서 보인 구간 시작): 닿기 중앙 {np.median(vs):.1f} s 전 [사분위 {np.percentile(vs, 25):.1f}-{np.percentile(vs, 75):.1f}], 상자까지 중앙 {np.median(dv):.1f} m [{np.percentile(dv, 25):.1f}-{np.percentile(dv, 75):.1f}]")
out.append("\n\n| 정책·지도 | 세션 | 시각 s | 몇 번째 만남 | 가시성·회피 가능성 | 앞 바퀴 | 보임 s 전 / m | 속도 보임→닿음 m/s | 앞 바퀴 여유 cm | 1 m 앞 옆 거리 cm | 쪽 바꿈 | 계획 관통 | 상자 옆 빈 폭(넓은 쪽) m |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for x in sorted(rows, key=lambda x: (x["lab"], x["b"], x["t"])):
    out.append(f"| {NM[x['lab']]} | {x['b']} | {0.1 * x['t']:.1f} | {x['enc']} | {x['vis']} | {x['hist']} | {x['vis_s']:.1f} / {x['d_vis']:.1f} | "
               f"{x['v_vis']:.1f}→{x['v_hit']:.1f} | {'' if not np.isfinite(x['prev_clear']) else round(100 * x['prev_clear'])} | "
               f"{'' if not np.isfinite(x['rel']) else round(100 * x['rel'])} | {x['nsw'] if x['nsw'] >= 0 else ''} | {'예' if x['plan_through'] else '아니오'} | {x['room']:.2f} |")
open(sys.argv[4], "w").write("\n".join(out) + "\n")
print("\n".join(out[:40]))
