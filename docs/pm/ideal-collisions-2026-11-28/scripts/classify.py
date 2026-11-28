"""events.py 결과 -> 사건 분류표, 비율(배치 단위 부트스트랩 95 %), 통과 여유 분포와 여유별 다음 바퀴 접촉률.
인자: events.json [out_prefix]"""
import sys, json
import numpy as np

RNG = np.random.default_rng(0)


def boot_prop(flags, clusters, n=2000):
    f = np.asarray(flags, float); cl = np.asarray(clusters)
    if len(f) == 0:
        return (np.nan, np.nan, np.nan)
    u = np.unique(cl); idx = {c: np.nonzero(cl == c)[0] for c in u}
    out = []
    for _ in range(n):
        s = np.concatenate([idx[c] for c in RNG.choice(u, len(u))])
        out.append(f[s].mean())
    return (f.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5))


def boot_ratio(num, den, clusters, n=2000):
    num = np.asarray(num, float); den = np.asarray(den, float); cl = np.asarray(clusters)
    u = np.unique(cl); idx = {c: np.nonzero(cl == c)[0] for c in u}
    out = []
    for _ in range(n):
        s = np.concatenate([idx[c] for c in RNG.choice(u, len(u))])
        out.append(num[s].sum() / max(den[s].sum(), 1e-9))
    return (num.sum() / max(den.sum(), 1e-9), np.percentile(out, 2.5), np.percentile(out, 97.5))


def cls(h):
    f0 = h["feas"]["d0.0_brake"][0]; f3 = h["feas"]["d0.3_brake"][0]
    f3s = h["feas"]["d0.3_steer"][0]
    if h["rehit_in_enc"]:
        vis = "끼임 중 재충돌"
    elif not h["seen"]:
        vis = "안 보임"
    elif not f0:
        vis = "늦게 보임: 물리적으로 불가"
    elif not f3:
        vis = "늦게 보임: 즉시 반응해야 가능"
    else:
        vis = "일찍 보임: 0.3 s 늦게 반응해도 가능"
    prev_clean = [p for p in h["prev"] if p["any"] == 0 and p["v"] > 1.0]
    if h["n_prev"] == 0:
        hist = "첫 만남"
    elif any(p["hard"] > 0 for p in h["prev"][-1:]):
        hist = "지난 만남에도 세게 닿음"
    elif h["prev"][-1]["any"] > 0:
        hist = "지난 만남에 약하게 닿음"
    else:
        last = h["prev"][-1]
        same = last["side"] == h["enc_side"]
        if last["clear"] < 0.10:
            hist = "지난 통과 여유 <10 cm" + (" (같은 쪽)" if same else " (반대쪽)")
        else:
            hist = "지난 통과 여유 >=10 cm" + (" (같은 쪽)" if same else " (반대쪽)")
    plan = "계획이 상자를 지나감" if h["plan_clear_last"] <= 0.0 else "계획은 비켜 감"
    return vis, hist, plan


def main():
    E = json.load(open(sys.argv[1]))
    pre = sys.argv[2] if len(sys.argv) > 2 else None
    allrows = []
    for lab, r in E.items():
        H = r["hard"]
        print(f"\n===== {lab}: 세게 닿은 상자 {len(H)}건 (세션 {r['B']}개)")
        rows = []
        for h in H:
            vis, hist, plan = cls(h)
            last = h["prev"][-1] if h["prev"] else None
            rows.append(dict(lab=lab, b=h["b"], t=h["t"], layout=h["b"] % 64, vis=vis, hist=hist, plan=plan,
                             vis_s=h["vis_s"], d_vis=h["d_vis"], v_vis=h["v_vis"], v_hit=h["v_hit"], D=h["D"],
                             yreq=min(h["yreq_l"] if h["room_l"] >= 0.31 else 9, h["yreq_r"] if h["room_r"] >= 0.31 else 9),
                             room=h["wall_free_max"], enc_no=h["enc_no"],
                             prev_clear=last["clear"] if last else np.nan, prev_side_same=(last["side"] == h["enc_side"]) if last else None,
                             through_s=(h["t"] - h["plan_through_from"]) * 0.1 if h["plan_through_from"] is not None else 0.0,
                             reach_s=(h["t"] - h["reach_from"]) * 0.1 if h["reach_from"] is not None else np.nan,
                             n_clear_after=h["n_plan_clear_after_reach"], f0=h["feas"]["d0.0_brake"][0],
                             f3=h["feas"]["d0.3_brake"][0], f3s=h["feas"]["d0.3_steer"][0], f0s=h["feas"]["d0.0_steer"][0]))
        allrows += rows
        for key in ("vis", "hist", "plan"):
            cnt = {}
            for x in rows:
                cnt[x[key]] = cnt.get(x[key], 0) + 1
            print(f"  [{key}] " + " | ".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])))
        print(f"  {'b':>3} {'t':>4} {'가시':<22} {'이력':<22} {'계획':<12} 보인s 보인m  v보임 v닿음  여유전 쪽같음 계획관통s 도달s 빈폭")
        for x in sorted(rows, key=lambda x: (x["b"], x["t"])):
            print(f"  {x['b']:>3} {x['t']:>4} {x['vis']:<22} {x['hist']:<22} {x['plan']:<12} {x['vis_s']:4.1f} {x['d_vis']:5.1f} {x['v_vis']:5.1f} {x['v_hit']:5.1f} "
                  f"{x['prev_clear']:6.2f} {str(x['prev_side_same']):>5} {x['through_s']:5.1f} {x['reach_s']:5.1f} {x['room']:4.2f}")
    # pooled shares with intervals (clusters = record x layout)
    print("\n===== 세 기록 합 (63건 안팎): 비율 [배치 단위 부트스트랩 95 %]")
    cl = [f"{x['lab']}:{x['layout']}" for x in allrows]
    for key in ("vis", "hist", "plan"):
        cats = sorted(set(x[key] for x in allrows))
        for c in cats:
            m, lo, hi = boot_prop([x[key] == c for x in allrows], cl)
            print(f"  [{key}] {c:<28} {sum(x[key] == c for x in allrows):3d}  {100 * m:4.0f} % [{100 * lo:.0f}-{100 * hi:.0f}]")
    if pre:
        json.dump(allrows, open(pre + "_rows.json", "w"), default=float)
    # ---------------- passes: margin distribution and next-pass contact
    print("\n===== 통과 여유 분포 (안 닿은 통과, 속도 > 1 m/s)와 여유별 다음 만남 접촉률")
    bins = [-1, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.50, 9]
    names = ["0-3", "3-6", "6-10", "10-15", "15-20", "20-30", "30-50", ">50"]
    tot = {k: [0, 0, 0] for k in names}
    P_rows = []
    for lab, r in E.items():
        enc = r["enc"]
        by = {}
        for e in enc:
            by.setdefault((e["b"], e["j"]), []).append(e)
        for (b, j), L in by.items():
            L = sorted(L, key=lambda e: e["t0"])
            for a, nx in zip(L[:-1], L[1:]):
                if a["any"] > 0 or a["v"] < 1.0:
                    continue
                P_rows.append(dict(lab=lab, cl=f"{lab}:{b % 64}", m=a["clear"], nany=int(nx["any"] > 0), nhard=int(nx["hard"] > 0),
                                   flip=int(nx["side"] != a["side"]), nclear=nx["clear"]))
        clean = [e["clear"] for e in enc if e["any"] == 0 and e["v"] > 1.0]
        q = np.percentile(clean, [5, 10, 25, 50, 75])
        print(f"  {lab}: 안 닿은 통과 {len(clean)}개, 여유 백분위 5/10/25/50/75 = " + " / ".join(f"{100 * x:.0f}" for x in q)
              + f" cm; 10 cm 안 {100 * np.mean(np.array(clean) < 0.10):.0f} %, 20 cm 안 {100 * np.mean(np.array(clean) < 0.20):.0f} %")
    m = np.array([p["m"] for p in P_rows])
    print(f"  {'여유 cm':<8} {'통과':>5} {'다음 닿음':>16} {'다음 세게':>16} {'다음에 쪽 바꿈':>14} {'쪽 바꿈일 때 닿음':>18} {'같은 쪽일 때 닿음':>18}")
    for i, nm in enumerate(names):
        sel = [p for p in P_rows if bins[i] <= p["m"] < bins[i + 1]]
        if not sel:
            continue
        c_ = [p["cl"] for p in sel]
        a_ = boot_prop([p["nany"] for p in sel], c_); h_ = boot_prop([p["nhard"] for p in sel], c_)
        fl = np.mean([p["flip"] for p in sel])
        sf = [p for p in sel if p["flip"]]; ss = [p for p in sel if not p["flip"]]
        print(f"  {nm:<8} {len(sel):5d} {100 * a_[0]:5.1f} [{100 * a_[1]:.1f}-{100 * a_[2]:.1f}] {100 * h_[0]:5.1f} [{100 * h_[1]:.1f}-{100 * h_[2]:.1f}] "
              f"{100 * fl:10.0f} % {sum(p['nany'] for p in sf):6d}/{len(sf):<5d}      {sum(p['nany'] for p in ss):6d}/{len(ss):<5d}")
    for thr in (0.10, 0.20):
        a = [p for p in P_rows if p["m"] < thr]; b_ = [p for p in P_rows if p["m"] >= thr]
        ra = boot_prop([p["nany"] for p in a], [p["cl"] for p in a]); rb = boot_prop([p["nany"] for p in b_], [p["cl"] for p in b_])
        print(f"  여유 < {100 * thr:.0f} cm: 다음 닿음 {sum(p['nany'] for p in a)}/{len(a)} = {100 * ra[0]:.1f} % [{100 * ra[1]:.1f}-{100 * ra[2]:.1f}]"
              f" | >= {100 * thr:.0f} cm: {sum(p['nany'] for p in b_)}/{len(b_)} = {100 * rb[0]:.1f} % [{100 * rb[1]:.1f}-{100 * rb[2]:.1f}]")
    # lap-to-lap margin change
    d = np.array([p["nclear"] - p["m"] for p in P_rows if p["flip"] == 0])
    print(f"  같은 쪽으로 다음에 지날 때 여유 변화: 중앙 {100 * np.median(d):.1f} cm, 절대값 중앙 {100 * np.median(np.abs(d)):.1f} cm, p90 {100 * np.percentile(np.abs(d), 90):.1f} cm")
    if pre:
        json.dump(P_rows, open(pre + "_passes.json", "w"), default=float)


if __name__ == "__main__":
    main()
