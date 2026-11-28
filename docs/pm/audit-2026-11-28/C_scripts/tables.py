"""Audit C: markdown tables from comp_metrics.py json files. Args: out.md metrics.json[:label] [...]
Qualifier rows (all sessions of a record); race rows per opponent condition (race4: per pace and start)."""
import sys, json
import numpy as np

RNG = np.random.default_rng(0)


def boot_idx(cl, n=2000):
    cl = np.asarray(cl); u = np.unique(cl); idx = [np.nonzero(cl == c)[0] for c in u]
    for _ in range(n):
        pick = RNG.integers(0, len(u), len(u))
        yield np.concatenate([idx[i] for i in pick])


def ci_stat(v, cl, f):
    v = np.asarray(v, float)
    est = f(v, np.arange(len(v)))
    bs = [f(v, s) for s in boot_idx(cl)]
    return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def mean_f(v, s):
    return np.nanmean(v[s])


def med_f(v, s):
    return np.nanmedian(v[s])


def fmt(t, d=2, pct=False):
    e, lo, hi = t
    if pct:
        return f"{100 * e:.0f} % [{100 * lo:.0f}–{100 * hi:.0f}]"
    return f"{e:.{d}f} [{lo:.{d}f}–{hi:.{d}f}]"


def ratio(S, num, den):
    a = np.array([sum(s[k] for k in num) for s in S], float); b = np.array([s[den] for s in S], float)
    cl = [s["cluster"] for s in S]
    f = lambda v, s: a[s].sum() / max(b[s].sum(), 1e-9)
    return ci_stat(a, cl, f)


def qual_rows(items):
    H = ("| 정책 · 조건 | 세션 | 최고 랩 s | 치운 뒤 최고 랩 s | 세션 중앙 랩 s | 연속 무(세게)접촉 바퀴 최대 평균 · p10 | 연속 무접촉 바퀴 최대 평균 | "
         "세게 닿음/바퀴 벽 · 상자 | 닿음(약한 것 포함)/바퀴 벽 · 상자 | 멈춘 시간 s/세션 | 완주 실패 |\n|" + "---|" * 11)
    out = [H]
    for lab, R in items:
        S = R["sessions"]; cl = [s["cluster"] for s in S]
        g = lambda k: np.array([s[k] for s in S], float)
        sh = g("streak_hard"); p10 = np.percentile(sh, 10)
        row = [lab, str(len(S)), fmt(ci_stat(g("best"), cl, med_f)), fmt(ci_stat(g("best_after"), cl, med_f)),
               fmt(ci_stat(g("median"), cl, med_f)),
               f"{fmt(ci_stat(sh, cl, mean_f), 1)} · {p10:.0f}", fmt(ci_stat(g("streak_any"), cl, mean_f), 1),
               f"{fmt(ratio(S, ['wall_hard'], 'n_laps'), 3)} · {fmt(ratio(S, ['box_hard'], 'n_laps'), 3)}",
               f"{fmt(ratio(S, ['wall_any'], 'n_laps'), 3)} · {fmt(ratio(S, ['box_any'], 'n_laps'), 3)}",
               fmt(ci_stat(g("stuck_s"), cl, mean_f), 1), fmt(ci_stat(g("dnf"), cl, mean_f), pct=True)]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def race_rows(items):
    H = ("| 정책 · 상대 · 조건 | 경기 | 승률 | M바퀴 먼저 | 바퀴 차(우리−상대) | 앞지름/경기 | 앞지름 당함/경기 | 차 닿음/경기 (세게) | "
         "우리가 뒤에서 닿음/경기 | 뒤에서 박힘/경기 | env 규칙 우리 책임/경기 | 뒤 4 m 안에 붙어 있던 시간 s | 벽·상자 세게/경기 | 완주 실패 |\n|" + "---|" * 14)
    out = [H]
    for lab, R, sel_name, sel in items:
        rows = [r for r, s in zip(R["race"], R["sessions"]) if sel(r)]
        ses = [s for r, s in zip(R["race"], R["sessions"]) if sel(r)]
        if not rows:
            continue
        cl = [r["cluster"] for r in rows]
        g = lambda k: np.array([r[k] for r in rows], float)
        gs = lambda k: np.array([s[k] for s in ses], float)
        won_m = np.array([1.0 if (r["how"] == "M laps" and r["win"] == 1.0) else 0.0 for r in rows])
        cf = g("cfault") if "cfault" in rows[0] else None
        row = [f"{lab} · {sel_name}", str(len(rows)), fmt(ci_stat(g("win"), cl, mean_f), pct=True), f"{100 * won_m.mean():.0f} %",
               fmt(ci_stat(g("lap_lead"), cl, mean_f)), fmt(ci_stat(g("passes"), cl, mean_f)), f"{g('passed_by').mean():.2f}",
               f"{fmt(ci_stat(g('car_onsets'), cl, mean_f))} ({g('car_hard').mean():.2f})",
               fmt(ci_stat(g("ego_fault"), cl, mean_f)), f"{g('rear_ended').mean():.2f}",
               f"{np.nanmean(cf):.2f}" if cf is not None else "–",
               fmt(ci_stat(g("behind_s"), cl, mean_f), 1),
               f"{(gs('wall_hard') + gs('box_hard')).mean():.2f}", f"{100 * gs('dnf').mean():.0f} %"]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    q, r = [], []
    for arg in sys.argv[2:]:
        fn, lab = (arg.split(":", 1) + [arg])[:2]
        for R in json.load(open(fn)):
            if "race" not in R:
                q.append((lab, R))
            else:
                q.append((lab + " (경주 세션)", R))
                r.append((lab, R, "전체", lambda x: True))
                if "c_pace" in R["race"][0]:
                    for p in sorted(set(x["c_pace"] for x in R["race"])):
                        r.append((lab, R, f"상대 {p:.2f}배", lambda x, p=p: x["c_pace"] == p))
                    for c, nm in ((0, "반 바퀴 출발"), (1, "5 m 뒤 출발")):
                        r.append((lab, R, nm, lambda x, c=c: x["c_start"] == c))
                    for c, nm in ((True, "상자 있음"), (False, "상자 없음")):
                        r.append((lab, R, nm, lambda x, c=c: x["c_box"] == c))
    md = "### 예선 형식\n\n" + qual_rows(q) + "\n\n### 본선 형식\n\n" + race_rows(r) + "\n"
    open(sys.argv[1], "w").write(md); print(md)
