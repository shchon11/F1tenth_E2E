"""Tables from scored records: the competition table, paired comparisons with the 'same' verdict, the noise table."""
import os
import glob
import json

import numpy as np

from . import DATA
from .suite import load_suite
from .score import score_record, boot, boot_ratio, paired, mde

QUAL_KEYS = ("best", "streak_hard", "clean", "dnf", "dist", "wall_hard", "box_hard", "wall_any", "box_any", "stuck_s")


def _fmt(t, d=2, pct=False):
    e, lo, hi = t
    if not np.isfinite(e):
        return "–"
    if pct:
        return f"{100 * e:.0f} % [{100 * lo:.0f}–{100 * hi:.0f}]"
    return f"{e:.{d}f} [{lo:.{d}f}–{hi:.{d}f}]"


def load_scores(suite, condition, fmt="qual", seed=None):
    """{driver id: [score dicts]} of every record of this suite / condition (scored now if not yet)"""
    s = load_suite(suite)
    root = os.path.join(DATA, "raw", f"{s['name']}-{s['version']}", fmt, condition)
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "*.npz"))):
        if seed is not None and not f.endswith(f"_s{seed}.npz"):
            continue
        sj = f[:-4] + ".score.json"
        if not os.path.exists(sj) or os.path.getmtime(sj) < os.path.getmtime(f):
            M = int(s["race"]["M"])
            R = score_record(f, M=M)
            json.dump(R, open(sj, "w"), default=float)
        R = json.load(open(sj))
        jf = f[:-4] + ".job.json"
        job = json.load(open(jf)) if os.path.exists(jf) else {}
        R["label"] = (job.get("driver") or {}).get("label", "")
        R["imported"] = bool(job.get("imported_from"))
        R["opponent"] = (job.get("opponent") or {}).get("id")
        R["start"] = job.get("start")
        # comparable only within one cell definition: the record's cell hash against the current suite file's
        from .suite import cell_sha256
        m = next((x for x in s["maps"] if x["id"] == R.get("map_id")), None)
        if m is not None and job.get("cell_sha256") and R.get("seed") is not None:
            want = cell_sha256(s, condition, m, int(R["seed"]))
            R["stale_cell"] = job["cell_sha256"] != want
            if R["stale_cell"]:
                print(f"[compbench] WARNING {f}: made under another definition of this cell (cell hash differs)")
        out.setdefault(os.path.basename(os.path.dirname(f)), []).append(R)
    return out


def sessions_of(Rs, maps=None, seed=None):
    rows = []
    for R in Rs:
        if maps and R["map_id"] not in maps:
            continue
        if seed is not None and R.get("seed") != seed:
            continue
        for r in R["sessions"]:
            rows.append(dict(r, map_id=R["map_id"], seed=R.get("seed")))
    return rows


def qual_row(S):
    cl = [(r["map_id"], r["layout"]) for r in S]
    g = lambda k: np.array([r[k] for r in S], float)
    light = g("wall_any") + g("box_any") - g("wall_hard") - g("box_hard")
    return dict(n=len(S), best=boot(g("best"), cl, "median"), streak=boot(g("streak_hard"), cl),
                p10=float(np.percentile(g("streak_hard"), 10)) if len(S) else float("nan"),
                clean=boot(g("clean"), cl), dnf=boot(g("dnf"), cl), dist=boot(g("dist"), cl),
                wall_hard=boot(g("wall_hard"), cl), box_hard=boot(g("box_hard"), cl), light=boot(light, cl),
                stuck=boot(g("stuck_s"), cl), laps=boot(g("n_laps"), cl),
                sev3=boot(g("sev_ge3"), cl) if "sev_ge3" in S[0] else (float("nan"),) * 3)


QUAL_HEAD = ("| 운전자 | 세션 | 최고 랩 s | 무충돌(세게) 연속 바퀴 최대, 평균 · 하위 10 % | 한 번도 세게 안 닿은 세션 | 완주 실패 | "
             "120 s 달린 거리 m | 세게 닿음/세션 벽 | 세게 닿음/세션 상자 | 약하게 닿음/세션 | 3 m/s 넘게 닿음/세션 | 멈춘 시간 s |\n|"
             + "---|" * 12)


def qual_line(name, q):
    return ("| " + " | ".join([name, str(q["n"]), _fmt(q["best"]), f"{_fmt(q['streak'], 1)} · {q['p10']:.0f}", _fmt(q["clean"], pct=True),
                               _fmt(q["dnf"], pct=True), _fmt(q["dist"], 0), _fmt(q["wall_hard"]), _fmt(q["box_hard"]),
                               _fmt(q["light"]), _fmt(q["sev3"]), _fmt(q["stuck"], 1)]) + " |")


def race_rows(R, sel=lambda r: True):
    rows = [r for r in R["race"] if sel(r)]
    ses = [s for r, s in zip(R["race"], R["sessions"]) if sel(r)]
    return rows, ses


RACE_HEAD = ("| 운전자 · 상대 · 출발 | 경기 | 승률 | M바퀴 먼저 채움 | 앞지름/경기 | 앞지름 당함/경기 | 뒤 0.5-4 m에 갇힌 시간 s | "
             "차 접촉/경기 (세게) | 우리 과실 접촉/경기 (뒤에서 닿음) | env 규칙상 우리 책임/경기 | 벽·상자 세게/경기 | 완주 실패 |\n|" + "---|" * 12)


def race_line(name, rows, ses):
    cl = [r["layout"] for r in rows]
    g = lambda k: np.array([r[k] for r in rows], float)
    gs = lambda k: np.array([s[k] for s in ses], float)
    cf = g("cfault") if rows and "cfault" in rows[0] else None
    return ("| " + " | ".join([name, str(len(rows)), _fmt(boot(g("win"), cl), pct=True), f"{100 * g('won_M').mean():.0f} %",
                               _fmt(boot(g("passes"), cl)), f"{g('passed_by').mean():.2f}", _fmt(boot(g("behind_s"), cl), 1),
                               f"{_fmt(boot(g('car_onsets'), cl))} ({g('car_hard').mean():.2f})", _fmt(boot(g("ego_fault"), cl)),
                               f"{np.nanmean(cf):.2f}" if cf is not None else "–",
                               f"{(gs('wall_hard') + gs('box_hard')).mean():.2f}", f"{100 * gs('dnf').mean():.0f} %"]) + " |")


def table_for(suite, condition, fmt="qual", drivers=None, maps=None, seed=None):
    from . import drivers as D
    s = load_suite(suite)
    seed = seed if seed is not None else s["seed"]
    sc = load_scores(suite, condition, fmt, seed=seed)
    ids = None
    if drivers:
        ids = [D.resolve(d)["id"] if not d.endswith("_tune") else d for d in drivers]
    out = [f"### {s['_tag']} · {fmt} · {condition} · seed {seed}\n"]
    if fmt == "qual":
        map_ids = [m["id"] for m in s["maps"] if (not maps or m["id"] in maps)]
        for mid in map_ids:
            lines = []
            for did, Rs in sc.items():
                if ids and did not in ids:
                    continue
                S = sessions_of(Rs, [mid])
                if not S:
                    continue
                lab = Rs[0].get("label") or did
                tag = " (C 기록)" if any(R.get("imported") for R in Rs if R["map_id"] == mid) else ""
                lines.append(qual_line(f"{did}{tag}", qual_row(S)))
            if lines:
                out.append(f"\n지도 {mid}\n\n" + QUAL_HEAD + "\n" + "\n".join(lines))
        pool_maps = [m["id"] for m in s["maps"] if (not maps or m["id"] in maps) and not m.get("check_only")]
        if len(pool_maps) > 1:
            lines = []
            for did, Rs in sc.items():
                if ids and did not in ids:
                    continue
                have = {R["map_id"] for R in Rs}
                if not set(pool_maps) <= have:
                    continue
                lines.append(qual_line(did, qual_row(sessions_of(Rs, pool_maps))))
            if lines:
                out.append(f"\n지도 {len(pool_maps)}개를 같은 무게로 합침 ({', '.join(pool_maps)}; 모두 갖춘 운전자만)\n\n" + QUAL_HEAD + "\n" + "\n".join(lines))
    else:
        lines = []
        for did, Rs in sc.items():
            if ids and did not in ids:
                continue
            for R in Rs:
                if "race" not in R:
                    continue
                nm = f"{did} · {R.get('opponent')} · {R.get('start')}"
                lines.append(race_line(nm, *race_rows(R)))
                if R["race"] and "c_pace" in R["race"][0]:
                    for p in sorted({r["c_pace"] for r in R["race"]}):
                        lines.append(race_line(f"{nm} · 상대 {p:.2f}배", *race_rows(R, lambda r, p=p: r["c_pace"] == p)))
                    for c, cn in ((0, "반 바퀴"), (1, "5 m 뒤")):
                        lines.append(race_line(f"{nm} · {cn}", *race_rows(R, lambda r, c=c: r["c_start"] == c)))
        out.append(RACE_HEAD + "\n" + "\n".join(lines))
    return "\n".join(out) + "\n"


COMPARE_KEYS = (("best", "최고 랩 s (중앙값 차)", "median"), ("streak_hard", "무충돌 연속 바퀴 최대", "mean"),
                ("clean", "무충돌 세션 비율", "mean"), ("dnf", "완주 실패 비율", "mean"), ("dist", "120 s 거리 m", "mean"),
                ("wall_hard", "세게 닿음 벽/세션", "mean"), ("box_hard", "세게 닿음 상자/세션", "mean"))


def compare(suite, condition, a, b, maps=None, fmt="qual", seed=None, maps_b=None):
    """B - A on the same sessions; `maps_b`: B's records from these maps instead, paired with A's maps in order (the same
    layouts on another version of the floor, e.g. the raw-grid stairs against the smoothed walls)"""
    from . import drivers as D
    s = load_suite(suite)
    seed = seed if seed is not None else s["seed"]
    sc = load_scores(suite, condition, fmt, seed=seed)
    ia, ib = D.resolve(a)["id"], D.resolve(b)["id"]
    A = sessions_of(sc[ia], maps)
    if maps_b:
        ren = dict(zip(maps_b, maps))
        B = [dict(r, map_id=ren[r["map_id"]]) for r in sessions_of(sc[ib], maps_b)]
    else:
        B = sessions_of(sc[ib], maps)
    lines = [f"### {s['_tag']} · {condition} · {ib}{(' @ ' + ','.join(maps_b)) if maps_b else ''} − {ia} (같은 세션끼리 짝지은 차이, 95 % 구간) · 지도 {maps or '전체'}\n",
             "| 지표 | 차이 [95 % 구간] | 짝 수 | 판정 |", "|---|---|---|---|"]
    for k, name, st in COMPARE_KEYS:
        p = paired(A, B, k, st)
        verdict = "같다고 봄 (구간이 0을 포함)" if p["same"] else (f"{ib}가 큼" if p["est"] > 0 else f"{ib}가 작음")
        lines.append(f"| {name} | {p['est']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] | {p['n']} | {verdict} |")
    lines.append("\n규칙: 같은 세션(지도·시드·세션 번호가 같은 것)끼리 차이를 내고, 상자 배치 단위로 다시 뽑는 부트스트랩 2000번의 "
                 "95 % 구간이 0을 포함하면 '같다'고 판정한다.")
    return "\n".join(lines)


def noise(suite, condition, driver, maps=("k26scan",), seeds=None):
    """the same driver on the suite's seed and its noise seeds: per-seed means, their spread, and the paired
    seed-vs-seed difference per session -> the smallest difference N sessions resolve"""
    from . import drivers as D
    s = load_suite(suite)
    did = D.resolve(driver)["id"]
    seeds = seeds or [s["seed"]] + list(s.get("noise_seeds") or [])
    per = {}
    for sd in seeds:
        sc = load_scores(suite, condition, "qual", seed=sd)
        if did in sc:
            per[sd] = sessions_of(sc[did], list(maps))
    keys = (("best", "최고 랩 s", "median"), ("streak_hard", "무충돌 연속 바퀴", "mean"), ("clean", "무충돌 세션", "mean"),
            ("dnf", "완주 실패", "mean"), ("dist", "거리 m", "mean"), ("wall_hard", "벽 세게/세션", "mean"),
            ("box_hard", "상자 세게/세션", "mean"))
    lines = [f"### 시험 잡음: {did} · {condition} · 지도 {list(maps)} · 시드 {list(per)}\n",
             "| 지표 | 시드별 값 | 시드 사이 최대−최소 | 한 시드 95 % 구간 폭(평균) | 세션당 짝 차이 SD | 32 | 64 | 128 | 256 | 512 세션이면 가를 수 있는 차이 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    sds = sorted(per)
    for k, name, st in keys:
        vals, widths = [], []
        for sd in sds:
            S = per[sd]
            cl = [(r["map_id"], r["layout"]) for r in S]
            e, lo, hi = boot([r[k] for r in S], cl, st)
            vals.append(e)
            widths.append(hi - lo)
        dif = []
        for i in range(len(sds)):
            for j in range(i + 1, len(sds)):
                ia = {(r["map_id"], r["b"]): r for r in per[sds[i]]}
                for r in per[sds[j]]:
                    q = ia.get((r["map_id"], r["b"]))
                    if q is not None and np.isfinite(q[k]) and np.isfinite(r[k]):
                        dif.append(r[k] - q[k])
        sd_pair = float(np.std(dif)) if dif else float("nan")
        m = [mde(sd_pair, n) for n in (32, 64, 128, 256, 512)]
        lines.append(f"| {name} | {' / '.join(f'{v:.3f}' for v in vals)} | {max(vals) - min(vals):.3f} | {np.mean(widths):.3f} | "
                     f"{sd_pair:.3f} | " + " | ".join(f"{x:.3f}" for x in m) + " |")
    lines.append("\n세션당 짝 차이 SD는 같은 정책을 같은 세션(같은 배치·같은 물리 뽑기 번호)에서 시드만 바꿔 다시 잰 값의 차이다. "
                 "'가를 수 있는 차이' = 2.8 × SD / √N (짝지은 비교, 유의수준 5 %, 검정력 80 %). 다른 두 정책의 짝 차이 SD는 "
                 "이보다 크거나 같다(정책끼리의 세션별 차이가 더해짐) -- 아래 비교 표의 구간 폭으로 확인.")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- 1v1 with the catch-up
CATCH_HEAD = ("| 운전자 · 상대 · 출발 | 경기 | 승률 | 따라잡은 경기 | 따라잡은 경기의 승률 | 따라잡은 뒤 차 접촉/경기 (세게) | "
              "따라잡은 뒤 우리 과실 접촉/경기 | 따라잡은 뒤 잃은 시간 s | 따라잡은 뒤 앞지름/경기 | 따라잡은 뒤 뒤에 붙은 시간 s | "
              "따라잡은 뒤 벽·상자 세게/경기 | 완주 실패 |\n|" + "---|" * 12)


def catch_line(name, rows, ses):
    if not rows:
        return None
    cl = [r["layout"] for r in rows]
    c = [r for r in rows if r.get("caught")]
    ccl = [r["layout"] for r in c]
    g = lambda k, rr: np.array([r[k] for r in rr], float)
    gs = lambda k: np.array([s[k] for s in ses], float)
    f = lambda k, d=2: _fmt(boot(g(k, c), ccl), d) if c else "–"
    return ("| " + " | ".join([name, str(len(rows)), _fmt(boot(g("win", rows), cl), pct=True), f"{len(c)}",
                               _fmt(boot(g("win", c), ccl), pct=True) if c else "–",
                               f"{f('car_after')} ({np.nanmean(g('car_hard_after', c)):.2f})" if c else "–", f("fault_after"),
                               f("lost_after_s", 1), f("passes_after"), f("behind_after_s", 1), f("wallbox_hard_after"),
                               f"{100 * gs('dnf').mean():.0f} %"]) + " |")


def race_catch_table(suite, condition, drivers=None, seed=None):
    """the 1v1 table: per opponent and start; race4 also per pace (half a lap apart = REPRESENTATIVE)"""
    s = load_suite(suite)
    seed = seed if seed is not None else s["seed"]
    sc = load_scores(suite, condition, "race", seed=seed)
    lines = []
    for did, Rs in sc.items():
        if drivers and did not in drivers:
            continue
        for R in Rs:
            if "race" not in R:
                continue
            base = f"{did} · {R.get('opponent')}"
            if R["race"] and "c_pace" in R["race"][0]:
                for c0, cn in ((0, "반 바퀴(대표)"), (1, "5 m 뒤")):
                    for p in sorted({r["c_pace"] for r in R["race"]}):
                        ln = catch_line(f"{base} {p:.2f}배 · {cn}", *race_rows(R, lambda r, p=p, c0=c0: r["c_pace"] == p and r["c_start"] == c0))
                        if ln:
                            lines.append(ln)
            else:
                ln = catch_line(f"{base} · {'반 바퀴(대표)' if R.get('start') == 'half_lap' else R.get('start')}", *race_rows(R))
                if ln:
                    lines.append(ln)
    return f"### {s['_tag']} · 1대1 · {condition} · seed {seed}\n\n" + CATCH_HEAD + "\n" + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------- opponent qualification
#: what each opponent kind reads as ground truth, and where its car settings come from -- from the code (11-28)
OPP_FACTS = {
    "ftg_at_k26comp": ("Follow-the-Gap", "없음: 자기 LiDAR 270° 스캔 + 오도메트리 속도만 (`gym_env._external_commands`: lr.scan, lr.odom)",
                       "명령은 공칭 차 기준 (조향각·속도), env가 차마다 서보 보정만 해 줌; 설정 = 옆 지도 korea_2026_competition 6b차 튜닝 solo #15"),
    "disparity_at_k26comp": ("Disparity Extender", "없음: 자기 LiDAR 스캔만",
                             "공칭 차; 설정 = korea_2026_competition 1차 튜닝 solo #8"),
    "forzaeth_at_k26comp": ("ForzaETH 모듈형 (내 차 자리용)", "없음: 자기 파티클 필터 위치추정(오도메트리+스캔), 스캔 검출·추적, 공칭 구동기 모델 (`modular_stack.ModularStack`)",
                            "최소시간 레이싱라인(이 지도에서 최적화) + MAP 조향표는 공칭 차(params.py 기본값)로 만듦; 설정 = korea_2026_competition 6b차 튜닝 solo #14"),
    "forzaeth_gt_at_k26comp": ("ForzaETH (상대 자리 구현 = 참값)", "자기 위치·속도 참값(env.sim.state), 다른 차와 상자의 참 위치 (`spliner_teacher.py` 301-344행 '오라클'), 이 에피소드의 지연·서보·보정·마찰 값까지 보정에 씀 (`teacher.RacelineTeacher.__call__`의 P)",
                               "레이싱라인 한계 = 내 차 자리에선 튜닝값, 상대 자리에선 state_rl.build_env의 7 / 6.5 / 4 고정; 차 모델은 공칭 차"),
    "aurorus_at_default": ("AuRoRus 회피 주행", "자기 위치 참값(env.sim.state; 원본의 시뮬레이션 모드), 장애물은 자기 LiDAR",
                           "최소시간 라인(한계 7 / 6.5 / 4) 속도의 0.8배; 공개 기본값, 이 차에 맞춘 적 없음"),
    "raceline_at_k26comp": ("레이싱라인 + MAP", "내 차 자리: 파티클 필터 위치추정 / 상대 자리: 참값; 상자를 피하지 않음",
                            "korea_2026_competition 6b차 튜닝; 공칭 차"),
    "exam4ref_u500": ("자기 복사본 (scan31b u500)", "참값 입력은 없음(자기 LiDAR·IMU·기억) -- 대신 plan을 '완벽히' 실행(ideal: 미끄러짐·지연·서보 없음)",
                      "학습된 정책 그대로; 실행이 실차와 다름"),
}


def qualification(suite, maps, e2e_ref="scan31b_u475"):
    """opponent qualification: every opponent kind ALONE on the suite's sessions (as the opponent seat runs it), with the
    E2E reference for scale. Qualifies = DNF <= 10 %, mean clean streak >= 5 laps, best lap <= 1.5 x the E2E's."""
    s = load_suite(suite)
    rows = []
    ref_best = {}
    for cond in ("real", "ideal"):
        sc = load_scores(suite, cond, "qual", seed=s["seed"])
        for did, Rs in sc.items():
            for mid in maps:
                S = sessions_of(Rs, [mid])
                if not S:
                    continue
                q = qual_row(S)
                if did == e2e_ref and cond == "real":
                    ref_best[mid] = q["best"][0]
                if did in OPP_FACTS and not (did == "exam4ref_u500" and cond != "ideal"):
                    rows.append((did, cond, mid, q))
    out = [f"### 상대 자격 표 · {s['_tag']} · 지도 {list(maps)} (혼자 달린 120 s 세션, 상자 60 s에 치움)\n",
           "| 상대 종류 | 조건 · 지도 | 세션 | 최고 랩 s | 무충돌 연속 바퀴 평균 | 완주 실패 | 세게 닿음/세션 벽 · 상자 | 참값(GT) 사용 | 차량·설정 출처 | 자격 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for did, cond, mid, q in rows:
        name, gt, src = OPP_FACTS[did]
        rb = ref_best.get(mid, float("nan"))
        ok = (q["dnf"][0] <= 0.10) and (q["streak"][0] >= 5.0) and (np.isfinite(q["best"][0]) and np.isfinite(rb)
                                                                  and q["best"][0] <= 1.5 * rb)
        out.append(f"| {name} | {cond} · {mid} | {q['n']} | {_fmt(q['best'])} | {_fmt(q['streak'], 1)} | {_fmt(q['dnf'], pct=True)} | "
                   f"{q['wall_hard'][0]:.2f} · {q['box_hard'][0]:.2f} | {gt} | {src} | {'통과' if ok else '탈락'} |")
    out.append(f"\n자격 기준(이 표에서 정함): 완주 실패 ≤ 10 %, 무충돌 연속 바퀴 평균 ≥ 5, 최고 랩 ≤ 같은 지도 E2E({e2e_ref}, real) 최고 랩의 1.5배"
               f" (E2E 최고 랩: {', '.join(f'{k} {v:.2f} s' for k, v in ref_best.items())}).")
    return "\n".join(out)


def mde_pairs(suite, condition, a, b, maps=None, seed=None):
    """two different drivers on the same sessions: the SD of the per-session difference per metric and the smallest
    true difference N paired sessions resolve (80 % power, 5 %)"""
    from . import drivers as D
    s = load_suite(suite)
    seed = seed if seed is not None else s["seed"]
    sc = load_scores(suite, condition, "qual", seed=seed)
    ia, ib = D.resolve(a)["id"], D.resolve(b)["id"]
    A = {(r["map_id"], r["b"]): r for r in sessions_of(sc[ia], maps)}
    Bs = sessions_of(sc[ib], maps)
    lines = [f"### 두 정책 짝 비교의 해상도: {ib} − {ia} · {condition} · 지도 {maps or '전체'}\n",
             "| 지표 | 세션당 짝 차이 SD | 32 | 64 | 128 | 256 | 512 세션이면 가를 수 있는 차이 |", "|---|---|---|---|---|---|---|"]
    for k, name, _ in COMPARE_KEYS:
        dif = [r[k] - A[(r["map_id"], r["b"])][k] for r in Bs if (r["map_id"], r["b"]) in A
               and np.isfinite(r[k]) and np.isfinite(A[(r["map_id"], r["b"])][k])]
        sd = float(np.std(dif)) if dif else float("nan")
        lines.append(f"| {name} | {sd:.3f} | " + " | ".join(f"{mde(sd, n):.3f}" for n in (32, 64, 128, 256, 512)) + " |")
    return "\n".join(lines)
