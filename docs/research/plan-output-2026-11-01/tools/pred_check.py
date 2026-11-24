"""상대 차 예측을 policy가 쓰나 (11-24, race32c --pred-opp; 사용자가 OK한 두 시험만). exam_record.py race4 기록 셋을 읽음:
  보통  = 그 checkpoint의 보통 exam4 기록 (lab_race_exam_loop.sh가 만든 것)
  probe = F1SIM_EXAM_PRED_PROBE=1 기록 (+ _pred.npz): 주행은 보통과 같아야 함, 결정마다 예측만 바꾼 plan의 변화
  oracle = F1SIM_EXAM_PRED_ORACLE=1 기록 (+ _pred.npz): 예측 입력을 상대 차 복사본의 자기 plan으로 바꿔 몬 exam4
찍는 것:
  (0) probe 기록이 보통 기록과 비트까지 같나 + 같은 입력으로 다시 돈 plan이 몬 plan과 같나
  (가) 시험 A: exam_race_table.py 표([전체], 출발별/상대 페이스별/상자별) 보통 vs 오라클, 세션 짝 차이(오라클 - 보통)와 95 % 구간
  (나) 예측이 실제와 얼마나 맞나: 머리와 오라클의 0.5 / 1.0 s 뒤 오차 (실제 = 기록에서 나중에 본 상대 차 자리, plan_rl 타깃 규칙:
       그 사이 다시 출발·2 m 넘는 순간 이동 없음, 10 m 안)
  (다) 시험 B: '있음' 결정에서 예측을 바꿨을 때 plan 10점이 움직인 거리(평균, 0.5 s 점, 1.0 s 점)와 속도 변화, 두 차 거리별 중앙값 / p90,
       기준 = 우리 속도 입력 +0.5 m/s, 우리 지난 plan 0.5 m 왼쪽
인자: 보통npz probe npz oracle npz"""
import io, os, sys, runpy, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(0)
f_norm, f_probe, f_orc = sys.argv[1:4]


def table(fn):
    """exam_race_table.py on a record: (its printed text, its per-session arrays)"""
    buf = io.StringIO()
    argv0 = sys.argv
    sys.argv = [os.path.join(HERE, "exam_race_table.py"), fn]
    try:
        with contextlib.redirect_stdout(buf):
            g = runpy.run_path(os.path.join(HERE, "exam_race_table.py"), run_name="__main__")
    finally:
        sys.argv = argv0
    return buf.getvalue(), g


def head_part(txt):
    """the table up to its per-condition block"""
    out = []
    for ln in txt.splitlines():
        if ln.startswith("[조건마다]"):
            break
        out.append(ln)
    return "\n".join(out)


def boot_mean(d, n=4000):
    d = np.asarray(d, float)
    idx = RNG.integers(0, len(d), (n, len(d)))
    s = d[idx].mean(1)
    return d.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


# ---------------------------------------------------------------- (0) bit-identical
zn, zp, zo = np.load(f_norm), np.load(f_probe), np.load(f_orc)
pp, po = np.load(f_probe[:-4] + "_pred.npz"), np.load(f_orc[:-4] + "_pred.npz")
print("== 상대 차 예측을 policy가 쓰나 ·", str(zn["name"]), "· exam4 (시드", int(zn["seed"]), ") ==")
print("\n(0) probe 기록(예측만 바꿔 policy를 더 돌림, 주행은 그대로) = 보통 exam4 기록인가")
META = {"name", "exam", "pred_mode", "bank", "ref"}
diff = []
for k in sorted(set(zn.files) & set(zp.files)):
    if k in META:
        continue
    a, b = zn[k], zp[k]
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
        md = float(np.nanmax(np.abs(a.astype(float) - b.astype(float)))) if a.shape == b.shape and a.dtype.kind in "fiub" else float("nan")
        diff.append(f"{k} (최대 차 {md:.3g})")
drive = [k for k in ("st", "op", "pl", "cnt", "ocnt", "rem", "orem", "rs", "ors", "lap", "cfault", "held", "vy") if k in zn.files]
print(f"  비교한 배열 {len(set(zn.files) & set(zp.files) - META)}개 (주행: {', '.join(drive)}): "
      + ("전부 비트까지 같음" if not diff else "다름 -> " + "; ".join(diff)))
be = pp["base_err"].max(0)
print(f"  같은 입력으로 다시 돈 plan과 몬 plan의 차 최대: 점 {be[0]:.2e} m, 속도 {be[1]:.2e} m/s (결정 {pp['base_err'].shape[0]}개 x 세션 전부)")

# ---------------------------------------------------------------- (가) test A
tn, gn = table(f_norm)
to, go = table(f_orc)
print("\n(가) 시험 A: 예측 입력을 오라클(상대 차 복사본의 자기 plan, P(보임) = '있음' 규칙대로 1 / 0)로 바꿔 몬 exam4, 같은 시드·세션")
print("-- 보통 (머리의 예측) --")
print(head_part(tn))
print("-- 오라클 --")
print(head_part(to))
B = len(gn["made"])
assert np.array_equal(gn["c_pace"], go["c_pace"]) and np.array_equal(gn["c_start"], go["c_start"]) and np.array_equal(gn["c_box"], go["c_box"])
MET = [("바퀴", lambda g: g["laps"]), ("앞지름", lambda g: g["made"]), ("우리 책임 차 접촉", lambda g: g["fault"]),
       ("세게 닿음 차", lambda g: g["hard"][:, 2]), ("세게 닿음 벽", lambda g: g["hard"][:, 0]),
       ("세게 닿음 상자", lambda g: g["hard"][:, 1]), ("멈춘 세션", lambda g: (g["stalls"] > 0).astype(float))]
c_start, c_pace, c_box = gn["c_start"], gn["c_pace"], gn["c_box"]
GROUPS = [("전체", np.ones(B, bool)), ("출발 반 바퀴", c_start == 0), ("출발 5 m 뒤", c_start == 1)]
GROUPS += [(f"상대 {'제 페이스' if p >= 0.999 else f'{p:.2f}배'}", np.isclose(c_pace, p)) for p in sorted(set(c_pace.tolist()))]
GROUPS += [("상자 있음", c_box), ("상자 없음", ~c_box)]
print("-- 세션 짝 차이: 오라클 - 보통, 평균 [95 % 구간, 세션 다시 뽑기] (멈춘 세션은 비율의 차) --")
print(f"{'조건':<14} | {'세션':>3} | " + " | ".join(f"{m:>22}" for m, _ in MET))
for lab, m in GROUPS:
    cells = []
    for _, f in MET:
        mu, lo, hi = boot_mean(f(go)[m] - f(gn)[m])
        cells.append(f"{mu:+.2f} [{lo:+.2f} {hi:+.2f}]")
    print(f"{lab:<14} | {int(m.sum()):>3} | " + " | ".join(f"{c:>22}" for c in cells))
n_same = int(sum(np.array_equal(gn["made"][i:i + 1], go["made"][i:i + 1]) for i in range(B)))
print(f"  (앞지름 수가 같은 세션 {n_same}/{B})")

# ---------------------------------------------------------------- (나) realised error
def realised(z):
    """(T, B, 10, 2) where the other car really was 0.1 .. 1.0 s after each decision, in our car's frame at it, and (T, B, 10)
    whether that counts (plan_rl --pred-opp's target rule)"""
    st, op = z["st"].astype(float), z["op"].astype(float)
    rs, ors = z["rs"].astype(bool), z["ors"].astype(bool)
    T, Bz = rs.shape
    jmp = np.linalg.norm(op[1:T + 1, :, :2] - op[:T, :, :2], axis=-1) > 2.0
    cb = np.cumsum(rs | ors | jmp, 0)
    tgt = np.full((T, Bz, 10, 2), np.nan); ok = np.zeros((T, Bz, 10), bool)
    for k in range(1, 11):
        n = T + 1 - k
        if n <= 0:
            break
        n = min(n, T)
        d = op[k:k + n, :, :2] - st[:n, :, :2]
        c, s = np.cos(st[:n, :, 2]), np.sin(st[:n, :, 2])
        rel = np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], -1)
        before = np.concatenate([np.zeros((1, Bz)), cb[:n - 1]], 0)
        ok_ = (cb[k - 1:k - 1 + n] - before == 0) & (np.linalg.norm(rel, axis=-1) <= 10.0)
        tgt[:n, :, k - 1] = rel; ok[:n, :, k - 1] = ok_
    return tgt, ok


def err_line(lab, pred, tgt, ok, sel):
    cells = []
    for k in (5, 10):
        m = ok[..., k - 1] & sel
        e = np.linalg.norm(pred[:, :, k - 1] - tgt[:, :, k - 1], axis=-1)[m]
        cells.append(f"{k / 10:.1f} s: 중앙 {np.median(e):.2f} · 평균 {e.mean():.2f} · p90 {np.percentile(e, 90):.2f} m (결정 {m.sum()})")
    print(f"  {lab:<26} " + " | ".join(cells))


print("\n(나) 예측이 실제와 얼마나 맞나: 상대 차가 0.5 / 1.0 s 뒤 실제로 있던 자리(기록에서 나중에 봄)와의 거리")
for nm, z, pd_ in (("보통 주행(probe 기록)", zp, pp), ("오라클 주행", zo, po)):
    tgt, ok = realised(z)
    th = pd_["there"].astype(bool)
    print(f" [{nm}] '있음' 결정 {100 * th.mean():.0f} %, 그때 머리의 P(보임) 중앙 {np.median(pd_['head_p'][th]):.2f} "
          f"(p10 {np.percentile(pd_['head_p'][th], 10):.2f}), '없음'일 때 중앙 {np.median(pd_['head_p'][~th]):.2f}")
    err_line("머리 · '있음' 결정", pd_["head"], tgt, ok, th)
    err_line("오라클 · '있음' 결정", pd_["orc"], tgt, ok, th)
    err_line("머리 · 타깃 있는 결정 전부", pd_["head"], tgt, ok, np.ones_like(th))
    pace = np.broadcast_to(pd_["pace_of"][None, :], th.shape)
    for lo_, hi_, lab in ((0.0, 0.7, "상대 0.6배"), (0.7, 0.9, "상대 0.8배"), (0.9, 1.01, "상대 0.95배·제 페이스")):
        sel = th & (pace >= lo_) & (pace < hi_)
        err_line(f"  머리 · '있음' · {lab}", pd_["head"], tgt, ok, sel)
        err_line(f"  오라클 · '있음' · {lab}", pd_["orc"], tgt, ok, sel)

# ---------------------------------------------------------------- (다) test B
print("\n(다) 시험 B: '있음' 결정에서 예측 입력만 바꿨을 때 plan이 움직인 정도 (주행은 그대로, 바꾼 plan은 몰지 않음)")
names, cols = [str(x) for x in pp["probe_names"]], [str(x) for x in pp["probe_cols"]]
LAB = {"pred_left": "예측 점 0.5 m 왼쪽", "pred_right": "예측 점 0.5 m 오른쪽", "pred_ahead": "예측 점 0.5 m 앞",
       "pred_behind": "예측 점 0.5 m 뒤", "pred_gone": "P(보임) = 없음", "own_speed+0.5": "[기준] 우리 속도 입력 +0.5 m/s",
       "own_lastplan_left0.5": "[기준] 우리 지난 plan 0.5 m 왼쪽"}
PR = pp["probe"]; th = pp["there"].astype(bool); dist = pp["dist"]
ci = {c: cols.index(c) for c in cols}
BINS = [("3 m 안", 0.0, 3.0), ("3-6 m", 3.0, 6.0), ("6-10 m", 6.0, 10.01), ("전체(10 m 안)", 0.0, 10.01)]
print("  칸 = 중앙값 / p90. plan 10점 평균 이동·0.5 s 점·1.0 s 점 [m], 속도 변화 = 10점 속도 차 절댓값 평균 [m/s];"
      " 부호 = 1.0 s 점의 옆(+왼쪽)·앞(+) 이동과 속도 차 평균(+빠르게)의 중앙값")
for blab, lo_, hi_ in BINS:
    m = th & (dist >= lo_) & (dist < hi_)
    print(f" [두 차 거리 {blab}: 결정 {int(m.sum())}개, 머리 P(보임) 중앙 {np.median(pp['head_p'][m]) if m.any() else float('nan'):.2f}]")
    print(f"  {'바꾼 것':<26} | {'평균 이동':>11} | {'0.5 s 점':>11} | {'1.0 s 점':>11} | {'속도 변화':>11} | {'부호: 옆 · 앞 · 속도':>20}")
    for j, nm in enumerate(names):
        v = PR[:, :, j][m]
        f2 = lambda c: f"{np.median(v[:, ci[c]]):.3f}/{np.percentile(v[:, ci[c]], 90):.3f}"
        sg = f"{np.median(v[:, ci['dy_10']]):+.3f} · {np.median(v[:, ci['dx_10']]):+.3f} · {np.median(v[:, ci['dv_mean']]):+.3f}"
        print(f"  {LAB.get(nm, nm):<26} | {f2('mean_disp'):>11} | {f2('disp_05'):>11} | {f2('disp_10'):>11} | {f2('dv_abs'):>11} | {sg:>20}")
m = th & (pp["head_p"] > 0.5)
print(f" [머리도 P(보임) > 0.5라고 한 '있음' 결정만: {int(m.sum())}개] 1.0 s 점 이동 중앙/p90: "
      + ", ".join(f"{LAB.get(nm, nm)} {np.median(PR[:, :, j][m][:, ci['disp_10']]):.3f}/{np.percentile(PR[:, :, j][m][:, ci['disp_10']], 90):.3f}"
                  for j, nm in enumerate(names)))
