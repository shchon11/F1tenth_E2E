"""T-미로 넛지 시험 (11-27, rq5_offpolicy_design.md 6.1 '0단계: 참 기울기'와 같은 물음을 깨끗한 쌍 비교로; T-미로 v4 자리 고르기의 목적 함수).
물음: 상자가 보이기 전에 미리 움직이는 것(기억으로 할 수 있는 일)이 지금 policy가 모는 이 자리에서 실제로 몇 초를 버나, 그리고 상자가 없을 때 늘
그렇게 하면(기억 없는 조심) 몇 초를 잃나.
방법: 같은 checkpoint, 같은 world, 같은 시드로 T-미로 시험(exam_record.py ... tmaze)을 두 번 -- 그대로(기준) / F1SIM_EXAM_NUDGE(갈래).
  갈래는 정한 b의 세션에서 차가 상자 자리 D_start -> D_end m 앞에 있는 동안만 policy의 행동을 바꿈:
    lat y: policy 자신의 빈 트랙 길(기준 시험의 '없음' 세션, tmaze_measure.ref_line)을 y m 옆(왼쪽 +)으로 옮긴 길을 따라가게 첫 0.1 s 조향을 바꿈
          (옮긴 길은 벽에서 0.3 m 안쪽까지만 + 길 1 m에 0.6 m 넘게 꺾이지 않게; 조향은 policy 자신의 것에서 옆가속도 alat(기본
          3 m/s^2)까지만 벗어남 -- 11-27 첫 판은 제한이 없어 옮기는 조향이 타이어 한계를 넘었고 plan 속도가 ~4.5 m/s로 깎였음: 그 느려짐은
          넛지가 만든 것이지 policy가 고른 것이 아님)
          주의: 'hold'(통과까지)는 짠 조향이 상자 옆 통로까지 몰아 주는지에 달림 -- 벽 여유·꺾임 제한 때문에 통로에 못 닿으면 막힌 쪽에서
          부딪힘(표의 옆@0으로 확인). 그때 그 갈래는 '미리 고른 길의 값'을 재지 못함
    cap C: plan 속도를 C m/s 아래로
  바뀐 행동으로 plan 전체를 다시 굴려 차는 그 plan을 정확히 몰고(ideal) 다음 결정도 그 plan을 읽음 -- policy가 그렇게 골랐을 때와 같음.
  창은 처음 보임(기준 시험에서 막는 쪽 상자 어느 부분이든 처음 보인 자리, 2번째 이후 통과의 중앙값 D_s) 앞 T s: D_end = D_s, D_start = 기준
  시험의 통과들이 D_s에 닿기 T s 전에 있던 자리의 중앙값. 'hold' 갈래는 D_end = 0 (통과까지 유지: 미리 고른 길을 끝까지 가는 policy).
  시드가 같아 첫 창 전까지 두 시험은 같은 차·같은 잡음 -> 통과를 (세션, 몇 번째)로 짝지어 차이를 냄(짝 표준오차); 짝 없는 평균 차이도 함께.
재는 것(통과마다): 구간 시간(상자 자리 20 m 앞 -> 5 m 뒤, tmaze_measure.SEC_M), 구간 안 상자 세게 닿음, 벽 세게 닿음, 창 끝(D_s)과 자리(D 0)에서
  옆 위치(기준 길에서, 왼쪽 +)와 속도. b(L/R/없음) x 통과(1번째 / 2번째 이후)별.
사용:
  python3 tmaze_nudge.py prep 기준.npz 접두어          -> 접두어_line.npy(기준 길), 접두어_prep.json(D_s, T -> D_start 표)
  python3 tmaze_nudge.py run checkpoint world 접두어 갈래json [갈래json ...]   (기준 시험 + 갈래마다 시험, CUDA_VISIBLE_DEVICES는 부르는 쪽)
     갈래json = {"name": .., "b": [..], "T": 초 | "d": [D_end, D_start], "lat": y, "cap": C, "alat": a, "hold": 0|1, "side": "L"|"R"}
  python3 tmaze_nudge.py table 기준.npz 갈래.npz [...]   -> 표 (갈래 이름은 파일 이름에서)
분석 도구일 뿐 -- 학습에 쓰지 않음."""
import os, sys, json, math, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rec_lib import Rec, along_track
import tmaze_measure as TMM

PY = "/home/shchon11/F1tenth_E2E/.venv/bin/python"
NAME_B = ("L", "R", "없음")
D_GRID = np.round(np.arange(0.0, 30.01, 0.25), 2)
N_ENV = int(os.environ.get("TMAZE_NUDGE_ENVS", "384"))
SECS = os.environ.get("TMAZE_NUDGE_SECS", "32")
MEM_MIN_GB = 6.0             # [GB] the host's available memory a run needs before it starts (14 GB host, an OOM froze it once)


def spots(z, P, L):
    """each b's spot along the track [m] (the empty sessions: the middle of L and R) and its xy"""
    Lb, Rb = (z[k_].astype(float).reshape(-1, 4)[:, :2].mean(0) for k_ in ("tm_L", "tm_R"))
    xy_ = {0: Lb, 1: Rb, 2: 0.5 * (Lb + Rb)}
    sb = {}
    for k_, c_ in xy_.items():
        j_ = int(np.linalg.norm(P - c_, axis=1).argmin())
        sb[k_] = float(TMM.project(P, c_[None], np.array([j_ * TMM.BIN]), L)[0][0])
    return sb, xy_


def passes(fn, P, d_at):
    """one row per pass: sess, b, n (1 = first), t, section time, box / wall hard contacts in the section (nan: never left it),
    offset (left +, from P) and speed when the car's along-track place is each d_at m before the spot, nudged in this approach"""
    R = Rec(fn); z = R.z; T, Bn, dt = R.T, R.B, R.dt
    b = z["tm_b"].astype(int)
    st = R.st.astype(float); xy = st[:, :, :2]; V = np.abs(st[:, :, 3])
    s, lap, L = along_track(z, T + 1)
    prog = s + lap * L
    sb, _ = spots(z, P, L)
    nud = z["nudge"].astype(bool) if "nudge" in z.files else np.zeros((T, Bn), bool)
    out = []
    for i in range(Bn):
        s_b = sb[b[i]]
        n0 = np.floor((prog[0, i] - s_b) / L)
        nmax = np.maximum.accumulate(np.floor((prog[:, i] - s_b) / L))
        for t in np.nonzero(np.diff(nmax) > 0)[0]:
            if t < 2:
                continue
            k_ep = int(R.k0[min(t, T - 1), i])
            tgt = s_b + nmax[t + 1] * L

            def cross(v_, fwd):
                rng = range(t, min(t + 80, T)) if fwd else range(t, k_ep - 1, -1)
                for j in rng:
                    if prog[j, i] <= v_ < prog[j + 1, i] and R.k0[min(j, T - 1), i] == k_ep:
                        return j + (v_ - prog[j, i]) / max(prog[j + 1, i] - prog[j, i], 1e-9)
                return np.nan
            c0, c1 = cross(tgt - TMM.SEC_M[0], False), cross(tgt + TMM.SEC_M[1], True)
            tsec = (c1 - c0) * dt if np.isfinite(c0) and np.isfinite(c1) else np.nan
            j0 = int(np.floor(c0)) if np.isfinite(c0) else max(t - 40, k_ep)
            j1 = int(np.floor(c1)) if np.isfinite(c1) else min(t + 40, T - 1)
            box = float(R.d[j0:j1 + 1, i, 4].sum() > 0) if b[i] < 2 else np.nan
            wall = float(R.d[j0:j1 + 1, i, 3].sum())
            offs, vs = [], []
            for D_ in d_at:
                c_ = cross(tgt - D_, False)
                if not np.isfinite(c_):
                    offs.append(np.nan); vs.append(np.nan); continue
                j = int(np.floor(c_)); f = c_ - j
                pq = xy[j, i] + f * (xy[j + 1, i] - xy[j, i])
                offs.append(float(TMM.project(P, pq[None], np.array([(tgt - D_) % L]), L)[1][0]))
                vs.append(float(V[j, i] + f * (V[j + 1, i] - V[j, i])))
            out.append(dict(sess=i, b=int(b[i]), n=int(nmax[t + 1] - n0), t=int(t), tsec=tsec, box=box, wall=wall,
                            off=np.array(offs), v=np.array(vs), nudged=bool(nud[j0:t + 1, i].any())))
    return R, out


def prep(base, prefix):
    """the policy's empty-track line, each b's first sight (median over passes 2+), and how far before it the cars are T s earlier"""
    R, rows, g = TMM.one(base)
    assert g is not None, "기준 시험에 빈 트랙 길이 없음"
    P, L = g["P"], g["L"]
    # the wall along the normal on each side [m] (exam_record keeps the shifted line 0.3 m off them)
    tan = np.roll(P, -1, 0) - np.roll(P, 1, 0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1); r_ = np.arange(0.0, 4.0, 0.01); tk = int(R.tid[0, 0])
    side = lambda sg: np.array([float(r_[np.argmax(R.wall_dist(tk, P[j][None] + sg * r_[:, None] * nrm[j][None]) <= 0.0)])
                                for j in range(len(P))])
    np.save(prefix + "_line.npy", np.concatenate([P, side(1.0)[:, None], side(-1.0)[:, None]], 1).astype(np.float32))
    z = R.z; T = R.T; dt = R.dt
    s, lap, _ = along_track(z, T + 1); prog = s + lap * L
    info = {"base": os.path.abspath(base), "lap_m": L, "sb": {str(k): v for k, v in g["sb"].items()}, "sight": {}, "tau": {}}
    for k in (0, 1):
        gl = np.array([r["glimpse"] for r in rows if r["b"] == k and r["n"] >= 2], float)
        D_s = float(np.median(gl)) if gl.size else 0.0
        info["sight"][NAME_B[k]] = {"median": D_s, "p90": float(np.percentile(gl, 90)) if gl.size else 0.0, "n": int(gl.size)}
        # time from the car's crossing of D before the spot to its crossing of D_s, passes 2+ (median over passes)
        taus = []
        for r in rows:
            if r["b"] != k or r["n"] < 2:
                continue
            i, t = r["sess"], r["t"]; k_ep = int(R.k0[min(t, T - 1), i])
            tgt = g["sb"][k] + np.floor((prog[t + 1, i] - g["sb"][k]) / L) * L

            def cross(v_):
                for j in range(t, k_ep - 1, -1):
                    if prog[j, i] <= v_ < prog[j + 1, i]:
                        return (j + (v_ - prog[j, i]) / max(prog[j + 1, i] - prog[j, i], 1e-9)) * dt
                return np.nan
            ts = cross(tgt - D_s)
            taus.append([ts - cross(tgt - D_) if D_ >= D_s else np.nan for D_ in D_GRID])
        tau = np.nanmedian(np.array(taus, float), 0) if taus else np.full(len(D_GRID), np.nan)
        info["tau"][NAME_B[k]] = [None if not np.isfinite(x) else round(float(x), 3) for x in tau]
    info["D_grid"] = D_GRID.tolist()
    json.dump(info, open(prefix + "_prep.json", "w"), indent=1)
    print(f"기준 길 {prefix}_line.npy ({P.shape[0]}점), 한 바퀴 {L:.2f} m, 자리 {g['sb']}")
    for k in (0, 1):
        si = info["sight"][NAME_B[k]]
        tau = np.array([np.nan if x is None else x for x in info["tau"][NAME_B[k]]])
        print(f"  {NAME_B[k]}: 처음 보임 중앙 {si['median']:.2f} m (90 % {si['p90']:.2f}, {si['n']}번); 그 T s 전 자리: "
              + ", ".join(f"T {T_:g} s -> {d_start(info, NAME_B[k], T_):.2f} m" for T_ in (0.5, 1.0, 2.0)))
    return info


def d_start(info, side, T_):
    tau = np.array([np.nan if x is None else x for x in info["tau"][side]]); D = np.array(info["D_grid"])
    ok = np.isfinite(tau)
    if not ok.any() or tau[ok].max() < T_:
        return float(D[ok].max()) if ok.any() else np.nan
    return float(np.interp(T_, tau[ok], D[ok]))


def mem_ok():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / 1e6 >= MEM_MIN_GB
    return True


def exam(ck, world, out, nudge=None, log=None):
    while not mem_ok():
        print(f"  메모리 여유 {MEM_MIN_GB} GB 기다림", flush=True); time.sleep(30)
    env = dict(os.environ, F1SIM_EXAM_WORLD=os.path.abspath(world))
    env.pop("F1SIM_EXAM_NUDGE", None)
    if nudge:
        env["F1SIM_EXAM_NUDGE"] = json.dumps(nudge)
    with open(log or out[:-4] + ".log", "w") as fh:
        r = subprocess.run([PY, os.path.join(HERE, "exam_record.py"), ck, out, "ideal", str(N_ENV), "tmaze", SECS, "999"], cwd=HERE,
                           stdout=fh, stderr=subprocess.STDOUT, env=env)
    return r.returncode


def run(ck, world, prefix, arms):
    base = prefix + "_base.npz"
    if not os.path.exists(base):
        t0 = time.time(); rc = exam(ck, world, base)
        print(f"기준 시험 {base}: {rc} ({time.time() - t0:.0f} s)", flush=True)
    info = json.load(open(prefix + "_prep.json")) if os.path.exists(prefix + "_prep.json") else prep(base, prefix)
    for a in arms:
        side = a.get("side", "L")
        D_s = float(info["sight"][side]["median"])
        if "d" in a:
            d0, d1 = (float(x) for x in a["d"])
        else:
            d1 = d_start(info, side, float(a["T"])); d0 = 0.0 if a.get("hold") else D_s
        nd = {"b": a.get("b", [0, 1, 2]), "d": [round(d0, 3), round(d1, 3)]}
        if a.get("lat"):
            nd.update(lat=float(a["lat"]), line=os.path.abspath(prefix + "_line.npy"))
        if a.get("cap"):
            nd["cap"] = float(a["cap"])
        if a.get("alat"):
            nd["alat"] = float(a["alat"])
        out = f"{prefix}_{a['name']}.npz"
        if os.path.exists(out):
            print(f"  {a['name']}: 있음", flush=True); continue
        t0 = time.time(); rc = exam(ck, world, out, nd)
        print(f"  갈래 {a['name']} {nd}: {rc} ({time.time() - t0:.0f} s)", flush=True)


def table(base, arms, d_at=None):
    info_f = base[:-len("_base.npz")] + "_prep.json" if base.endswith("_base.npz") else None
    info = json.load(open(info_f)) if info_f and os.path.exists(info_f) else None
    P = np.load(base[:-len("_base.npz")] + "_line.npy").astype(float)[:, :2] if info else TMM.one(base)[2]["P"]
    if d_at is None:
        ds = sorted({round(float(info["sight"][k]["median"]), 1) for k in ("L", "R")}) if info else [8.0]
        d_at = tuple([0.0] + ds)
    _, B = passes(base, P, d_at)
    key = lambda r: (r["sess"], r["n"])
    Bk = {key(r): r for r in B}
    grp = (("1번째", lambda r: r["n"] == 1), ("2번째 이후", lambda r: r["n"] >= 2))
    print(f"== 기준 {os.path.basename(base)} · 통과 {len(B)}번 · 옆 위치·속도는 상자 자리 {d_at} m 앞 (기준 길에서 왼쪽 +)")
    if info:
        print("   처음 보임 중앙: " + ", ".join(f"{k} {info['sight'][k]['median']:.2f} m" for k in ("L", "R")))
    hdr = (f"{'갈래':<16}{'b':<5}{'통과':<9}{'n':>4}{'넛지%':>6}{'구간 s':>8}{'차(짝)':>15}{'차(짝 없음)':>15}{'상자 hard':>10}{'벽 hard':>8}"
           + "".join(f"{'옆@' + format(d, 'g'):>8}{'v@' + format(d, 'g'):>7}" for d in d_at))
    def line(name, rows, rows_b=None):
        for k in range(3):
            for gname, f in grp:
                rr = [r for r in rows if r["b"] == k and f(r)]
                if not rr:
                    continue
                x = np.array([r["tsec"] for r in rr], float)
                cell = f"{np.nanmean(x):8.3f}"
                dp = du = ""
                if rows_b is not None:
                    pr_ = [(r["tsec"], Bk[key(r)]["tsec"]) for r in rr if key(r) in Bk]
                    d_ = np.array([a_ - b_ for a_, b_ in pr_], float); d_ = d_[np.isfinite(d_)]
                    if len(d_) > 1:
                        dp = f"{d_.mean():+.3f}±{d_.std(ddof=1) / np.sqrt(len(d_)):.3f}"
                    xb = np.array([r["tsec"] for r in rows_b if r["b"] == k and f(r)], float)
                    xa, xb = x[np.isfinite(x)], xb[np.isfinite(xb)]
                    if len(xa) > 1 and len(xb) > 1:
                        du = f"{xa.mean() - xb.mean():+.3f}±{np.sqrt(xa.var(ddof=1) / len(xa) + xb.var(ddof=1) / len(xb)):.3f}"
                bx = np.array([r["box"] for r in rr], float)
                wl = np.array([r["wall"] for r in rr], float)
                nd = np.mean([r["nudged"] for r in rr])
                o = np.array([r["off"] for r in rr], float); v = np.array([r["v"] for r in rr], float)
                ov = "".join(f"{np.nanmean(o[:, q]):+8.2f}{np.nanmean(v[:, q]):7.2f}" for q in range(len(d_at)))
                print(f"{name:<16}{NAME_B[k]:<5}{gname:<9}{len(rr):4d}{100 * nd:6.0f}{cell}{dp:>15}{du:>15}"
                      f"{(f'{100 * np.nanmean(bx):5.1f} %' if k < 2 else '-'):>10}{np.nanmean(wl):8.3f}{ov}")
    print(hdr)
    line("기준", B)
    for fn in arms:
        _, A = passes(fn, P, d_at)
        z = np.load(fn)
        nm = os.path.basename(fn)[:-4]
        nm = nm[len(os.path.basename(base)[:-len("base.npz")]):] if base.endswith("_base.npz") else nm
        cfg = str(z["nudge_cfg"]) if "nudge_cfg" in z.files else "{}"
        print(f"-- {nm}: {cfg}")
        line(nm, A, B)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(sys.argv[2], sys.argv[3])
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], [json.loads(a) for a in sys.argv[5:]])
    elif cmd == "table":
        table(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
