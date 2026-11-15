"""(11-15 21:20, 사용자: "Map 튜닝으로는 안돼?", "잘좀 해봐봐") MAP을 지금 policy의 plan에 맞춰 튜닝: 목표 = plan 경로와의 옆 거리 RMSE
(10-20 규칙: 제어기는 궤적과의 거리로 맞춤, 랩타임·충돌로 맞추지 않음 -- 벽·속도는 참고로 같이 적음).

같은 시험 세션(exam1, 시드 고정, 128 세션 x 30 s, 상자 그대로)을 MAP 설정만 바꿔 달림 -> 각 결정 t에서 낸 plan(10점)의 경로와
0.1-0.4 s 뒤 실제 위치의 거리. 진화 탐색: 1세대 = 지금 설정, 논문대로(B), 무작위 4; 이후 상위 2개 둘레에서 6개씩, 폭을 줄여 가며.
탐색 공간: 논문/코드 갈림 4개(각도 기준, 앞점, a_lat 속도, 표 보간) + L1(m, q, 최소) + 조향 배율·가감속 배율 + 지연 보상 + 조향용 속도 앞보기
+ 속도 명령 앞당김. 결과: out/maptune/tune.jsonl (한 줄 = 설정 하나), 기록 out/maptune/cNNN.npz
인자: 체크포인트 [세대 수 10]"""
import json, os, subprocess, sys, time
import numpy as np

T = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(T, "..", "out", "maptune")
CK = sys.argv[1]
GENS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SECS = 30
SLOTS = [("0", 2), ("1", 1)]                     # CUDA_VISIBLE_DEVICES (0 = 4070, 1 = 5060), concurrent runs each
ENV = dict(os.environ, PYTHONPATH="/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros", WANDB_MODE="disabled",
           F1SIM_RACELINE_CACHE_ONLY="1", F1SIM_PLAN_LAT_MARGIN="0.95", F1SIM_LIDAR_MOUNT_X="0.12555")

NOW = dict(eta_ref="heading", l1_point="arc", steer_speed="planned", lut_interp="code", m_l1=0.47, q_l1=0.0, t_clip_min=0.35,
           v_steer_gain_max=1.281, acc_scaler_for_steer=0.886, dec_scaler_for_steer=0.809, delay_comp=0.0214,
           speed_lookahead_for_steer=0.06, lead=0.0)
PAPER = dict(NOW, eta_ref="velocity", l1_point="euclid", steer_speed="measured", lut_interp="linear", v_steer_gain_max=1.0,
             acc_scaler_for_steer=1.0, dec_scaler_for_steer=1.0)
CAT = {"eta_ref": ("heading", "velocity"), "l1_point": ("arc", "euclid"), "steer_speed": ("planned", "measured"),
       "lut_interp": ("code", "linear")}
NUM = {"m_l1": (0.2, 0.7), "q_l1": (-0.5, 0.4), "t_clip_min": (0.3, 1.0), "v_steer_gain_max": (1.0, 1.5),
       "acc_scaler_for_steer": (0.7, 1.3), "dec_scaler_for_steer": (0.7, 1.3), "delay_comp": (0.0, 0.1),
       "speed_lookahead_for_steer": (0.0, 0.2), "lead": (0.0, 0.15)}
rng = np.random.default_rng(1115)


def mutate(c, sig):
    c = dict(c)
    for k, (lo, hi) in NUM.items():
        c[k] = float(np.clip(c[k] + rng.normal() * sig * (hi - lo), lo, hi))
    for k, opts in CAT.items():
        if rng.random() < 0.15:
            c[k] = opts[1 - opts.index(c[k])]
    return c


def rand():
    c = {k: str(rng.choice(v)) for k, v in CAT.items()}
    c.update({k: float(rng.uniform(lo, hi)) for k, (lo, hi) in NUM.items()})
    return c


def seg_dist(p, a, b):
    """distance from points p (..., 2) to segments a->b (..., S, 2): min over S"""
    ab = b - a; ap = p[..., None, :] - a
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(ap - t[..., None] * ab, axis=-1).min(-1)


VB = (2.0, 4.0, 5.5, 7.0, 10.0)
AB = (0.0, 2.0, 4.0, 6.0, 20.0)
#: bin weights = how often the current setting's plans fall in each (speed, lateral accel) cell (c000), fixed
W = None


def score(fn):
    z = np.load(fn)
    st, pl, rs = z["st"], z["pl"], z["rs"]
    c = list(z["cols"]); cnt = z["cnt"]
    Tn = pl.shape[0]; lat = {k: [] for k in (1, 2, 3, 4)}; dv = []; pe, pv_, pa = [], [], []
    for t in range(Tn - 5):
        path = np.concatenate([st[t, :, None, :2], pl[t]], 1)                      # (B, 11, 2)
        v0 = np.linalg.norm(pl[t, :, 0] - st[t, :, :2], axis=-1) / 0.1
        for k in (1, 2, 3, 4):
            ok = ~rs[t:t + k + 1].any(0) & (v0 > 2.0)
            d = seg_dist(st[t + k, :, :2], path[:, :-1], path[:, 1:])
            lat[k].append(d[ok])
        # the trajectory error: where the plan says the car is at t + k (k = 1..4), point to point
        seg = path[:, 1:6] - path[:, :5]
        vv = np.linalg.norm(seg, axis=-1) / 0.1
        h = np.arctan2(seg[..., 1], seg[..., 0]); turn = np.abs(np.angle(np.exp(1j * (h[:, -1] - h[:, 0]))))
        alat = vv.mean(1) ** 2 * turn / np.maximum(np.linalg.norm(seg, axis=-1).sum(1), 0.05)
        ok4 = ~rs[t:t + 5].any(0) & (v0 > 2.0)
        e = np.stack([np.linalg.norm(st[t + k, :, :2] - pl[t, :, k - 1], axis=-1) for k in (1, 2, 3, 4)], 1)
        pe.append(e[ok4]); pv_.append(v0[ok4]); pa.append(alat[ok4])
        ok = ~rs[t:t + 4].any(0) & (v0 > 2.0)
        vp = np.linalg.norm(pl[t, :, 3] - pl[t, :, 2], axis=-1) / 0.1
        dv.append((st[t + 3, :, 3] - vp)[ok])
    r = {f"lat{k}": float(np.sqrt(np.mean(np.concatenate(v) ** 2))) for k, v in lat.items()}
    r["lat_rmse"] = float(np.mean([r[f"lat{k}"] for k in (1, 2, 3, 4)]))
    r["lat4_p90"] = float(np.percentile(np.concatenate(lat[4]), 90))
    r["dv_med"] = float(np.median(np.concatenate(dv)))
    e = np.concatenate(pe); v0 = np.concatenate(pv_); al = np.concatenate(pa)
    iv = np.digitize(v0, VB) - 1; ia = np.digitize(al, AB) - 1
    cell = np.full((len(VB) - 1, len(AB) - 1), np.nan); n = np.zeros_like(cell)
    for i in range(len(VB) - 1):
        for j in range(len(AB) - 1):
            m = (iv == i) & (ia == j)
            n[i, j] = m.sum()
            if m.sum() >= 50:
                cell[i, j] = np.sqrt(np.mean(e[m] ** 2))
    r["cells"] = cell.tolist(); r["cell_n"] = n.tolist()
    r["v_mean"] = float(st[:-1, :, 3].mean())
    for nm in ("hard_wall", "hard_prop", "hard_car"):
        r[nm] = float(cnt[-1, :, c.index(nm)].mean())
    return r


def launch(i, cfg, gpu):
    out = os.path.join(O, f"c{i:03d}.npz")
    mp = {k: v for k, v in cfg.items() if k != "lead"}
    mp["tw_speed_lead"] = float(cfg["lead"])                   # 11-15: the follower's default now carries a lead of its own
    env = dict(ENV, CUDA_VISIBLE_DEVICES=gpu, F1SIM_MAP_OVERRIDE=json.dumps(mp), F1SIM_EXAM_SPEED_LEAD=str(cfg["lead"]))
    log = open(out.replace(".npz", ".log"), "w")
    return subprocess.Popen(["timeout", "1200", sys.executable, "-u", "-W", "ignore", os.path.join(T, "exam_record.py"), CK, out,
                             "map", "256", "uniform", str(SECS), "50"], cwd="/home/shchon11/F1tenth_E2E/f1sim", env=env,
                            stdout=log, stderr=subprocess.STDOUT), out


def objective(d):
    if "cells" not in d:
        return 9.9
    c = np.array(d["cells"], float)
    w = W if W is not None else np.array(d["cell_n"], float)
    c = np.where(np.isnan(c), 0.5, c)
    return float((c * w).sum() / w.sum())


done = []                                                      # (score dict, cfg)
jl = os.path.join(O, "tune.jsonl")
if os.path.exists(jl):                                         # resume
    for line in open(jl):
        d = json.loads(line); done.append(d)
# rescore every finished record with the current objective (11-15 21:35: the first objective, lateral RMSE pooled, preferred
# settings that drove slowly -- 3.1-3.6 m/s mean vs 4.5 -- because slow driving has small errors)
for d in done:
    fn = os.path.join(O, f"c{d['i']:03d}.npz")
    if os.path.exists(fn) and "cells" not in d:
        d.update(score(fn))
c0 = [d for d in done if d["i"] == 0 and "cell_n" in d]
if c0:
    W = np.array(c0[0]["cell_n"], float)
for d in done:
    d["obj"] = objective(d)
with open(jl, "w") as f:
    for d in done:
        f.write(json.dumps(d) + "\n")
idx = 1 + max([d["i"] for d in done], default=-1)


def run_batch(cfgs):
    global idx
    queue = list(cfgs); running = []
    while queue or running:
        bank = subprocess.run(["pgrep", "-f", "layout_bank.py"], capture_output=True).returncode == 0
        for gpu, cap in SLOTS:
            cap = 1 if (gpu == "0" and bank) else cap               # RAM 14 GB: the Korea 2026 bank (2 GB) runs beside
            while queue and sum(1 for g, *_ in running if g == gpu) < cap:
                cfg = queue.pop(0)
                p, out = launch(idx, cfg, gpu); running.append((gpu, p, out, cfg, idx)); idx += 1
        time.sleep(5)
        for item in list(running):
            gpu, p, out, cfg, i = item
            if p.poll() is not None:
                running.remove(item)
                if p.returncode == 0 and os.path.exists(out):
                    d = dict(i=i, cfg=cfg, **score(out))
                else:
                    d = dict(i=i, cfg=cfg, lat_rmse=9.9, failed=p.returncode)
                global W
                if d.get("i") == 0 and "cell_n" in d:
                    W = np.array(d["cell_n"], float)
                d["obj"] = objective(d)
                done.append(d)
                with open(jl, "a") as f:
                    f.write(json.dumps(d) + "\n")
                print(f"[{time.strftime('%H:%M')}] c{i:03d} 궤적 오차 {d['obj'] * 100:.2f} cm, 평균 속도 {d.get('v_mean', 0):.2f} m/s, 옆 RMSE {d['lat_rmse'] * 100:.2f} cm"
                      + (f" (0.1/0.4 s {d['lat1'] * 100:.1f}/{d['lat4'] * 100:.1f}, 90 % {d['lat4_p90'] * 100:.1f}) 벽 {d['hard_wall']:.2f} 상자 {d['hard_prop']:.2f} 속도차 {d['dv_med']:+.2f}" if 'lat1' in d else " 실패")
                      + f" | {json.dumps(cfg)}", flush=True)


if not done:
    run_batch([NOW, PAPER] + [rand() for _ in range(4)])
gen0 = max(0, len(done) // 6)
for g in range(gen0, GENS):
    ranked = sorted(done, key=lambda d: d["obj"])
    sig = 0.25 * (0.75 ** g)
    kids = [mutate(ranked[0]["cfg"], sig) for _ in range(4)] + [mutate(ranked[1]["cfg"], sig) for _ in range(2)]
    run_batch(kids)
ranked = sorted(done, key=lambda d: d["obj"])
print("\n상위 5:")
for d in ranked[:5]:
    print(f"  c{d['i']:03d} 궤적 오차 {d['obj'] * 100:.2f} cm, 속도 {d.get('v_mean', 0):.2f}, 옆 RMSE {d['lat_rmse'] * 100:.2f} cm, 벽 {d.get('hard_wall', 0):.2f} | {json.dumps(d['cfg'])}")
