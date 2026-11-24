"""(11-24, 사용자: scan31b2 u600을 새 지도(korea26_scan+duct+smooth) 콘솔에서 보고 "제어기 튜닝 이상하게 된거 아님? 존나 흔들리네", 이어서
"가 ㄱ") MAP을 새 지도·지금 policy의 plan에 다시 맞춤. map_tune.py(11-15, exam1·옛 지도·avg26s u900)를 exam3·scan31b2 u600·lab PC로 옮김.

목표(10-20 규칙): 따라가기 오차 = 결정마다 낸 plan과 0.1-0.5 s 뒤 실제 위치의 거리(점 대 점), 칸(plan 속도 x 옆가속도)별 RMSE를 지금 설정의
칸 수로 가중 평균 (maptune_metrics.py). 랩타임·충돌은 적기만 함 (속도로 맞추면 코너를 자름).
시험: exam3(얼린 시험, 시드 고정), 128 세션 x 30 s(상자 세션 64개는 상자 그대로), MAP 설정만 F1SIM_MAP_OVERRIDE로 바꿈 -- 학습·policy는 그대로.
lab PC에서 한 번에 하나씩(systemd-run MemoryMax=10G). race32c의 lab 시험(exam_record)이 돌고 있거나, 50의 배수 checkpoint가 새로 났는데
그 exam4가 아직 안 끝났으면 기다림. 이 PC에서는 기록을 받아 점수만 매김(작은 numpy).

탐색: 1세대 = 지금 설정, 지금 설정 한 번 더(같은 시드 -- 잡음 크기), 한 가지씩 바꾼 것들; 2세대부터 = 나아진 한 가지들을 합친 것 + 상위 2개
둘레에서 무작위로 바꾼 것들, 폭을 줄여 가며. 결과: out/maptune_scan/tune.jsonl (한 줄 = 설정 하나), 기록 out/maptune_scan/cNNN.npz
인자: search HH:MM(이 시각 뒤로는 새 시도 안 함) | one 이름 설정json [초 90] (전체 시험 하나: out/exam/exam3mt_<이름>.npz)"""
import glob, json, os, re, shlex, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maptune_metrics import score

R = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01"
O = os.path.join(R, "out", "maptune_scan")
H = "shchon11@10.100.0.2"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", H]
LC = "/home/shchon11/F1tenth/console_main/f1sim"            # the lab's code (lab_sync.sh: this repo's main HEAD)
LT = "/home/shchon11/F1tenth/exam_lab/tools_maptune"        # exam_record.py + world_of.py from this branch
LO = "/home/shchon11/F1tenth/exam_lab/out/maptune"
CK = "/home/shchon11/f1sim_runs/scan31b2/plan_u600.pt"
RACE = "/home/shchon11/f1sim_runs/race32c"
RACE_U0 = 650                                               # race32c's lab exam loop started here
ENV = dict(PYTHONPATH=LC, WANDB_MODE="disabled", F1SIM_RACELINE_CACHE_ONLY="1", F1SIM_PLAN_LAT_MARGIN="0.95",
           F1SIM_LIDAR_MOUNT_X="0.12555", CUDA_VISIBLE_DEVICES="0")

# the follower the console drives today: state_rl.PLAN_MAP under TW_MAP (state_rl.plan_env_fields, 11-15) -- the searched keys
NOW = dict(t_clip_min=0.78, m_l1=0.37, q_l1=-0.49, v_steer_gain_max=1.47, acc_scaler_for_steer=1.0, dec_scaler_for_steer=1.0,
           delay_comp=0.009, speed_lookahead_for_steer=0.133, tw_speed_lead=0.141, steer_rate_limit=0.4,
           l1_point="arc", steer_speed="planned", lut_interp="linear")
CAT = {"l1_point": ("arc", "euclid"), "steer_speed": ("planned", "measured"), "lut_interp": ("code", "linear")}
NUM = {"t_clip_min": (0.3, 1.6), "m_l1": (0.15, 0.75), "q_l1": (-0.7, 0.4), "v_steer_gain_max": (1.0, 1.7),
       "acc_scaler_for_steer": (0.7, 1.3), "dec_scaler_for_steer": (0.7, 1.3), "delay_comp": (0.0, 0.12),
       "speed_lookahead_for_steer": (0.0, 0.25), "tw_speed_lead": (0.0, 0.35), "steer_rate_limit": (0.02, 0.4)}
PROBES = [
    ("now", {}), ("now_again", {}),
    ("gain1.0", dict(v_steer_gain_max=1.0)), ("gain1.2", dict(v_steer_gain_max=1.2)),
    ("L1_long", dict(t_clip_min=1.0, m_l1=0.47, q_l1=-0.3)), ("L1_short", dict(t_clip_min=0.6, m_l1=0.3, q_l1=-0.5)),
    ("rate0.1", dict(steer_rate_limit=0.1)), ("lead0.3", dict(tw_speed_lead=0.3)), ("delay0.06", dict(delay_comp=0.06)),
    ("steer_v_measured", dict(steer_speed="measured")), ("euclid", dict(l1_point="euclid")),
]
rng = np.random.default_rng(1124)


def full(c):
    return dict(NOW, **c)


def mutate(c, sig):
    c = dict(c)
    for k, (lo, hi) in NUM.items():
        if rng.random() < 0.6:                                  # a few keys at a time: 13 dims, ~50 trials
            c[k] = float(np.clip(c[k] + rng.normal() * sig * (hi - lo), lo, hi))
    for k, opts in CAT.items():
        if rng.random() < 0.12:
            c[k] = opts[1 - opts.index(c[k])]
    return c


def ssh(cmd, timeout=120):
    try:
        return subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def lab_exam_running():
    r = ssh("ps -eo args | grep -c '[e]xam_record'")
    if r is None or r.returncode not in (0, 1):
        return True                                             # unknown = busy
    return int((r.stdout.strip() or "0")) > 0


def race_exam_pending():
    for f in glob.glob(os.path.join(RACE, "plan_u*.pt")):
        u = int(re.findall(r"plan_u(\d+)\.pt", f)[0])
        if u >= RACE_U0 and u % 50 == 0 and not os.path.exists(os.path.join(R, "out", "analysis", f"race32c_u{u}_exam4.done")):
            return u
    return 0


def wait_free(log):
    said = False
    while True:
        u = race_exam_pending()
        busy = lab_exam_running()
        if not u and not busy:
            return
        if not said:
            log(f"lab 시험 기다림 (race32c u{u} exam4 아직 안 끝남)" if u else "lab에서 exam_record가 돌고 있어 기다림")
            said = True
        time.sleep(30)


def send_tools():
    here = os.path.dirname(os.path.abspath(__file__))
    tar = subprocess.Popen(["tar", "-cf", "-", "-C", here, "exam_record.py", "world_of.py"], stdout=subprocess.PIPE)
    r = subprocess.run(SSH + [f"mkdir -p {LT} {LO} && tar -x -C {LT}"], stdin=tar.stdout, capture_output=True, text=True)
    tar.wait()
    assert r.returncode == 0, r.stderr


def run_lab(name, override, secs, n_env=256, local=None, log=print):
    """one exam on the lab: exam_record.py CK -> LO/name.npz (map, line3), MAP override JSON; returns (local npz or None, peak MB, s)"""
    env = dict(ENV, F1SIM_MAP_OVERRIDE=json.dumps(override))
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    out = f"{LO}/{name}.npz"; lg = f"{LO}/{name}.log"
    # the scope's memory.peak (the whole job, what MemoryMax limits) is printed at the end; the timeout inside, on python itself
    inner = (f"CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup); {exports} timeout {int(secs * 12 + 900)} ~/F1tenth/.venv/bin/python3 -u -W ignore "
             f"{LT}/exam_record.py {CK} {out} map {n_env} line3 {secs:g} 50; rc=$?; echo peak_mb $(( $(cat $CG/memory.peak) / 1048576 )); exit $rc")
    script = (f"#!/bin/bash\ncd {LC}\nsystemd-run --user --scope -p MemoryMax=10G -- bash -c {shlex.quote(inner)} "
              f"> {lg} 2>&1\necho rc $? >> {lg}\n")
    # (the script goes in on stdin; the lab's login shell is zsh -- plain names, no globs)
    rmv = f"python3 -c \"import os; [os.remove(p) for p in ('{out}', '{lg}') if os.path.exists(p)]\""
    r = subprocess.run(SSH + [f"mkdir -p {LO} && {rmv} && cat > {LO}/{name}.sh && (setsid nohup bash {LO}/{name}.sh > /dev/null 2>&1 < /dev/null &)"],
                       input=script, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        log(f"{name}: 시작 실패 {r.stderr.strip()[:200]}")
        return None, 0, 0
    t0 = time.time(); txt = ""
    while time.time() - t0 < secs * 12 + 1000:
        time.sleep(20)
        q = ssh(f"cat {lg} 2>/dev/null | tail -c 3000")
        if q is not None and q.returncode == 0:
            txt = q.stdout
            if re.search(r"^rc \d+", txt, re.M):
                break
    rc = re.findall(r"^rc (\d+)", txt, re.M)
    pk = re.findall(r"peak_mb (\d+)", txt)
    el = time.time() - t0
    if not rc or rc[-1] != "0":
        log(f"{name}: 실패 rc={rc} | {txt[-400:]}")
        return None, int(pk[-1]) if pk else 0, el
    local = local or os.path.join(O, f"{name}.npz")
    with open(local, "wb") as f:
        q = subprocess.run(SSH + [f"cat {out}"], stdout=f, timeout=600)
    if q.returncode != 0:
        return None, int(pk[-1]) if pk else 0, el
    ssh(f"python3 -c \"import os; os.remove('{out}')\"")
    return local, int(pk[-1]) if pk else 0, el


#: (11-24 16:40) a constraint, not the objective: c011 (the improving probes combined) had the smallest tracking error, 22.5 cm, and
#: hit walls 3.9 times a 30 s session (now 0.18) -- the error counts only driving decisions (plan speed > 2 m/s, no restart), so a car
#: that crashes and stands loses its worst moments from the score. Parents and the pick: hard wall + box hits per session at most
#: the current setting's + C_SLACK (Poisson noise of ~50 hits in 128 sessions: +-0.06)
C_SLACK = 0.15
#: (16:40) after the probes: the one that helped most without more contacts (speed lead 0.3 s) with the others that did not add
#: contacts, then the evolution around the two best feasible
SEEDS2 = [("s2_lead_gain1.2", dict(tw_speed_lead=0.3, v_steer_gain_max=1.2)),
          ("s2_lead_gain1.2_rate", dict(tw_speed_lead=0.3, v_steer_gain_max=1.2, steer_rate_limit=0.1)),
          ("s2_lead_euclid", dict(tw_speed_lead=0.3, l1_point="euclid")),
          ("s2_lead.25_gain1.3", dict(tw_speed_lead=0.25, v_steer_gain_max=1.3)),
          ("s2_lead_L1mid", dict(tw_speed_lead=0.3, t_clip_min=0.85, m_l1=0.42, q_l1=-0.4)),
          ("s2_lead.35", dict(tw_speed_lead=0.35))]


def contacts(d):
    return d.get("hard_wall", 9.9) + d.get("hard_prop", 9.9)


def search(until):
    """runs trials until the wall-clock time `until` (epoch s): no new trial starts after it"""
    os.makedirs(O, exist_ok=True)
    jl = os.path.join(O, "tune.jsonl")
    logf = open(os.path.join(O, "search.log"), "a")

    def log(s):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {s}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()
    done = [json.loads(l) for l in open(jl)] if os.path.exists(jl) else []
    W = None
    c0 = [d for d in done if d.get("tag") == "now" and "cell_n" in d]
    if c0:
        W = c0[0]["cell_n"]
    send_tools()

    def run(tag, cfg):
        nonlocal W
        i = 1 + max([d["i"] for d in done], default=-1)
        wait_free(log)
        if time.time() > until:
            return None
        name = f"c{i:03d}"
        fn, pk, el = run_lab(name, cfg, 30, log=log)
        d = dict(i=i, tag=tag, cfg=cfg, peak_mb=pk, secs_wall=round(el))
        if fn:
            d.update(score(fn, W))
            if tag == "now" and W is None:
                W = d["cell_n"]; d["obj"] = score(fn, W)["obj"]
        else:
            d["obj"] = 9.9
        done.append(d)
        with open(jl, "a") as f:
            f.write(json.dumps(d) + "\n")
        if fn:
            log(f"{name} {tag:<18} 오차 {d['obj'] * 100:5.2f} cm | 0.2/0.5 s 점 {d['pos2'] * 100:4.1f}/{d['pos5'] * 100:4.1f} 옆 {d['lat2'] * 100:4.1f}/{d['lat5'] * 100:4.1f} cm"
                f" | 흔들림 {d['wobble_solo']:.2f} 미끄럼90 {d['slip_p90']:.1f}도 | 빈트랙 바퀴 {d['lap_solo']:.2f} s 벽 {d['hard_wall']:.2f} 상자 {d['hard_prop']:.2f}"
                f" 속도차 {d['dv_med']:+.2f} | {pk} MB {el:.0f} s | {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in cfg.items() if cfg[k] != NOW.get(k)})}")
        else:
            log(f"{name} {tag} 실패")
        return d

    have = {d["tag"] for d in done}
    for tag, ch in PROBES:
        if tag not in have and run(tag, full(ch)) is None:
            break
    now_d = [d for d in done if d["tag"] == "now" and d["obj"] < 9][0]
    base, base_c = now_d["obj"], contacts(now_d)
    log(f"지금 설정 {base * 100:.2f} cm, 세게 닿음(벽+상자) {base_c:.2f}/세션 -> 후보는 {base_c + C_SLACK:.2f} 이하만")

    def feasible():
        return sorted([d for d in done if d["obj"] < 9 and contacts(d) <= base_c + C_SLACK], key=lambda d: d["obj"])
    for tag, ch in SEEDS2:
        if tag not in have and run(tag, full(ch)) is None:
            break
    g = 2 + sum(1 for d in done if d["tag"].startswith("e")) // 6
    while time.time() < until:
        ranked = feasible()
        sig = 0.15 * (0.75 ** (g - 2))
        kids = [(f"e{g}_m", mutate(ranked[0]["cfg"] if j < 4 else ranked[min(1, len(ranked) - 1)]["cfg"], sig)) for j in range(6)]
        for tag, cfg in kids:
            if run(tag, cfg) is None:
                break
        r = feasible()
        log(f"세대 {g} 끝: 닿음 조건 안에서 가장 좋은 c{r[0]['i']:03d} {r[0]['obj'] * 100:.2f} cm (지금 설정 {base * 100:.2f})")
        g += 1
    log("닿음 조건 안 상위 5:")
    for d in feasible()[:5]:
        log(f"  c{d['i']:03d} {d['tag']} 오차 {d['obj'] * 100:.2f} cm 흔들림 {d['wobble_solo']:.2f} 벽 {d['hard_wall']:.2f} 상자 {d['hard_prop']:.2f} | {json.dumps(d['cfg'])}")


if __name__ == "__main__":
    if sys.argv[1] == "search":
        hh, mm = (int(x) for x in sys.argv[2].split(":"))     # HH:MM today: no new trial after it
        lt = time.localtime()
        search(time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1)))
    elif sys.argv[1] == "one":
        name, cfg = sys.argv[2], json.loads(sys.argv[3])
        secs = float(sys.argv[4]) if len(sys.argv) > 4 else 90.0
        send_tools()
        wait_free(print)
        fn, pk, el = run_lab(name, cfg, secs, local=os.path.join(R, "out", "exam", f"{name}.npz"))
        print(json.dumps(dict(name=name, npz=fn, peak_mb=pk, secs_wall=round(el), cfg=cfg)), flush=True)
