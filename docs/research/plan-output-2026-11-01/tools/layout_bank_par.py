"""상자 배치 bank 를 여러 프로세스로 나눠 만들기 (11-15, 사용자: "그 파이프라인들도 효율화 좀 해줘").

layout_bank.py 한 프로세스는 CPU 한 개를 100 % 쓰고 GPU 는 거의 놂 (Korea 2026 bank 17개 트랙에 약 70분).
트랙끼리는 서로 상관이 없으므로, 세계의 트랙(원본 + 뒤튼 것 + 크기 바꾼 것)을 N 개 프로세스에 나눠 주고
(각자 BANK_TRACKS = 자기 몫, BANK_SEED = 다른 시드), 끝나면 한 파일로 합침. 합친 파일은 layout_bank.py 한 번에
만든 것과 같은 형식 (per_track / C / judge) -> 학습·콘솔이 읽는 방식 그대로.

    python3 layout_bank_par.py <세계.json> <출력.pt> <트랙당배치수> [한번에판정할배치수 최적화반복]

  WORKERS=N      프로세스 수 (기본: 비어 있는 물리 코어 - 1, 최소 1; 비어 있는 = 물리 코어 수 - 1분 평균 부하)
  BANK_TRACKS    이 트랙들만 (기본: 세계의 트랙 전부)
  BANK_SHAPES    layout_bank.py 에 그대로 넘김
조각은 <출력 폴더>/shards/ 에 쓰고, 합친 뒤 지움 (bank 폴더의 *.pt 만 읽히므로 조각이 섞일 일은 없음).
"""
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
F1SIM = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "f1sim"))


def free_workers() -> int:
    """비어 있는 물리 코어 - 1 (최소 1). 이 컴퓨터는 8코어 16스레드: 하이퍼스레드 둘을 한 코어로 셈. 부하 = 1분 평균."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False) or (os.cpu_count() or 2) // 2
    except Exception:
        n = max(1, (os.cpu_count() or 2) // 2)
    busy = math.ceil(os.getloadavg()[0] - 0.25)
    return max(1, n - max(busy, 0) - 1)


def world_tracks(world: str):
    """state_rl.build_env 와 같은 트랙 목록 (원본마다 + warp0..warpN-1 또는 hose0..hoseN-1, 그다음 크기 바꾼 것)."""
    sys.path.insert(0, F1SIM)
    from f1sim.learn import state_rl as SR
    a = json.load(open(world))
    bases = [t.strip() for t in str(a.get("tracks") or SR.KOREA).split(",") if t.strip()]
    warps = int(a.get("warps") or 0)
    vk = "+hose" if str(a.get("variant") or "warp") == "hose" else "+warp"      # (11-23) the world's variant kind, as build_env
    names = [x for b in bases for x in [b] + [f"{b}{vk}{k}" for k in range(warps)]]
    return names + list(SR.scale_variants(bases, warps, a.get("track_scales")))


def merge(shards, out):
    import torch
    per, C, judge = {}, None, None
    for f in shards:
        d = torch.load(f, map_location="cpu", weights_only=False)
        if C is not None and d["C"] != C:
            raise SystemExit(f"조각마다 슬롯 수가 다름: {C} vs {d['C']} ({f})")
        C, judge = d["C"], judge or d["judge"]
        dup = set(per) & set(d["per_track"])
        if dup:
            raise SystemExit(f"같은 트랙이 두 조각에 있음: {sorted(dup)}")
        per.update(d["per_track"])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(dict(per_track=per, C=C, judge=judge), out)
    return per


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    world, out, n_per = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3])
    extra = sys.argv[4:6]
    names = [t.strip() for t in os.environ["BANK_TRACKS"].split(",") if t.strip()] if os.environ.get("BANK_TRACKS") \
        else world_tracks(world)
    n = max(1, min(int(os.environ.get("WORKERS") or free_workers()), len(names)))
    groups = [names[i::n] for i in range(n)]
    sd = os.path.join(os.path.dirname(out), "shards")
    os.makedirs(sd, exist_ok=True)
    stem = os.path.splitext(os.path.basename(out))[0]
    seed0 = int(os.environ.get("BANK_SEED") or int(time.time()) % 100000)
    print(f"== bank {out}: 트랙 {len(names)}개를 프로세스 {n}개로 (트랙당 {n_per}개), 조각 {sd}", flush=True)
    procs = []
    t0 = time.time()
    for i, g in enumerate(groups):
        env = dict(os.environ, BANK_WORLD=world, BANK_TRACKS=",".join(g), BANK_SEED=str(seed0 + 7919 * i),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", F1SIM_RACELINE_CACHE_ONLY="1",
                   PYTHONPATH=F1SIM + os.pathsep + os.environ.get("PYTHONPATH", ""))
        shard = os.path.join(sd, f"{stem}_{i}.pt")
        log = open(os.path.join(sd, f"{stem}_{i}.log"), "w")
        p = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "layout_bank.py"), world, shard, str(n_per)] + extra,
                             env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, shard, log, g))
        print(f"  조각 {i}: 트랙 {len(g)}개 ({', '.join(g[:3])}{' ...' if len(g) > 3 else ''}) pid {p.pid}", flush=True)
    last = time.time()
    while any(p.poll() is None for p, *_ in procs):
        time.sleep(5)
        if time.time() - last > 300:
            last = time.time()
            for i, (p, shard, log, g) in enumerate(procs):
                try:
                    tail = [l for l in open(log.name).read().splitlines() if l.strip()][-1:]
                except OSError:
                    tail = []
                print(f"  [{(time.time() - t0) / 60:5.1f} 분] 조각 {i}: {'끝' if p.poll() is not None else '진행'} {tail[0][:140] if tail else ''}",
                      flush=True)
    bad = [(i, p.returncode) for i, (p, *_ ) in enumerate(procs) if p.returncode != 0]
    if bad:
        raise SystemExit(f"실패한 조각 {bad}: 기록은 {sd}/{stem}_<번호>.log")
    shards = [s for _, s, _, _ in procs]
    per = merge(shards, out)
    for p, shard, log, g in procs:
        log.close()
        for f in (shard, log.name):
            try:
                os.remove(f)
            except OSError:
                pass
    try:
        os.rmdir(sd)
    except OSError:
        pass
    k = [int(v["sid"].shape[0]) for v in per.values()]
    print(f"== 끝 {(time.time() - t0) / 60:.1f} 분: 트랙 {len(per)}개, 트랙당 배치 {min(k)}-{max(k)}개 -> {out}", flush=True)


if __name__ == "__main__":
    main()
