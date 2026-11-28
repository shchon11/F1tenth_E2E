"""Error tables in the audit's format (D_sim_fidelity.md section 3.2) plus the spin rate."""
import numpy as np

from f1sim.bag_profile import models as M, windows as WN

WARM = WN.WARM


def subsets(W):
    n = len(W["r"])
    ay1 = np.abs(W["ay"][:, WARM:WARM + 101]).max(1)
    return {"all": np.arange(n), "ay>=7": np.flatnonzero(ay1 >= 7)}


def row(W, pr, idx):
    e = M.errors(W, {k: v[idx] for k, v in pr.items()}, idx)
    med = lambda a: float(np.nanmedian(a)); p90 = lambda a: float(np.nanpercentile(a, 90))
    real = np.degrees(np.abs(WN.real_sideslip_change(W)[idx]).max(1))
    spin = M.spin_flags({k: v[idx] for k, v in pr.items()})
    calm = real < 15.0
    return dict(n=int(len(idx)),
                yaw=[med(e[h]["yaw"]) for h in (50, 100, 200)], yaw1_p90=p90(e[100]["yaw"]),
                head1=med(e[100]["head"]), v=[med(e[h]["v"]) for h in (50, 100, 200)], ay1=med(e[100]["ay"]),
                pos=[med(e[h]["pos"]) for h in (50, 100, 200)], pos2_p90=p90(e[200]["pos"]),
                spin=float(spin.mean()), spin_calm=float(spin[calm].mean()) if calm.any() else float("nan"),
                n_calm=int(calm.sum()))


def table(W, preds, names=None):
    S = subsets(W)
    return {sub: {m: row(W, preds[m], idx) for m in (names or preds)} for sub, idx in S.items() if len(idx) >= 10}


def md(T, label):
    out = [f"**{label}** (n = {T['all'][next(iter(T['all']))]['n']}, 중앙값, [ ]는 90번째 백분위)", "",
           "| 모델 | 요레이트 0.5/1/2 s [rad/s] | 1 s 방향 [rad] | 속도 0.5/1/2 s [m/s] | 횡가속 1 s [m/s²] | 위치 0.5/1/2 s [m] | 스핀 비율 (전체 / 실차 안 미끄러짐) |",
           "|---|---|---|---|---|---|---|"]
    for m, r in T["all"].items():
        out.append(f"| {m} | {r['yaw'][0]:.3f} / **{r['yaw'][1]:.3f}** [{r['yaw1_p90']:.3f}] / {r['yaw'][2]:.3f} | {r['head1']:.3f} | "
                   f"{r['v'][0]:.2f} / {r['v'][1]:.2f} / {r['v'][2]:.2f} | {r['ay1']:.2f} | "
                   f"{r['pos'][0]:.2f} / {r['pos'][1]:.2f} / **{r['pos'][2]:.2f}** [{r['pos2_p90']:.2f}] | "
                   f"{100 * r['spin']:.1f} % / {100 * r['spin_calm']:.1f} % |")
    if "ay>=7" in T:
        n7 = T["ay>=7"][next(iter(T["ay>=7"]))]["n"]
        out += ["", f"한계 근처 (구간 첫 1 s 최대 |횡가속| ≥ 7 m/s², n = {n7}):", "",
                "| 모델 | 요레이트 1 s | 방향 1 s | 위치 2 s | 횡가속 1 s | 스핀 비율 |", "|---|---|---|---|---|---|"]
        for m, r in T["ay>=7"].items():
            out.append(f"| {m} | {r['yaw'][1]:.3f} [{r['yaw1_p90']:.3f}] | {r['head1']:.3f} | {r['pos'][2]:.2f} [{r['pos2_p90']:.2f}] | "
                       f"{r['ay1']:.2f} | {100 * r['spin']:.1f} % |")
    return "\n".join(out)
