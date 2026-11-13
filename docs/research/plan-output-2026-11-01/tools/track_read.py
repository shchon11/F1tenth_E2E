"""Read --track-steps records (11-14): per update, how far the actor went along each measured direction, and where it was lost.

    python3 track_read.py RUN_DIR [DIRECTION keys, default w c9 rand]
"""
import sys, glob, os
import numpy as np

run = sys.argv[1]; keys = sys.argv[2:] or ["w", "c9", "rand"]
fs = sorted(glob.glob(os.path.join(run, "track", "t*.npz")), key=lambda f: int(os.path.basename(f)[1:-4]))
for f in fs:
    z = np.load(f); u = int(os.path.basename(f)[1:-4]); lr = float(z["lr"])
    print(f"== update {u}: actor가 움직인 거리 {float(z['update_norm']):.3f} (걸음 {len(z['step_norm'])}번, 걸음 크기 중앙 {np.median(z['step_norm']):.4f}), 잘린 걸음 {np.mean(z['clip_coef'] < 1) * 100:.0f} % (자른 배율 중앙 {np.median(z['clip_coef']):.2f})")
    for k in keys:
        st = z[f"step_{k}"]; gr = z[f"grad_{k}"]
        cos_step = st / np.maximum(z["step_norm"], 1e-12); cos_grad = gr / np.maximum(z["grad_norm"], 1e-12)
        print(f"   방향 {k:>5}: update 전체로 간 거리 {float(z[f'update_{k}']):+.4f} | 걸음마다 간 거리 합 {st.sum():+.4f} (epoch별 {' '.join(f'{x:+.4f}' for x in st.reshape(-1, len(st) // 4).sum(1))}) | "
              f"걸음과의 cosine 평균 {cos_step.mean():+.4f} | 자른 gradient와의 cosine 평균 {cos_grad.mean():+.4f} | 같은 gradient로 SGD(lr {lr:.1e})였다면 간 거리 {lr * (gr).sum():+.5f}")
    terms = sorted({k_.split("_")[1] for k_ in z.files if k_.startswith("term_") and k_.endswith("_norm")})
    for t in terms:
        nm = z[f"term_{t}_norm"]; s = z[f"term_{t}_sum"].astype(np.float64)
        print(f"   손실 항 {t:>6}: minibatch gradient 크기 중앙 {np.median(nm):.4f}, 첫 epoch 합의 크기 {np.linalg.norm(s):.4f} | " +
              " · ".join(f"{k} 방향으로 민 크기 {z[f'term_{t}_{k}'].sum():+.5f} (cosine {z[f'term_{t}_{k}'].sum() / max(np.linalg.norm(s), 1e-12):+.4f})" for k in keys))
