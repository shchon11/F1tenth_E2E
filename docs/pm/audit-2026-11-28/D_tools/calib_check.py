import numpy as np, os, sys
import sig
for f in sig.files("all"):
    Z = np.load(f)
    nm = os.path.basename(f)[:40]
    m = sig.mapping(Z)
    S = sig.signals(Z, m["servo_g"], m["servo_o"], m["erpm_k"])
    # gyro vs pf heading rate (and vs odom wz)
    out = [nm, f"servo={m['servo_g']:.4f}*d+{m['servo_o']:.4f} ({m['servo_frac']:.3f})", f"erpm/mps={m['erpm_k']:.1f}"]
    mv = (S["v_wheel"] > 1.0)
    if "pyaw" in S:
        pr = sig.deriv(sig.smooth(S["pyaw"], 5), 5)
        ok = mv & np.isfinite(pr)
        gz = sig.smooth(S["gz"], 10)
        if ok.sum() > 200:
            k = np.sum(gz[ok] * pr[ok]) / np.sum(pr[ok] ** 2)
            cc = np.corrcoef(gz[ok], pr[ok])[0, 1]
            k_deg = np.sum(np.deg2rad(gz[ok]) * pr[ok]) / np.sum(pr[ok] ** 2)
            ko = np.sum(S["odom_wz"][ok] * pr[ok]) / np.sum(pr[ok] ** 2); co = np.corrcoef(S["odom_wz"][ok], pr[ok])[0,1]
            out.append(f"gz/pfrate={k:.3f} r={cc:.3f} (as deg/s: {k_deg:.3f}) odomwz/pfrate={ko:.3f} r={co:.3f}")
        # pf speed vs wheel speed
        vx = np.hypot(sig.deriv(S["px"], 5), sig.deriv(S["py"], 5))
        ok2 = mv & np.isfinite(vx)
        out.append(f"pf_v/wheel_v med={np.median(vx[ok2]/S['v_wheel'][ok2]):.3f}")
    out.append(f"gz p99={np.percentile(np.abs(S['gz']),99):.2f} az_med={np.median(S['az']):.2f} vmax={S['v_wheel'].max():.2f} moving_s={mv.sum()/100:.0f}")
    print(" | ".join(out), flush=True)
