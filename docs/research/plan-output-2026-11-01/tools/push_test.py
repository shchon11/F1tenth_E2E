"""천천히 밀기: 차를 상자 바로 앞에 두고 속도 명령 v_cmd로 3 s. 상자가 얼마나 움직이나 (실제: 정지 마찰 mu*m*g > 바퀴 힘이면 0)"""
import sys, math, torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim"); sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim/tests")
import test_contact_physics as T
for style in ("cardboard_box", "wooden_crate", "crate_stack_low", "barrier_block", "steel_drum"):
    for v_cmd in (1.0, 3.0):
        sim, po = T._prop_scene(style, B=2)
        st = T._touching_pose(sim, 15.0, 0.0, depth=0.001)
        st[0, 0] -= 0.05                                    # 5 cm short of the prop, at rest
        st[:, 3:] = 0.0
        sim.state.copy_(st) if hasattr(sim.state, "copy_") else None
        sim.state = st.clone() if not hasattr(sim.state, "copy_") else sim.state
        x0 = po.p_poses[0, 0, :2].clone()
        cmd = torch.zeros(sim.B, 2); cmd[0, 1] = v_cmd
        vmax = 0.0
        for k in range(120):                                 # 3 s at 40 Hz
            sim.step(cmd)
            vmax = max(vmax, float(sim.state[0, 3]))
        d = float((po.p_poses[0, 0, :2] - x0).norm())
        m = float(po.p_mass[0, 0])
        print(f"{style:16s} {m:5.1f} kg | 명령 {v_cmd:.0f} m/s | 3 s 뒤 상자 이동 {d:.2f} m, 차 속도 {float(sim.state[0, 3]):.2f} m/s (최고 {vmax:.2f}) | "
              f"정지 마찰 0.5mg = {0.5 * m * 9.81:.0f} N vs 차 최대 구동력 ~{float(sim.P['m'][0]) * 9.81:.0f} N")
