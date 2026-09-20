from pathlib import Path
p=Path('f1sim/f1sim/mpc.py');s=p.read_text()
s=s.replace('v_limit: Optional[torch.Tensor] = None, a_walk: Optional[torch.Tensor] = None):','v_limit: Optional[torch.Tensor] = None, a_walk: Optional[torch.Tensor] = None,\n              v_profile: Optional[torch.Tensor] = None):',1)
s=s.replace('    Both default to None, which is the original behaviour exactly.','    ``v_profile`` replaces the speed target (including its measured initial state).\n    ``a_walk`` also accepts local spatial budgets (B,n,2), interpolated at the walked arc.\n    Optional inputs default to None, preserving the original behaviour exactly.',1)
s=s.replace('    a_brk_w = (-spec.a_brake * spec.dt) if a_walk is None else (-a_walk[:, 0] * spec.dt)\n    a_acc_w = (spec.a_max * spec.dt) if a_walk is None else (a_walk[:, 1] * spec.dt)','    local_walk = a_walk is not None and a_walk.ndim == 3\n    a_brk_w = (-spec.a_brake * spec.dt) if a_walk is None else (-a_walk[..., 0] * spec.dt)\n    a_acc_w = (spec.a_max * spec.dt) if a_walk is None else (a_walk[..., 1] * spec.dt)')
s=s.replace('        if v_limit is not None:\n            # the envelope','        if v_profile is not None:\n            pos = frac * (v_profile.shape[1] - 1)\n            ip = pos.floor().clamp(max=v_profile.shape[1] - 2).long()\n            wp = pos - ip.to(v.dtype)\n            v[:, t_] = v_profile.gather(1, ip[:, None])[:, 0] * (1 - wp) + v_profile.gather(1, ip[:, None] + 1)[:, 0] * wp\n        if v_limit is not None:\n            # the envelope',1)
s=s.replace('        if t_ < T - 1:\n            dv =', '        if t_ < T - 1:\n            if local_walk:\n                pos = frac * (a_walk.shape[1] - 1)\n                iw = pos.floor().clamp(max=a_walk.shape[1] - 2).long()\n                ww = pos - iw.to(v.dtype)\n                bw = a_walk.gather(1, iw[:, None, None].expand(-1, 1, 2))[:, 0] * (1 - ww[:, None]) + a_walk.gather(1, (iw + 1)[:, None, None].expand(-1, 1, 2))[:, 0] * ww[:, None]\n                a_brk_w, a_acc_w = -bw[:, 0] * spec.dt, bw[:, 1] * spec.dt\n            dv =',1)
s=s.replace('consts=None, bounds: Optional[torch.Tensor] = None):','consts=None, bounds: Optional[torch.Tensor] = None, projector=None):',1)
s=s.replace('        lo, hi = bounds[:, 0], bounds[:, 1]','        lo, hi = (bounds[:, :, 0], bounds[:, :, 1]) if bounds.ndim == 4 else (bounds[:, 0], bounds[:, 1])',1)
s=s.replace('    def rollout(u):\n        z = [z0]\n        for t in range(N):\n            z.append(_dyn(z[-1], u[:, t], spec, wb))\n        return torch.stack(z, 1)\n\n    u = u_warm.clone(); z = rollout(u)', '''    def project(ut, state, t):
        if projector is not None:
            return projector(state, ut, t)
        lt, ht = (lo[:, t], hi[:, t]) if bounds is not None and bounds.ndim == 4 else (lo, hi)
        ut = torch.maximum(torch.minimum(ut, ht), lt)
        ut[:, 1] = torch.minimum(ut[:, 1], (v_max - state[:, 3]) / spec.dt)
        if bounds is not None:
            ut[:, 1] = torch.maximum(ut[:, 1], lt[:, 1])
        return ut

    def rollout(u):
        z = [z0]
        controls = []
        for t in range(N):
            # Historical warm starts stay unchanged. New local/stage constraints apply before
            # linearization, so the backward pass never linearizes an unchecked warm trajectory.
            ut = project(u[:, t], z[-1], t) if projector is not None or (bounds is not None and bounds.ndim == 4) else u[:, t]
            controls.append(ut)
            z.append(_dyn(z[-1], ut, spec, wb))
        return torch.stack(controls, 1), torch.stack(z, 1)

    u, z = rollout(u_warm.clone())''')
start=s.index('            ut = torch.maximum(torch.minimum(ut, hi), lo)',s.index('        # forward pass with clamped inputs'))
end=s.index('            un[:, t] = ut;',start)
s=s[:start]+'            ut = project(ut, zn, t)\n'+s[end:]
s=s.replace('        self._solver = None','        self._solver = None\n        # Optional nominal actuator conversion, installed only by the local automatic controller.\n        self._command_hook = None',1)
s=s.replace('        k_ = max(1, int(round(sp.v_cmd_lead / sp.dt)))','        if self._command_hook is not None:\n            return self._command_hook(u, z, v_meas, speed_cap)\n        k_ = max(1, int(round(sp.v_cmd_lead / sp.dt)))',1)
p.write_text(s)
