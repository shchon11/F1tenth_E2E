/* 목업용 정지 장면 생성기 (라이브러리 없음).
   - 트랙(덕트 호스 벽) 한 개를 수식으로 만들고, 캔버스 2D에 원근 투영 + 깊이 정렬로 그린다.
   - 실제 제품의 WebGL 장면이 따라야 할 "조명·재질·카메라"의 견본이지 구현 코드가 아니다. */
(function () {
  'use strict';
  const TAU = Math.PI * 2;
  const V = {
    add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    mul: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
    dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
    len: (a) => Math.hypot(a[0], a[1], a[2]),
    norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
  };
  function rng(seed) { let t = seed >>> 0; return () => { t += 0x6D2B79F5; let r = Math.imul(t ^ (t >>> 15), 1 | t); r ^= r + Math.imul(r ^ (r >>> 7), 61 | r); return ((r ^ (r >>> 14)) >>> 0) / 4294967296; }; }
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const mix = (a, b, t) => a + (b - a) * t;
  const hex = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  const rgb = (c, a) => a === undefined ? `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})` : `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;

  /* ---------------- 트랙: 2026 한국 대회장을 닮은 실내 코스 (길이 약 54 m) ---------------- */
  const CTRL = [
    [3.2, 1.4], [8.0, 1.3], [12.8, 1.4], [15.5, 2.3], [16.2, 4.4], [14.6, 6.0], [12.0, 5.3], [10.2, 4.0],
    [8.2, 4.1], [6.9, 5.6], [7.9, 7.5], [10.6, 8.0], [13.4, 8.4], [15.6, 9.7], [14.8, 11.6], [11.6, 11.9],
    [8.2, 10.8], [5.2, 11.2], [2.5, 10.4], [1.4, 8.2], [2.5, 6.2], [2.3, 4.3], [1.6, 2.8],
  ].map(p => [p[0] * 0.75, p[1] * 0.75]);
  function buildTrack() {
    const n = CTRL.length, raw = [];
    for (let i = 0; i < n; i++) {
      const p0 = CTRL[(i + n - 1) % n], p1 = CTRL[i], p2 = CTRL[(i + 1) % n], p3 = CTRL[(i + 2) % n];
      for (let k = 0; k < 40; k++) {
        const t = k / 40, t2 = t * t, t3 = t2 * t;
        raw.push([0, 1].map(d => 0.5 * ((2 * p1[d]) + (-p0[d] + p2[d]) * t + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2 + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3)));
      }
    }
    // 같은 호 길이로 다시 뽑기
    const cum = [0];
    for (let i = 1; i <= raw.length; i++) { const a = raw[i - 1], b = raw[i % raw.length]; cum.push(cum[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1])); }
    const L = cum[cum.length - 1], ds = 0.1, N = Math.round(L / ds), c = [];
    let j = 0;
    for (let i = 0; i < N; i++) {
      const s = i * L / N; while (cum[j + 1] < s) j++;
      const t = (s - cum[j]) / (cum[j + 1] - cum[j]), a = raw[j], b = raw[(j + 1) % raw.length];
      c.push([mix(a[0], b[0], t), mix(a[1], b[1], t)]);
    }
    const tan = [], nor = [], hw = [], kap = [];
    for (let i = 0; i < N; i++) {
      const a = c[(i + N - 2) % N], b = c[(i + 2) % N]; const dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy);
      tan.push([dx / l, dy / l]); nor.push([-dy / l, dx / l]);
      hw.push(1.0 + 0.12 * Math.sin(TAU * i / N * 3 + 0.8) + 0.04 * Math.sin(TAU * i / N * 7));
    }
    for (let i = 0; i < N; i++) {
      const a = tan[(i + N - 3) % N], b = tan[(i + 3) % N];
      kap.push(Math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / (6 * L / N));
    }
    const wall = (sgn) => {
      const w = [];
      for (let i = 0; i < N; i++) w.push([c[i][0] + sgn * nor[i][0] * hw[i], c[i][1] + sgn * nor[i][1] * hw[i], i]);
      // 안쪽 급커브에서 접힌 점 제거
      let out = w;
      for (let pass = 0; pass < 16; pass++) {
        const keep = [];
        for (let k = 0; k < out.length; k++) {
          const p = out[k], q = out[(k + 1) % out.length], t = tan[p[2]];
          if ((q[0] - p[0]) * t[0] + (q[1] - p[1]) * t[1] > 0.02) keep.push(p);
        }
        out = keep;
      }
      for (let it = 0; it < 3; it++) { const m = out.length; out = out.map((p, k) => { const a = out[(k + m - 1) % m], b = out[(k + 1) % m]; return [(a[0] + 2 * p[0] + b[0]) / 4, (a[1] + 2 * p[1] + b[1]) / 4, p[2]]; }); }
      return out;
    };
    return { c, tan, nor, hw, kap, N, L, ds: L / N, left: wall(1), right: wall(-1) };
  }
  const track = buildTrack();
  // 시험 장애물(상자 3개): [진행 위치 비율, 중심선에서 옆으로 m, 회전]
  const BOXES = [[0.105, 0.42, 0.3], [0.47, -0.5, 1.0], [0.80, 0.35, 0.2]];
  function at(frac, lat) {
    const i = ((Math.round(frac * track.N) % track.N) + track.N) % track.N, c = track.c[i], n = track.nor[i], t = track.tan[i];
    return { i, x: c[0] + n[0] * (lat || 0), y: c[1] + n[1] * (lat || 0), yaw: Math.atan2(t[1], t[0]) };
  }
  const boxes = BOXES.map(b => { const p = at(b[0], b[1]); return { x: p.x, y: p.y, yaw: p.yaw + b[2], sx: 0.34, sy: 0.30, sz: 0.30, i: p.i }; });

  /* 드라이버가 피해 가는 옆 위치(중심선 기준 m): 상자 반대쪽으로 부드럽게 */
  function lineOffset0(i) {
    let off = -0.55 * clamp(track.kap[i] * 1.6, -1, 1) * 0.6; // 코너 안쪽으로 조금
    for (const b of BOXES) {
      const bi = Math.round(b[0] * track.N); let d = (i - bi) * track.ds; if (d > track.L / 2) d -= track.L; if (d < -track.L / 2) d += track.L;
      off += -Math.sign(b[1]) * 0.62 * Math.exp(-(d * d) / (2 * 1.5 * 1.5));
    }
    return clamp(off, -0.7, 0.7);
  }
  const OFF = []; for (let i = 0; i < track.N; i++) { let a = 0, w = 0; for (let d = -9; d <= 9; d++) { const g = 10 - Math.abs(d); a += g * lineOffset0(((i + d) % track.N + track.N) % track.N); w += g; } OFF.push(a / w); }
  function lineOffset(i) { return OFF[((i % track.N) + track.N) % track.N]; }
  function linePoint(i) { i = ((i % track.N) + track.N) % track.N; const o = lineOffset(i), c = track.c[i], n = track.nor[i]; return [c[0] + n[0] * o, c[1] + n[1] * o]; }
  function speedAt(i) { i = ((i % track.N) + track.N) % track.N; let k = 0; for (let d = 0; d < 22; d++) k = Math.max(k, Math.abs(track.kap[(i + d) % track.N]) * (1 - d / 40)); return clamp(Math.sqrt(9.5 / Math.max(k, 0.1)) * 1.16, 3.3, 7.4); }

  /* ---------------- LiDAR 광선 ---------------- */
  function segments() {
    const segs = [];
    [track.left, track.right].forEach((w, wi) => { const sg = wi ? 0.15 : -0.15, q = w.map(p => { const n = track.nor[p[2]]; return [p[0] + n[0] * sg, p[1] + n[1] * sg]; }); for (let k = 0; k < q.length; k += 2) { const a = q[k], b = q[(k + 2) % q.length]; segs.push([a[0], a[1], b[0], b[1], 0]); } });
    for (const b of boxes) {
      const c = Math.cos(b.yaw), s = Math.sin(b.yaw), hx = b.sx / 2, hy = b.sy / 2;
      const P = [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]].map(q => [b.x + q[0] * c - q[1] * s, b.y + q[0] * s + q[1] * c]);
      for (let k = 0; k < 4; k++) segs.push([P[k][0], P[k][1], P[(k + 1) % 4][0], P[(k + 1) % 4][1], 1]);
    }
    return segs;
  }
  function scan(pose, opt) {
    opt = Object.assign({ n: 1081, fov: 270, rmax: 10, noise: 0.012, seed: 7 }, opt || {});
    const segs = segments(), R = rng(opt.seed), out = [];
    for (let k = 0; k < opt.n; k++) {
      const ang = (-opt.fov / 2 + opt.fov * k / (opt.n - 1)) * Math.PI / 180, th = pose.yaw + ang, dx = Math.cos(th), dy = Math.sin(th);
      let best = opt.rmax, kind = -1;
      for (const s of segs) {
        const ex = s[2] - s[0], ey = s[3] - s[1], den = dx * ey - dy * ex; if (Math.abs(den) < 1e-9) continue;
        const ox = s[0] - pose.x, oy = s[1] - pose.y, t = (ox * ey - oy * ex) / den, u = (ox * dy - oy * dx) / den;
        if (t > 0.03 && t < best && u >= 0 && u <= 1) { best = t; kind = s[4]; }
      }
      const r = kind < 0 ? opt.rmax : best + (R() - 0.5) * 2 * opt.noise * 1.7;
      out.push({ ang, r, hit: kind >= 0, kind, x: pose.x + dx * r, y: pose.y + dy * r });
    }
    return out;
  }

  /* ---------------- 작은 3D 그리개 ---------------- */
  function Scene(canvas, cam, env) {
    const W = canvas.width, H = canvas.height, ctx = canvas.getContext('2d');
    const f = V.norm(V.sub(cam.target, cam.pos)), r = V.norm(V.cross(f, [0, 0, 1])), u = V.cross(r, f);
    const focal = (H / 2) / Math.tan((cam.fov || 50) * Math.PI / 360), ox = cam.ox || 0, oy = cam.oy || 0, NEAR = 0.06;
    const L = V.norm(env.light || [-0.45, -0.35, 0.82]), amb = env.amb === undefined ? 0.38 : env.amb;
    const fogC = env.fog || [12, 13, 14], f0 = env.f0 || 8, f1 = env.f1 || 34;
    const items = [];
    const toCam = (p) => { const d = V.sub(p, cam.pos); return [V.dot(d, r), V.dot(d, u), V.dot(d, f)]; };
    const proj = (c) => [W / 2 + ox + c[0] * focal / c[2], H / 2 + oy - c[1] * focal / c[2]];
    function clip(poly) {
      const out = [];
      for (let i = 0; i < poly.length; i++) {
        const a = poly[i], b = poly[(i + 1) % poly.length], ia = a[2] > NEAR, ib = b[2] > NEAR;
        if (ia) out.push(a);
        if (ia !== ib) { const t = (NEAR - a[2]) / (b[2] - a[2]); out.push([mix(a[0], b[0], t), mix(a[1], b[1], t), NEAR]); }
      }
      return out;
    }
    const fogT = (d) => clamp((d - f0) / (f1 - f0), 0, 1);
    const S = {
      W, H, ctx, cam, proj: (p) => { const c = toCam(p); return c[2] > NEAR ? proj(c).concat(c[2]) : null; },
      poly(pts, col, o) {
        o = o || {};
        let cp = pts.map(toCam); if (cp.every(c => c[2] <= NEAR) || cp.every(c => c[2] > (o.far || 60))) return;
        let nrm = o.n; if (!nrm) nrm = V.norm(V.cross(V.sub(pts[1], pts[0]), V.sub(pts[2], pts[0])));
        const cen = pts.reduce((a, p) => V.add(a, p), [0, 0, 0]).map(x => x / pts.length);
        const facing = V.dot(nrm, V.sub(cam.pos, cen));
        if (o.cull && facing <= 0) return;
        if (!o.n && facing < 0) nrm = V.mul(nrm, -1);
        cp = clip(cp); if (cp.length < 3) return;
        const depth = cp.reduce((a, c) => a + c[2], 0) / cp.length + (o.bias || 0);
        let c = col;
        if (!o.flat) { const sh = amb + (1 - amb) * Math.max(0, V.dot(nrm, L)); c = [col[0] * sh, col[1] * sh, col[2] * sh]; }
        if (o.mul) c = c.map(x => x * o.mul);
        if (!o.nofog) { const t = fogT(depth); c = [mix(c[0], fogC[0], t), mix(c[1], fogC[1], t), mix(c[2], fogC[2], t)]; }
        items.push({ k: 0, d: depth, p: cp.map(proj), c, a: o.a === undefined ? 1 : o.a, stroke: o.stroke, lw: o.lw, glow: o.glow, sa: o.sa });
      },
      line(a, b, col, o) {
        o = o || {}; let ca = toCam(a), cb = toCam(b);
        if (ca[2] <= NEAR && cb[2] <= NEAR) return;
        if (ca[2] <= NEAR) { const t = (NEAR - ca[2]) / (cb[2] - ca[2]); ca = [mix(ca[0], cb[0], t), mix(ca[1], cb[1], t), NEAR]; }
        if (cb[2] <= NEAR) { const t = (NEAR - cb[2]) / (ca[2] - cb[2]); cb = [mix(cb[0], ca[0], t), mix(cb[1], ca[1], t), NEAR]; }
        const depth = (ca[2] + cb[2]) / 2 + (o.bias || 0); let al = o.a === undefined ? 1 : o.a; if (!o.nofog) al *= 1 - fogT(depth);
        items.push({ k: 1, d: depth, p: [proj(ca), proj(cb)], c: col, a: al, lw: o.lw || 1, dash: o.dash, glow: o.glow });
      },
      dot(p, col, o) {
        o = o || {}; const c = toCam(p); if (c[2] <= NEAR) return;
        const depth = c[2] + (o.bias || 0); let al = (o.a === undefined ? 1 : o.a) * (1 - fogT(depth) * 0.8);
        items.push({ k: 2, d: depth, p: proj(c), c: col, a: al, r: o.px ? o.px : Math.min(o.max || 99, Math.max(o.min || 0.8, (o.r || 0.02) * focal / c[2])), glow: o.glow });
      },
      /* 상자: 로컬 좌표(pose 기준) 중심 c, 크기 s */
      box(pose, c, s, col, o) {
        o = Object.assign({ cull: true }, o || {});
        const cy = Math.cos(pose.yaw + (o.yaw || 0)), sy = Math.sin(pose.yaw + (o.yaw || 0)), pc = Math.cos(pose.yaw), ps = Math.sin(pose.yaw);
        const wc = [pose.x + c[0] * pc - c[1] * ps, pose.y + c[0] * ps + c[1] * pc, (pose.z || 0) + c[2]];
        const P = (x, y, z) => [wc[0] + x * cy - y * sy, wc[1] + x * sy + y * cy, wc[2] + z];
        const hx = s[0] / 2, hy = s[1] / 2, hz = s[2] / 2;
        const v = [P(-hx, -hy, -hz), P(hx, -hy, -hz), P(hx, hy, -hz), P(-hx, hy, -hz), P(-hx, -hy, hz), P(hx, -hy, hz), P(hx, hy, hz), P(-hx, hy, hz)];
        const F = [[3, 2, 1, 0], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [3, 0, 4, 7]];
        for (const q of F) S.poly(q.map(i => v[i]), col, o);
      },
      /* 원기둥: axis 'y'(바퀴) 또는 'z' */
      cyl(pose, c, rad, len, axis, col, o) {
        o = Object.assign({ cull: true, seg: 18 }, o || {});
        const yaw = pose.yaw + (o.yaw || 0), cy = Math.cos(yaw), sy = Math.sin(yaw), pc = Math.cos(pose.yaw), ps = Math.sin(pose.yaw);
        const wc = [pose.x + c[0] * pc - c[1] * ps, pose.y + c[0] * ps + c[1] * pc, (pose.z || 0) + c[2]];
        const P = (x, y, z) => [wc[0] + x * cy - y * sy, wc[1] + x * sy + y * cy, wc[2] + z];
        const A = [], B = [], n = o.seg;
        for (let k = 0; k < n; k++) {
          const t = TAU * k / n, a = Math.cos(t) * rad, b = Math.sin(t) * rad;
          if (axis === 'y') { A.push(P(a, -len / 2, b)); B.push(P(a, len / 2, b)); } else { A.push(P(a, b, -len / 2)); B.push(P(a, b, len / 2)); }
        }
        for (let k = 0; k < n; k++) { const k2 = (k + 1) % n; S.poly(axis === 'y' ? [A[k2], A[k], B[k], B[k2]] : [A[k], A[k2], B[k2], B[k]], col, o); }
        const capO = Object.assign({}, o, { mul: (o.mul || 1) * (o.capMul || 1) });
        S.poly(axis === 'y' ? A.slice() : A.slice().reverse(), o.cap || col, capO);
        S.poly(axis === 'y' ? B.slice().reverse() : B.slice(), o.cap || col, capO);
      },
      /* 덕트 호스: 바닥에 놓인 주름 관 */
      tube(path, rad, col, o) {
        o = o || {}; const K = 8, n = path.length, T0 = -0.55, T1 = Math.PI + 0.55, RB = o.ribs || 3;
        for (let i = 0; i < n; i++) {
          const a = path[i], b = path[(i + 1) % n];
          const dxa = a[0] - cam.pos[0], dya = a[1] - cam.pos[1], d2 = dxa * dxa + dya * dya; if (d2 > (o.far || 42) ** 2) continue;
          const pa = path[(i + n - 1) % n], pb = path[(i + 2) % n];
          const na = nrm2(b[0] - pa[0], b[1] - pa[1]), nb = nrm2(pb[0] - a[0], pb[1] - a[1]);
          const rb = d2 < 36 ? RB : 1;
          for (let m = 0; m < rb; m++) {
            const u0 = m / rb, u1 = (m + 1) / rb, A = [mix(a[0], b[0], u0), mix(a[1], b[1], u0)], Bq = [mix(a[0], b[0], u1), mix(a[1], b[1], u1)];
            const nA = [mix(na[0], nb[0], u0), mix(na[1], nb[1], u0)], nB = [mix(na[0], nb[0], u1), mix(na[1], nb[1], u1)];
            const rib = rb === 1 ? 0.93 : (((i * rb + m) % 2) ? 0.84 : 1.0);
            for (let k = 0; k < K; k++) {
              const t0 = mix(T0, T1, k / K), t1 = mix(T0, T1, (k + 1) / K), tm = (t0 + t1) / 2;
              const q = (p, nn, t) => [p[0] + nn[0] * rad * Math.cos(t), p[1] + nn[1] * rad * Math.cos(t), rad + rad * Math.sin(t)];
              S.poly([q(A, nA, t0), q(Bq, nB, t0), q(Bq, nB, t1), q(A, nA, t1)], col, { n: V.norm([nA[0] * Math.cos(tm), nA[1] * Math.cos(tm), Math.sin(tm)]), mul: rib, stroke: true });
            }
          }
        }
      },
      ground(o) {
        o = Object.assign({ top: [9, 10, 11], near: [30, 33, 35], far: [14, 16, 17], grid: 1, gridA: 0.11, ext: 40 }, o || {});
        const fwd = V.norm([f[0], f[1], 0]); const hz = S.proj([cam.pos[0] + fwd[0] * 4000, cam.pos[1] + fwd[1] * 4000, 0]); const hy = hz ? hz[1] : -50;
        const g0 = ctx.createLinearGradient(0, 0, 0, Math.max(1, hy)); g0.addColorStop(0, rgb(o.top)); g0.addColorStop(1, rgb(o.far));
        ctx.fillStyle = g0; ctx.fillRect(0, 0, W, Math.max(0, hy) + 1);
        const g = ctx.createLinearGradient(0, hy, 0, H); g.addColorStop(0, rgb(o.far)); g.addColorStop(0.18, rgb([mix(o.far[0], o.near[0], .45), mix(o.far[1], o.near[1], .45), mix(o.far[2], o.near[2], .45)])); g.addColorStop(1, rgb(o.near));
        ctx.fillStyle = g; ctx.fillRect(0, Math.max(0, hy), W, H - Math.max(0, hy));
        if (!o.grid) return;
        const cx = Math.round(cam.pos[0]), cyy = Math.round(cam.pos[1]), E = o.ext, st = 2;
        for (let gI = -E; gI <= E; gI += o.grid) for (let t = -E; t < E; t += st) {
          const major = Math.abs(gI % 5) < 1e-6 ? 1.6 : 1;
          S.line([cx + gI, cyy + t, 0], [cx + gI, cyy + t + st, 0], [200, 210, 205], { a: o.gridA * major, lw: 1, bias: 500 });
          S.line([cx + t, cyy + gI, 0], [cx + t + st, cyy + gI, 0], [200, 210, 205], { a: o.gridA * major, lw: 1, bias: 500 });
        }
      },
      flush() {
        items.sort((a, b) => b.d - a.d);
        for (const it of items) {
          ctx.globalAlpha = it.a; ctx.shadowBlur = 0;
          if (it.glow) { ctx.shadowColor = rgb(it.c, 0.9); ctx.shadowBlur = it.glow; }
          if (it.k === 0) {
            ctx.beginPath(); ctx.moveTo(it.p[0][0], it.p[0][1]); for (let i = 1; i < it.p.length; i++) ctx.lineTo(it.p[i][0], it.p[i][1]); ctx.closePath();
            ctx.fillStyle = rgb(it.c); ctx.fill();
            if (it.stroke === true) { ctx.strokeStyle = rgb(it.c); ctx.lineWidth = 0.7; ctx.stroke(); }
            else if (it.stroke) { ctx.globalAlpha = it.sa === undefined ? it.a : it.sa; ctx.strokeStyle = it.stroke; ctx.lineWidth = it.lw || 1; ctx.lineJoin = 'round'; ctx.stroke(); }
          } else if (it.k === 1) {
            ctx.beginPath(); ctx.moveTo(it.p[0][0], it.p[0][1]); ctx.lineTo(it.p[1][0], it.p[1][1]);
            ctx.setLineDash(it.dash || []); ctx.strokeStyle = rgb(it.c); ctx.lineWidth = it.lw; ctx.lineCap = 'round'; ctx.stroke(); ctx.setLineDash([]);
          } else {
            ctx.beginPath(); ctx.arc(it.p[0], it.p[1], it.r, 0, TAU); ctx.fillStyle = rgb(it.c); ctx.fill();
          }
        }
        ctx.globalAlpha = 1; ctx.shadowBlur = 0; items.length = 0;
      },
    };
    function nrm2(x, y) { const l = Math.hypot(x, y) || 1; return [-y / l, x / l]; }
    return S;
  }

  /* ---------------- 1/10 차 모형 (뒤축 중심이 원점, x 앞, y 왼쪽, 단위 m) ---------------- */
  function car(S, pose, o) {
    o = Object.assign({ steer: 0, style: 'race', hi: null }, o || {});
    const clay = o.style === 'clay';
    const E = clay ? { stroke: 'rgba(240,244,240,0.28)', lw: 0.8 } : { stroke: 'rgba(0,0,0,0.5)', lw: 0.7 };
    const C = clay
      ? { tire: [44, 47, 50], hub: [150, 156, 158], plate: [70, 75, 79], dark: [52, 56, 60], deck: [112, 118, 122], comp: [92, 98, 102], foam: [40, 43, 46], fin: [128, 134, 138], lidar: [60, 64, 68], cap: [198, 241, 53], batt: [84, 90, 94] }
      : { tire: [40, 42, 45], hub: [210, 215, 217], plate: [112, 119, 124], dark: [84, 90, 96], deck: [236, 239, 235], comp: [128, 136, 142], foam: [60, 64, 68], fin: [184, 191, 195], lidar: [66, 70, 74], cap: [198, 241, 53], batt: [96, 104, 112] };
    const hi = (name, col) => (o.hi === name ? [198, 241, 53] : col);
    const B = (c, s, col, x) => S.box(pose, c, s, col, Object.assign({}, E, x || {}));
    // 그림자
    for (let k = 0; k < 4; k++) { const g = 1 + k * 0.05; const P = [[-0.14, -0.17], [0.49, -0.17], [0.49, 0.17], [-0.14, 0.17]].map(q => { const x = (q[0] - 0.17) * g + 0.17, y = q[1] * g; return [pose.x + x * Math.cos(pose.yaw) - y * Math.sin(pose.yaw), pose.y + x * Math.sin(pose.yaw) + y * Math.cos(pose.yaw), 0.001]; }); S.poly(P, [0, 0, 0], { flat: true, a: 0.2, bias: 200, nofog: true }); }
    // 바퀴
    for (const w of [[0, 0.13, 0], [0, -0.13, 0], [0.33, 0.13, o.steer], [0.33, -0.13, o.steer]]) {
      S.cyl(pose, [w[0], w[1], 0.055], 0.055, 0.05, 'y', C.tire, Object.assign({ yaw: w[2], seg: 20, capMul: 0.9 }, E));
      S.cyl(pose, [w[0], w[1] + Math.sign(w[1]) * 0.022, 0.055], 0.03, 0.012, 'y', C.hub, { yaw: 0, seg: 12, bias: -0.02 });
    }
    B([0.165, 0, 0.042], [0.52, 0.13, 0.014], C.plate);                 // 하판
    B([0.0, 0, 0.052], [0.03, 0.24, 0.012], C.dark); B([0.33, 0, 0.052], [0.03, 0.24, 0.012], C.dark); // 암
    B([0.455, 0, 0.05], [0.05, 0.27, 0.045], C.foam);                   // 앞 범퍼
    B([-0.115, 0, 0.05], [0.03, 0.2, 0.03], C.foam);                    // 뒤 범퍼
    if (!clay) S.box(pose, [-0.132, 0, 0.052], [0.004, 0.17, 0.012], [198, 241, 53], { flat: true, nofog: true, glow: 10, cull: true });
    B([0.335, 0, 0.095], [0.008, 0.17, 0.075], C.dark); B([-0.005, 0, 0.095], [0.008, 0.17, 0.075], C.dark); // 쇽 타워
    B([0.05, -0.035, 0.068], [0.07, 0.04, 0.04], hi('motor', C.comp)); // 모터
    B([0.19, 0.03, 0.066], [0.15, 0.048, 0.03], C.batt);                // 배터리
    for (const q of [[0.03, 0.075], [0.03, -0.075], [0.35, 0.075], [0.35, -0.075]]) B([q[0], q[1], 0.085], [0.008, 0.008, 0.07], C.fin);
    B([0.19, 0, 0.122], [0.36, 0.18, 0.006], C.deck);                   // 상판
    B([0.055, 0.03, 0.137], [0.075, 0.06, 0.024], hi('vesc', C.comp)); // VESC (IMU 내장)
    B([0.165, 0, 0.143], [0.11, 0.105, 0.036], hi('compute', C.dark)); // 컴퓨터
    for (let k = -3; k <= 3; k++) B([0.165, k * 0.014, 0.168], [0.095, 0.004, 0.014], hi('compute', C.fin));
    B([0.297, 0, 0.142], [0.05, 0.05, 0.034], hi('lidar', C.lidar));    // LiDAR 받침
    S.cyl(pose, [0.297, 0, 0.176], 0.024, 0.034, 'z', hi('lidar', [18, 19, 21]), Object.assign({ seg: 18 }, E));
    S.cyl(pose, [0.297, 0, 0.199], 0.025, 0.012, 'z', C.cap, { seg: 18, flat: o.hi === 'lidar' });
    const an = (z) => [pose.x + 0.02 * Math.cos(pose.yaw) - 0.07 * Math.sin(pose.yaw), pose.y + 0.02 * Math.sin(pose.yaw) + 0.07 * Math.cos(pose.yaw), z];
    S.line(an(0.125), an(0.3), clay ? [170, 176, 178] : [200, 205, 205], { lw: 1.4 });
  }

  /* 경기장 홀: 먼 벽과 기둥, 천장 조명 띠 */
  function hall(S) {
    const X0 = -7, X1 = 20, Y0 = -6, Y1 = 16, Hh = 5.5, wc = [30, 33, 36];
    const W = (a, b) => { for (let k = 0; k < 12; k++) { const p = [mix(a[0], b[0], k / 12), mix(a[1], b[1], k / 12)], q = [mix(a[0], b[0], (k + 1) / 12), mix(a[1], b[1], (k + 1) / 12)]; S.poly([[p[0], p[1], 0], [q[0], q[1], 0], [q[0], q[1], Hh], [p[0], p[1], Hh]], wc, { flat: true, mul: (k % 3 === 0) ? 1.25 : 1, bias: 5, far: 200, nofog: true, a: 0.9 }); S.poly([[p[0], p[1], 0], [q[0], q[1], 0], [q[0], q[1], 0.9], [p[0], p[1], 0.9]], [20, 22, 24], { flat: true, bias: 4.9, far: 200, nofog: true }); } };
    W([X0, Y0], [X1, Y0]); W([X1, Y0], [X1, Y1]); W([X1, Y1], [X0, Y1]); W([X0, Y1], [X0, Y0]);
    for (let y = Y0 + 2; y < Y1; y += 4) for (let x = X0 + 1; x < X1 - 1; x += 4.5) S.poly([[x, y, Hh], [x + 3, y, Hh], [x + 3, y + 0.22, Hh], [x, y + 0.22, Hh]], [235, 240, 230], { flat: true, nofog: true, far: 200, a: 0.85, glow: 14 });
  }

  /* ---------------- 레이스 장면 ---------------- */
  function race(canvas, opt) {
    opt = Object.assign({ frac: 0.063, view: 'chase', lidar: true, plan: true }, opt || {});
    const i0 = Math.round(opt.frac * track.N), lp = linePoint(i0), lp2 = linePoint(i0 + 3);
    const yaw = Math.atan2(lp2[1] - lp[1], lp2[0] - lp[0]) + 0.035, pose = { x: lp[0], y: lp[1], yaw };
    const fw = [Math.cos(yaw), Math.sin(yaw)];
    let cam;
    if (opt.view === 'chase') cam = { pos: [pose.x - fw[0] * 1.42 + fw[1] * 0.14, pose.y - fw[1] * 1.42 - fw[0] * 0.14, 0.62], target: [pose.x + fw[0] * 2.1, pose.y + fw[1] * 2.1, 0.06], fov: 58, oy: opt.oy || 0, ox: opt.ox || 0 };
    else cam = Object.assign({ fov: 40 }, opt.cam);
    const S = Scene(canvas, cam, { fog: [13, 15, 16], f0: 7, f1: 30, amb: 0.34, light: [-0.5, -0.25, 0.83] });
    S.ground({ near: [50, 54, 57], far: [20, 22, 24], top: [8, 9, 10], gridA: 0.16 });
    S.flush();
    // 코스 바닥을 아주 조금 밝게(주행 가능한 면)
    for (let i = 0; i < track.N; i += 2) {
      const a = track.c[i], b = track.c[(i + 2) % track.N], na = track.nor[i], nb = track.nor[(i + 2) % track.N], ha = track.hw[i], hb = track.hw[(i + 2) % track.N];
      S.poly([[a[0] + na[0] * ha, a[1] + na[1] * ha, 0], [b[0] + nb[0] * hb, b[1] + nb[1] * hb, 0], [b[0] - nb[0] * hb, b[1] - nb[1] * hb, 0], [a[0] - na[0] * ha, a[1] - na[1] * ha, 0]], [64, 69, 72], { flat: true, a: 0.45, bias: 300, stroke: true });
    }
    S.flush();
    // 출발선
    { const p = at(0.02), n = track.nor[p.i], t = track.tan[p.i], h = track.hw[p.i];
      for (let k = -8; k < 8; k++) for (let r2 = 0; r2 < 2; r2++) if ((k + r2) % 2 === 0) { const a0 = k * h / 8, a1 = (k + 1) * h / 8, b0 = r2 * 0.12, b1 = b0 + 0.12; const Q = (a, b) => [p.x + n[0] * a + t[0] * b, p.y + n[1] * a + t[1] * b, 0.002]; S.poly([Q(a0, b0), Q(a1, b0), Q(a1, b1), Q(a0, b1)], [225, 228, 224], { flat: true, a: 0.75, bias: 250 }); } }
    S.flush();
    // 드라이버가 낸 궤적: 1.0 s 앞까지 10점
    const plan = []; let s = 0, ii = i0;
    for (let k = 0; k <= 60; k++) { const p = linePoint(i0 + k); plan.push([p[0], p[1], speedAt(i0 + k)]); }
    if (opt.plan) {
      for (let k = 2; k < 52; k++) {
        const a = plan[k], b = plan[k + 1], d = nrm(b[0] - a[0], b[1] - a[1]), w = 0.05, fade = 1 - Math.pow(k / 52, 2.2);
        const cc = [mix(70, 198, fade), mix(84, 241, fade), mix(60, 53, fade)];
        S.poly([[a[0] + d[0] * w, a[1] + d[1] * w, 0.006], [b[0] + d[0] * w, b[1] + d[1] * w, 0.006], [b[0] - d[0] * w, b[1] - d[1] * w, 0.006], [a[0] - d[0] * w, a[1] - d[1] * w, 0.006]], cc, { flat: true, bias: 90, nofog: true, stroke: true, glow: k < 30 ? 8 : 0 });
      }
      S.flush();
      for (let k = 5; k <= 50; k += 5) { const p = plan[k]; S.dot([p[0], p[1], 0.014], [18, 20, 16], { r: 0.014, min: 1.2, nofog: true, bias: 79 }); }
      S.flush();
    }
    // 지나온 자취
    for (let k = 4; k < 120; k += 1) { const a = linePoint(i0 - k), b = linePoint(i0 - k - 1); S.line([a[0], a[1], 0.004], [b[0], b[1], 0.004], [240, 244, 240], { a: 0.28 * (1 - k / 120), lw: 1.5, bias: 60 }); }
    S.flush();
    // 덕트 벽, 상자, 차
    S.tube(track.left, 0.15, [188, 194, 198]); S.tube(track.right, 0.15, [188, 194, 198]); hall(S);
    for (const b of boxes) { S.box({ x: b.x, y: b.y, yaw: b.yaw }, [0, 0, b.sz / 2], [b.sx, b.sy, b.sz], [201, 162, 107], { stroke: 'rgba(60,40,10,0.5)', lw: 0.8 }); S.box({ x: b.x, y: b.y, yaw: b.yaw }, [0, 0, b.sz + 0.001], [b.sx, 0.045, 0.001], [150, 112, 66], { cull: false }); }
    car(S, pose, { steer: -0.16 });
    if (opt.lidar) {
      const lx = pose.x + fw[0] * 0.297, ly = pose.y + fw[1] * 0.297, pts = scan({ x: lx, y: ly, yaw });
      for (let k = 0; k < pts.length; k += 1) { const p = pts[k]; if (!p.hit) continue; const rr = p.r, th = yaw + p.ang; S.dot([lx + Math.cos(th) * rr, ly + Math.sin(th) * rr, 0.171], p.kind === 1 ? [255, 255, 255] : [127, 227, 242], { r: 0.006, min: 0.9, max: 2.4, bias: -0.2, a: 0.95 }); }
    }
    S.flush();
    return { pose, plan, S, i0 };
    function nrm(x, y) { const l = Math.hypot(x, y) || 1; return [-y / l, x / l]; }
  }

  /* ---------------- 차고 장면: 센서 시야 ---------------- */
  function garage(canvas, opt) {
    opt = Object.assign({ az: -42, el: 29, dist: 2.3, fov: 28, hi: 'lidar', fan: true, target: [0.3, 0, 0.08], ox: 0, oy: 0, fanR: 2.4, dims: true }, opt || {});
    const az = opt.az * Math.PI / 180, el = opt.el * Math.PI / 180, T = opt.target;
    const cam = { pos: [T[0] + opt.dist * Math.cos(el) * Math.cos(az), T[1] + opt.dist * Math.cos(el) * Math.sin(az), T[2] + opt.dist * Math.sin(el)], target: T, fov: opt.fov, ox: opt.ox, oy: opt.oy };
    const S = Scene(canvas, cam, { fog: [10, 11, 12], f0: 3.5, f1: 9, amb: 0.42, light: [-0.35, -0.5, 0.8] });
    const ctx = S.ctx, W = S.W, H = S.H;
    const bg = ctx.createRadialGradient(W * 0.5 + opt.ox, H * 0.56 + opt.oy, 40, W * 0.5 + opt.ox, H * 0.56 + opt.oy, W * 0.62);
    bg.addColorStop(0, '#24282b'); bg.addColorStop(0.55, '#131517'); bg.addColorStop(1, '#090a0b'); ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    const pose = { x: 0, y: 0, yaw: 0 }, LX = 0.297, LZ = 0.171;
    // 바닥 눈금: 0.1 m 격자(가까이), 거리 고리
    for (let g = -1.0; g <= 3.4001; g += 0.1) { const major = Math.abs(Math.round(g * 10) % 5) === 0; S.line([g, -2.2, 0], [g, 2.2, 0], [220, 228, 222], { a: major ? 0.10 : 0.03, bias: 50 }); }
    for (let g = -2.2; g <= 2.2001; g += 0.1) { const major = Math.abs(Math.round(g * 10) % 5) === 0; S.line([-1.0, g, 0], [3.4, g, 0], [220, 228, 222], { a: major ? 0.10 : 0.03, bias: 50 }); }
    S.flush();
    if (opt.fan) {
      const R = opt.fanR, a0 = -135 * Math.PI / 180, a1 = 135 * Math.PI / 180, n = 90;
      // 스캔 평면(센서 높이)
      for (let k = 0; k < n; k++) {
        const t0 = mix(a0, a1, k / n), t1 = mix(a0, a1, (k + 1) / n);
        for (let ring = 0; ring < 6; ring++) {
          const r0 = R * ring / 6, r1 = R * (ring + 1) / 6, al = 0.085 * (1 - ring / 6.4);
          S.poly([[LX + r0 * Math.cos(t0), r0 * Math.sin(t0), LZ], [LX + r1 * Math.cos(t0), r1 * Math.sin(t0), LZ], [LX + r1 * Math.cos(t1), r1 * Math.sin(t1), LZ], [LX + r0 * Math.cos(t1), r0 * Math.sin(t1), LZ]], [198, 241, 53], { flat: true, a: al, nofog: true, bias: 0.25 });
        }
      }
      // 뒤쪽 사각(90°): 빗금
      for (let k = 0; k < 26; k++) { const t = mix(135, 225, (k + 0.5) / 26) * Math.PI / 180; S.line([LX + 0.12 * Math.cos(t), 0.12 * Math.sin(t), LZ], [LX + R * 0.62 * Math.cos(t), R * 0.62 * Math.sin(t), LZ], [255, 184, 31], { a: 0.5, lw: 1, dash: [3, 5], nofog: true }); }
      for (const t of [a0, a1]) S.line([LX, 0, LZ], [LX + R * Math.cos(t), R * Math.sin(t), LZ], [198, 241, 53], { a: 0.9, lw: 1.4, nofog: true });
      for (const rr of [1, 2]) for (let k = 0; k < n; k++) { const t0 = mix(a0, a1, k / n), t1 = mix(a0, a1, (k + 1) / n); S.line([LX + rr * Math.cos(t0), rr * Math.sin(t0), LZ], [LX + rr * Math.cos(t1), rr * Math.sin(t1), LZ], [198, 241, 53], { a: 0.55, lw: 1, nofog: true, dash: [2, 4] }); }
      for (let k = 0; k < n; k++) { const t0 = mix(a0, a1, k / n), t1 = mix(a0, a1, (k + 1) / n); S.line([LX + R * Math.cos(t0), R * Math.sin(t0), LZ], [LX + R * Math.cos(t1), R * Math.sin(t1), LZ], [198, 241, 53], { a: 0.35, lw: 1, nofog: true }); }
      // 광선 몇 가닥(0.25° 간격을 과장 없이: 15°마다 한 가닥)
      for (let d = -135; d <= 135; d += 15) { const t = d * Math.PI / 180; S.line([LX, 0, LZ], [LX + R * Math.cos(t), R * Math.sin(t), LZ], [198, 241, 53], { a: 0.09, lw: 1, nofog: true }); }
    }
    // 기준 물체: 덕트 호스 한 토막(지름 0.30 m)과 낮은 상자(높이 0.12 m) — 스캔 평면이 어디를 지나는지
    if (opt.duct) {
      const D = opt.duct, K = 14, rad = 0.15, dx = D[1][0] - D[0][0], dy = D[1][1] - D[0][1], dl = Math.hypot(dx, dy), nn = [-dy / dl, dx / dl], M = Math.round(dl / 0.035);
      for (let i = 0; i < M; i++) {
        const a = [mix(D[0][0], D[1][0], i / M), mix(D[0][1], D[1][1], i / M)], b = [mix(D[0][0], D[1][0], (i + 1) / M), mix(D[0][1], D[1][1], (i + 1) / M)];
        for (let k = 0; k < K; k++) {
          const t0 = -0.6 + (Math.PI + 1.2) * k / K, t1 = -0.6 + (Math.PI + 1.2) * (k + 1) / K, tm = (t0 + t1) / 2;
          const q = (p, t) => [p[0] + nn[0] * rad * Math.cos(t), p[1] + nn[1] * rad * Math.cos(t), rad + rad * Math.sin(t)];
          S.poly([q(a, t0), q(b, t0), q(b, t1), q(a, t1)], [168, 174, 178], { n: V.norm([nn[0] * Math.cos(tm), nn[1] * Math.cos(tm), Math.sin(tm)]), mul: (i % 2) ? 0.84 : 1, stroke: true, nofog: true });
        }
      }
    }
    if (opt.lowbox) S.box({ x: opt.lowbox[0], y: opt.lowbox[1], yaw: opt.lowbox[2] || 0 }, [0, 0, 0.06], [0.22, 0.18, 0.12], [150, 124, 86], { stroke: 'rgba(255,184,31,0.9)', lw: 1, nofog: true });
    car(S, pose, { style: 'clay', hi: opt.hi, steer: 0.0 });
    // IMU 축
    if (opt.dims) {
      const O = [0.055, 0.03, 0.152];
      S.line(O, [O[0] + 0.14, O[1], O[2]], [255, 77, 61], { lw: 2, nofog: true, bias: -0.3 }); S.line(O, [O[0], O[1] + 0.14, O[2]], [46, 220, 140], { lw: 2, nofog: true, bias: -0.3 }); S.line(O, [O[0], O[1], O[2] + 0.14], [127, 227, 242], { lw: 2, nofog: true, bias: -0.3 });
    }
    S.flush();
    return { S, cam, LX, LZ };
  }

  /* ---------------- 2D 도우미 ---------------- */
  function mapPaths(x, y, w, h, pad) {
    pad = pad === undefined ? 10 : pad;
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (const p of track.left.concat(track.right)) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
    const sc = Math.min((w - 2 * pad) / (x1 - x0), (h - 2 * pad) / (y1 - y0)), ox = x + (w - (x1 - x0) * sc) / 2, oy = y + (h - (y1 - y0) * sc) / 2;
    const T = (p) => [ox + (p[0] - x0) * sc, oy + (y1 - p[1]) * sc];
    const path = (pts, close) => pts.map((p, i) => { const q = T(p); return (i ? 'L' : 'M') + q[0].toFixed(1) + ' ' + q[1].toFixed(1); }).join('') + (close === false ? '' : 'Z');
    return { T, sc, left: path(track.left), right: path(track.right), center: path(track.c), path };
  }
  function linePath(vals, x, y, w, h, lo, hi) {
    return vals.map((v, i) => (i ? 'L' : 'M') + (x + w * i / (vals.length - 1)).toFixed(1) + ' ' + (y + h - h * (v - lo) / (hi - lo)).toFixed(1)).join('');
  }
  function series(n, f, seed, noise) { const R = rng(seed); const out = []; for (let i = 0; i < n; i++) out.push(f(i / (n - 1)) + (R() - 0.5) * 2 * noise * (0.6 + R())); return out; }
  const svg = (tag, attrs, inner) => `<${tag} ${Object.entries(attrs).map(([k, v]) => `${k}="${v}"`).join(' ')}>${inner || ''}</${tag}>`;

  window.PW = { V, rng, clamp, mix, track, boxes, at, linePoint, lineOffset, speedAt, scan, Scene, car, race, garage, mapPaths, linePath, series, svg, TAU };
})();
