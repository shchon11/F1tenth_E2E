import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// ------------------------------------------------------------------ renderer / scene
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1116);
scene.fog = new THREE.Fog(0x0e1116, 60, 140);
const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 400);
camera.up.set(0, 0, 1);                       // z-up (ROS convention)
camera.position.set(-6, -6, 5);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.12;

scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.add(new THREE.HemisphereLight(0xdfe7ff, 0x1a1d24, 0.6));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(-20, -30, 40); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
const sc = sun.shadow.camera; sc.near = 1; sc.far = 150;
scene.add(sun); scene.add(sun.target);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize); resize();

// ------------------------------------------------------------------ state
let init = null, cars = [], carModel = null, wheelRadius = 0.055;
let scanPts = null, scanGeo = null, raceLine = null, trails = [], trailOn = true;
let mode = 0; const MODES = ['chase', 'top', 'orbit', 'overview', 'closeup'];
const MATS = {   // geometry-name suffix -> PBR params (the GLB only carries vertex colors)
  rubber: { roughness: 0.92, metalness: 0.0 }, plastic: { roughness: 0.62, metalness: 0.05 },
  metal: { roughness: 0.35, metalness: 0.9 }, blue: { roughness: 0.3, metalness: 0.9 },
  glass: { roughness: 0.15, metalness: 0.2, transparent: true, opacity: 0.85 },
  paint: { roughness: 0.45, metalness: 0.1 }, pcb: { roughness: 0.55, metalness: 0.1 } };
function applyMaterials(root) {
  root.traverse(o => {
    if (!o.isMesh) return;
    if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    const cls = Object.keys(MATS).find(k => o.name.endsWith('_' + k) || o.name.endsWith(k));
    const m = new THREE.MeshStandardMaterial({ vertexColors: true, ...(MATS[cls] || MATS.plastic) });
    o.material = m; o.castShadow = true; o.receiveShadow = true;
  });
}
let focus = 0, showLidar = true, showRace = true;
let lastFrame = null, frames = 0, fpsT = performance.now(), fps = 0, rxHz = 0, rxCount = 0;
const hudBody = document.getElementById('hud-body'), status = document.getElementById('status');

// ------------------------------------------------------------------ track
function buildTrack(tr) {
  const g = new THREE.Group();
  const [ox, oy] = tr.origin, [sx, sy] = tr.size;
  // ground far plane
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(sx * 3, sy * 3),
    new THREE.MeshStandardMaterial({ color: 0x14181f, roughness: 1 }));
  ground.position.set(ox + sx / 2, oy + sy / 2, -0.02); ground.receiveShadow = true; g.add(ground);
  const grid = new THREE.GridHelper(Math.max(sx, sy) * 3, Math.max(sx, sy) * 3, 0x22283a, 0x1a1f2b);
  grid.rotation.x = Math.PI / 2; grid.position.set(ox + sx / 2, oy + sy / 2, -0.015); g.add(grid);
  // contours: the largest (by bbox area) is the outer boundary of the drivable region
  const cs = tr.contours.map(c => ({ pts: c, area: bboxArea(c) })).sort((a, b) => b.area - a.area);
  if (cs.length) {
    const shape = new THREE.Shape(cs[0].pts.map(p => new THREE.Vector2(p[0], p[1])));
    for (let i = 1; i < cs.length; i++) shape.holes.push(new THREE.Path(cs[i].pts.map(p => new THREE.Vector2(p[0], p[1]))));
    const floor = new THREE.Mesh(new THREE.ShapeGeometry(shape, 1),
      new THREE.MeshStandardMaterial({ color: 0x3a3d43, roughness: 0.95, metalness: 0.0 }));
    floor.position.z = 0.0; floor.receiveShadow = true; g.add(floor);
  }
  // boundaries: flexible duct hoses (corrugated grey tubes) laid on the floor along every contour
  const R = (tr.duct_diameter || 0.2) / 2;
  const ductMat = new THREE.MeshStandardMaterial({ map: ductTexture(), color: 0xd9dbe0, roughness: 0.55, metalness: 0.35 });
  for (const c of cs) {
    const pts = simplify(c.pts, 0.06).map(p => new THREE.Vector3(p[0], p[1], R));
    if (pts.length < 4) continue;
    const curve = new THREE.CatmullRomCurve3(pts, true, 'centripetal', 0.5);
    const len = curve.getLength();
    const geo = new THREE.TubeGeometry(curve, Math.max(32, Math.round(len / 0.05)), R, 10, true);
    // corrugation: repeat the ring texture every ~2 cm along the hose
    const uv = geo.attributes.uv; for (let i = 0; i < uv.count; i++) uv.setX(i, uv.getX(i) * len / 0.02);
    const duct = new THREE.Mesh(geo, ductMat); duct.castShadow = true; duct.receiveShadow = true; g.add(duct);
  }
  return g;
}
function ductTexture() {
  const cv = document.createElement('canvas'); cv.width = 64; cv.height = 8; const ctx = cv.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 64, 0);
  grad.addColorStop(0, '#6f737a'); grad.addColorStop(0.35, '#c4c7cc'); grad.addColorStop(0.5, '#e6e8eb'); grad.addColorStop(0.65, '#c4c7cc'); grad.addColorStop(1, '#6f737a');
  ctx.fillStyle = grad; ctx.fillRect(0, 0, 64, 8);
  const t = new THREE.CanvasTexture(cv); t.wrapS = THREE.RepeatWrapping; t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8; return t;
}
function simplify(pts, minDist) {   // drop near-duplicate contour vertices so the spline stays smooth
  const out = [pts[0]]; let last = pts[0];
  for (let i = 1; i < pts.length; i++) { const p = pts[i]; if (Math.hypot(p[0] - last[0], p[1] - last[1]) >= minDist) { out.push(p); last = p; } }
  return out;
}
function bboxArea(c) { let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9; for (const p of c) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); } return (x1 - x0) * (y1 - y0); }

function buildRaceline(rl) {
  const pts = [], cols = [], vmin = Math.min(...rl.map(p => p[2])), vmax = Math.max(...rl.map(p => p[2]));
  for (const p of [...rl, rl[0]]) { pts.push(new THREE.Vector3(p[0], p[1], 0.012)); const c = new THREE.Color().setHSL(0.66 * (1 - (p[2] - vmin) / (vmax - vmin + 1e-6)), 1, 0.55); cols.push(c.r, c.g, c.b); }
  const geo = new THREE.BufferGeometry().setFromPoints(pts); geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 }));
}
function buildCenterline(cl) {
  const pts = [...cl, cl[0]].map(p => new THREE.Vector3(p[0], p[1], 0.008));
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: 0x4a90e2, dashSize: 0.3, gapSize: 0.2 }));
}

// ------------------------------------------------------------------ cars
function makeCar(i) {
  const root = new THREE.Group();
  const body = carModel.clone(true);
  applyMaterials(body);
  root.add(body);
  const wheels = ['wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr'].map(n => body.getObjectByName(n));
  // collision overlay
  const box = new THREE.Mesh(new THREE.BoxGeometry(init.car.length, init.car.width, 0.22),
    new THREE.MeshBasicMaterial({ color: 0xff3b30, transparent: true, opacity: 0.35 }));
  box.position.set(init.car.wheelbase / 2, 0, 0.11); box.visible = false; root.add(box);
  // id label sprite
  const label = makeLabel(String(i)); label.position.set(0.15, 0, 0.36); root.add(label);
  scene.add(root);
  const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: i === focus ? 0x7ee787 : 0x3d4a5c, transparent: true, opacity: 0.8 }));
  trail.frustumCulled = false; scene.add(trail); trails.push({ line: trail, pts: [] });
  return { root, wheels, box, label, roll: 0, lastS: null };
}
function makeLabel(text) {
  const cv = document.createElement('canvas'); cv.width = 64; cv.height = 32; const ctx = cv.getContext('2d');
  ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fillRect(0, 0, 64, 32); ctx.fillStyle = '#fff'; ctx.font = 'bold 22px sans-serif'; ctx.textAlign = 'center'; ctx.fillText(text, 32, 24);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), depthTest: false })); sp.scale.set(0.16, 0.08, 1); sp.material.opacity = 0.85; return sp;
}

// ------------------------------------------------------------------ lidar points
function makeScan(n) {
  scanGeo = new THREE.BufferGeometry();
  scanGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  scanGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  scanPts = new THREE.Points(scanGeo, new THREE.PointsMaterial({ size: 0.06, vertexColors: true, sizeAttenuation: true }));
  scanPts.frustumCulled = false; scene.add(scanPts);
}

// ------------------------------------------------------------------ frame handling
function onFrame(buf) {
  const f = new Float32Array(buf);
  const t = f[0], n = f[1] | 0, nb = f[2] | 0, foc = f[3] | 0;
  const carsArr = f.subarray(4, 4 + n * 8), scan = f.subarray(4 + n * 8, 4 + n * 8 + nb);
  lastFrame = { t, n, nb, foc, cars: carsArr, scan };
  rxCount++;
  while (cars.length < n) cars.push(makeCar(cars.length));
  const L = init.car.wheelbase, cogx = init.car.cog_x;
  for (let i = 0; i < n; i++) {
    const o = i * 8, x = carsArr[o], y = carsArr[o + 1], yaw = carsArr[o + 2], steer = carsArr[o + 3], vx = carsArr[o + 4], coll = carsArr[o + 6] > 0.5;
    const c = cars[i];
    // sim state is at the CoG; model origin is the rear axle
    c.root.position.set(x - cogx * Math.cos(yaw), y - cogx * Math.sin(yaw), 0); c.root.rotation.set(0, 0, yaw);
    c.roll += vx * init.control_dt / wheelRadius;
    c.wheels.forEach((w, k) => { if (!w) return; w.rotation.set(0, 0, k < 2 ? steer : 0); w.rotateY(c.roll); });
    c.box.visible = coll;
    if (trailOn) { const tr = trails[i]; tr.pts.push(new THREE.Vector3(x, y, 0.02)); if (tr.pts.length > 600) tr.pts.shift(); tr.line.geometry.setFromPoints(tr.pts); tr.line.visible = true; }
    else trails[i].line.visible = false;
  }
  // lidar (focus car)
  if (scanPts === null) makeScan(nb);
  const o = foc * 8, x = carsArr[o], y = carsArr[o + 1], yaw = carsArr[o + 2];
  const lx = x - cogx * Math.cos(yaw) + init.lidar.mount_x * Math.cos(yaw), ly = y - cogx * Math.sin(yaw) + init.lidar.mount_x * Math.sin(yaw);
  const pos = scanGeo.attributes.position.array, col = scanGeo.attributes.color.array, a0 = -init.lidar.fov / 2, da = init.lidar.fov / (nb - 1);
  for (let i = 0; i < nb; i++) {
    const r = scan[i];
    if (r < 0) { pos[i * 3 + 2] = -10; continue; }
    const a = yaw + a0 + i * da; pos[i * 3] = lx + r * Math.cos(a); pos[i * 3 + 1] = ly + r * Math.sin(a); pos[i * 3 + 2] = init.lidar.height || 0.15;
    const h = 0.33 * Math.min(1, r / init.lidar.range_max); const cc = new THREE.Color().setHSL(0.02 + h, 1, 0.55); col[i * 3] = cc.r; col[i * 3 + 1] = cc.g; col[i * 3 + 2] = cc.b;
  }
  scanGeo.attributes.position.needsUpdate = true; scanGeo.attributes.color.needsUpdate = true;
  scanPts.visible = showLidar;
  focus = foc;
}

// ------------------------------------------------------------------ camera
const camTarget = new THREE.Vector3(), camPos = new THREE.Vector3();
function updateCamera(dt) {
  if (!lastFrame) return;
  const o = focus * 8, c = lastFrame.cars, x = c[o], y = c[o + 1], yaw = c[o + 2];
  const m = MODES[mode];
  if (m === 'orbit') { controls.enabled = true; controls.update(); return; }
  controls.enabled = false;
  if (m === 'chase') { camPos.set(x - 1.7 * Math.cos(yaw), y - 1.7 * Math.sin(yaw), 0.75); camTarget.set(x + 1.2 * Math.cos(yaw), y + 1.2 * Math.sin(yaw), 0.1); }
  else if (m === 'closeup') { const a = performance.now() * 0.0004; camPos.set(x + 1.1 * Math.cos(a), y + 1.1 * Math.sin(a), 0.45); camTarget.set(x, y, 0.1); }
  else if (m === 'top') { camPos.set(x - 0.01 * Math.cos(yaw), y - 0.01 * Math.sin(yaw), 14); camTarget.set(x, y, 0); }
  else { const tr = init.track; camPos.set(tr.origin[0] + tr.size[0] / 2, tr.origin[1] + tr.size[1] / 2 - 0.01, Math.max(tr.size[0], tr.size[1]) * 1.15); camTarget.set(tr.origin[0] + tr.size[0] / 2, tr.origin[1] + tr.size[1] / 2, 0); }
  const k = 1 - Math.exp(-dt * (m === 'chase' ? 8 : m === 'closeup' ? 30 : 4));
  camera.position.lerp(camPos, k); controls.target.lerp(camTarget, k); camera.lookAt(controls.target);
  sun.target.position.set(x, y, 0);
}

// ------------------------------------------------------------------ hud
function hud() {
  if (!lastFrame) return;
  const f = lastFrame, o = focus * 8, c = f.cars, coll = [...Array(f.n).keys()].filter(i => c[i * 8 + 6] > 0.5).length;
  document.getElementById('hud-title').textContent = `f1sim · ${init.track.name} · ${f.n} cars`;
  const row = (k, v) => `<div class="row"><span>${k}</span><b>${v}</b></div>`;
  hudBody.innerHTML = row('sim time', f.t.toFixed(2) + ' s') + row('focus car', focus) + row('speed', c[o + 4].toFixed(2) + ' m/s') +
    row('steer', (c[o + 3] * 57.3).toFixed(1) + '°') + row('lap', c[o + 5] | 0) + row('s', c[o + 7].toFixed(1) + ' m') +
    row('collided', `<span class="${coll ? 'bad' : 'good'}">${coll}/${f.n}</span>`) + row('camera', MODES[mode]);
  status.textContent = `render ${fps.toFixed(0)} fps · stream ${rxHz.toFixed(0)} Hz`;
}

// ------------------------------------------------------------------ input
window.addEventListener('keydown', e => {
  if (e.key === 'c' || e.key === 'C') mode = (mode + 1) % MODES.length;
  if (e.key === ']') { focus = (focus + 1) % Math.max(1, cars.length); sendFocus(); }
  if (e.key === '[') { focus = (focus - 1 + cars.length) % Math.max(1, cars.length); sendFocus(); }
  if (e.key === 'l' || e.key === 'L') showLidar = !showLidar;
  if (e.key === 'r' || e.key === 'R') { showRace = !showRace; if (raceLine) raceLine.visible = showRace; }
  if (e.key === 't' || e.key === 'T') trailOn = !trailOn;
});
let ws = null;
function sendFocus() { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ focus })); trails.forEach((t, i) => t.line.material.color.set(i === focus ? 0x7ee787 : 0x3d4a5c)); }

// ------------------------------------------------------------------ connect
const url = new URL(window.location.href);
function connect(port) {
  ws = new WebSocket(`ws://${url.hostname}:${port}`); ws.binaryType = 'arraybuffer';
  ws.onmessage = ev => {
    if (typeof ev.data === 'string') { const m = JSON.parse(ev.data); if (m.type === 'init') onInit(m); return; }
    if (init && carModel) onFrame(ev.data);
  };
  ws.onclose = () => { status.textContent = 'disconnected – retrying…'; setTimeout(() => connect(port), 1500); };
}
function onInit(m) {
  init = m; status.textContent = 'loading…';
  scene.add(buildTrack(m.track));
  if (m.centerline) scene.add(buildCenterline(m.centerline));
  if (m.raceline) { raceLine = buildRaceline(m.raceline); scene.add(raceLine); }
  new GLTFLoader().load(m.car.url, g => {
    carModel = g.scene; applyMaterials(carModel);
    const wl = carModel.getObjectByName('wheel_fl'); if (wl) { const b = new THREE.Box3().setFromObject(wl); wheelRadius = (b.max.z - b.min.z) / 2; }
    status.textContent = 'ready';
  }, undefined, err => { status.textContent = 'car model failed: ' + err; carModel = new THREE.Group(); });
}
const wsPort = url.searchParams.get('ws') || (parseInt(url.port || '8765') + 1);
mode = Math.max(0, MODES.indexOf(url.searchParams.get('cam') || 'chase'));
connect(wsPort);

// ------------------------------------------------------------------ loop
let prev = performance.now();
function loop(now) {
  const dt = Math.min(0.1, (now - prev) / 1000); prev = now;
  updateCamera(dt); renderer.render(scene, camera); hud();
  frames++; if (now - fpsT > 500) { fps = frames * 1000 / (now - fpsT); rxHz = rxCount * 1000 / (now - fpsT); frames = 0; rxCount = 0; fpsT = now; }
  requestAnimationFrame(loop);
window.dbg = { scene, get cars() { return cars; }, get carModel() { return carModel; }, THREE };
}
requestAnimationFrame(loop);
