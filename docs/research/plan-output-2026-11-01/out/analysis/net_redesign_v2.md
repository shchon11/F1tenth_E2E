# Policy network redesign v2 — representation from first principles (2026-11-25)

Status: proposal. Nothing built, nothing run. Replaces `net_redesign_draft.md` (the 0.125 m BEV plan). English first, Korean below.

Fixed by the user's decisions: output = 1 s plan of 10 points with speeds (`plan_steps` decode, MAP follows on the car); training = exact plan execution, PPO, lap-time reward, at-fault contact cost, plan–car overlap cost; one network for {solo, 1v1} × {boxes, none}; inputs = own sensors (2D LiDAR 1081 beams / 270° / 0.25° / 10 m clamp — `params.py n_beams 1081`, the policy reads all of them, `scan_subsample 1`, 3 stacked scans; IMU; wheel odometry) and dead-reckoned ego motion only; no racing line, centreline, lap position, global map or scan-matching localisation; Jetson AGX Orin, 0.1 s decisions.

To beat: qualifier scan31b2 u600 — 7.10 s laps, 0.11 hard box hits per box session; race33b u975–u1075 — at-fault 0.97–0.99 per 1v1 session, passes ~1.55, against a 0.6x car at-fault 1.84–2.19 and plan–future overlap stalled at ~13 %.

---

## 1. Why the dense-grid plan was the wrong level of thinking

The draft asked "which cell size" and answered 0.125 m. That is a question about a design constant, and every patch I then offered (keep the raw scan beside the grid, store sub-cell offsets, a finer near-field grid) was a way to pay for the constant's damage somewhere else. The three measured facts that motivated the grid never required one:

- The opponent-prediction gain (polar+GRU 0.50 m → BEV 10 frames 0.34 m at 1 s) came from **ego-compensated frames with time channels**, not from the grid. The uncompensated polar stack scored 0.60 m even at 12× compute; polar+GRU, which must learn the compensation, 0.50 m. The grid was the carrier of the compensation, never shown to be its cause.
- The hidden-box result (grid memory moved by odometry 0.95 recall vs GRU 0.16–0.29 = no memory) is about **where stored content lives**: at a position carried forward by the SE(2) increment. That is a property of "content with coordinates", not of cells.
- The policy today passes boxes with a 30 cm median gap, 10 % under 10 cm, and the sensor measures returns to ~1–4 cm (the UST-10LX datasheet quotes ±40 mm; beam spacing 0.25° = 2.2 cm at 5 m, 4.4 cm at 10 m; a 0.22 m box at 5 m is 10 beams). A 12.5 cm cell throws away one order of magnitude of what the sensor gives — on exactly the quantity the plan head reads.

The right question is: **what does the plan head read, at what precision is that information available, and how does each piece reach the plan point it concerns?**

1. The plan head's decision variable is a continuous quantity: free space around plan point k at time t_k (0.1 k s). It is a function of exact coordinates — of returns, of the dead-reckoned ego path, of other cars' motion.
2. Information precision is not uniform in time:
   - returns in the newest frame: 1–4 cm (sensor-limited);
   - returns 0.1–0.5 s old, dead-reckoned into the current frame: odometry adds ~2 cm and 0.5° per 0.1 s (the noise model trained with in the probe; harmless there) → 4–5 cm and 1° over 0.5 s, i.e. still a few cm at 5 m;
   - things seen a lap ago (6–7 s, 60–70 increments): 17 cm and 4° random walk → 0.4–0.7 m at 5–10 m lever arm. The probe measured this directly: 0.95 recall with true motion, 0.78 at 1× noise, 0.49 at 2×.
3. Therefore one representation precision for everything is wrong twice: a 0.125 m grid is 10× too coarse for the present and 2–5× finer than the past can ever be, and its cost grows with area / cell². The present should keep exact coordinates; the past should be stored at the precision odometry allows; and the read-out should be a **query at exact (x, y, t)** — resolution-free regardless of how the content is stored.
4. Count-agnostic follows for free when the unit is "a return with coordinates and an age" and the reader is attention: the number of boxes, cars or walls changes the number of tokens, never the module.

This is the frame the field has moved to since 2024: sparse instance tokens with raw coordinates embedded by an MLP, propagated across frames by an explicit SE(2)/SE(3) transform, read by planning queries anchored on candidate trajectory points; dense BEV survives where the sensor is a camera and the grid is the only place to fuse views. We have a 2D LiDAR: the points are already there.

---

## 2. What recent work does (links I opened, and what the scouts opened)

Opened by me:
- SparseDrive — https://github.com/swc-17/SparseDrive (code 2024-06, ICRA 2025; SparseDriveV2 released 2026-04-01). Sparse perception (900 detection anchors + 100 map polylines, instance memory queue) → parallel motion/planning, no dense BEV; 9.0 FPS vs UniAD 1.8, training 20 h vs 144 h.
- DiffusionDrive — https://github.com/hustvl/DiffusionDrive and https://arxiv.org/abs/2411.15139 (CVPR 2025 highlight). Trajectory anchors (20, K-means) refined by 2 denoising steps; trajectory coordinates drive deformable sampling of the scene; 45 FPS on a 4090; NAVSIM PDMS 88.1.
- Sparse4D v3 — https://arxiv.org/html/2311.11722 : 900 instances of which 600 temporal, 6 decoder layers, 19.8 FPS (R50); anchors carried across frames by ego pose and velocity; the anchor's raw components are embedded by small MLPs and (v3) **concatenated** with the content features so position is not washed out in attention.
- StreamPETR — https://arxiv.org/html/2303.11926 : memory queue 4 frames × 256 top-K objects; stored 3D centres transformed by the relative ego pose E_{t-1}^t (static assumption — their explicit velocity compensation did not help); Motion-aware LayerNorm conditions the query content and positional encoding on (ego transform, velocity, Δt); memory beyond ~1 s saturates (Table 7). The 2024-26 sparse planners above build on this.
- Point Transformer V3 — https://arxiv.org/html/2312.10035v2 (CVPR 2024): sorts points along a space-filling curve and attends inside patches of 1024 points; dropped relative position encoding (26 % of forward time) for a sparse-conv conditional PE; 44 ms / 1.2 GB at inference on a 4090 for a full nuScenes sweep. For a 2D scan the serialisation is the beam order itself.
- Memory Gym — https://arxiv.org/abs/2309.17207 (JMLR 2025) and https://github.com/MarcoMeter/episodic-transformer-memory-ppo : under PPO, TransformerXL over past observations beats a GRU only on finite tasks and only with an observation-reconstruction auxiliary loss; on endless tasks the GRU wins by large margins (normalised peak 0.98 vs 0.39). Attention over **time** is not the memory device to adopt under PPO; attention over **space** within a decision is a different thing.

Opened by the scouts (facts quoted from their reads):
- SparseDriveV2 — https://arxiv.org/html/2603.29163v1 (2026-03): scores a path vocabulary (1024) × speed-profile vocabulary (256) instead of generating; path–scene interaction by deformable aggregation beat cross-attention; NAVSIM PDMS 92.0. Same output factorisation as ours (path + speed profile).
- iPad — https://arxiv.org/html/2505.15111v1 : 64 proposals, each proposal's four corners become reference points for deformable attention, iterated 4×; 43 ms on a 4090; PDMS 91.7 with no past frames.
- UnO — https://arxiv.org/html/2406.08691v1 (CVPR 2024 oral): 5–6 past LiDAR sweeps → encoder → an **implicit decoder queried at any (x, y, z, t)**, self-supervised from future returns (free along the ray, occupied within 0.1 m of the return), 3 s horizon, class- and count-agnostic by construction. This is the resolution-free read-out pattern.
- CaRL — https://arxiv.org/html/2504.17838 (CoRL 2025): the closest PPO-at-scale precedent (BEV raster 256×256, 300 M samples, minibatch 16 384, 8 ms inference). Direct control output, so not our format.
- Recent learned racing: TinyLidarNet https://arxiv.org/html/2410.07447v1 (1D conv on the raw 1081-beam scan, real car, BC; opponents as static obstacles); End2Race https://arxiv.org/html/2509.16894v1 (360-beam scan + GRU, IL, sim, passive opponent, 59 % overtakes); AID-4 https://github.com/thdrbgur112/AID-4 (108 min-pooled sectors, memoryless MLP, 8192 Isaac Lab envs, dynamic case unsolved); Steiner et al. https://arxiv.org/html/2510.26040 (10 sectors, TD3, real car at 1.5 m/s); Sony GT7 vision agent https://arxiv.org/html/2504.09021v1 (conv → GRU → MLP, QR-SAC, "removing the RNN = complete failure"). Nobody in 2024–26 races a learned LiDAR policy against moving cars at our speed; all real-car LiDAR policies compress the scan early (sectors or a 1D conv flatten), which is exactly where our motion and precision are lost.
- 2D-scan attention in RL exists only with ~30 sector tokens (de Heuvel https://arxiv.org/html/2310.19670v2 , DDPG, Turtlebot) or 10 time-step tokens (Transqer https://arxiv.org/html/2502.19892v1 , DQN, runs on an Orin Nano with a UST-10LX in 1.2 ms). No published RL policy attends over ~1000 beams; that is the open risk of design A/C below and the reason for gate 1.
- RLBenchNet https://arxiv.org/html/2505.15040v1 : under CleanRL PPO a TransformerXL memory costs ~10× the GPU memory of a GRU (1.77 vs 0.19 GB) — another reason to keep the recurrent part a GRU.

---

## 3. Three architectures

Common to all three (the user's fixed parts): the plan head emits 20 numbers → `plan_steps` decode → 10 points + speeds; ego inputs as today (IMU, last commands, previous plan relative to its own 0.1 s pose, speed estimate head); a GRU (128) carries the implicit place code PPO already learns (pose decodable to 0.32 m) — Memory Gym and the GT7 agent say the recurrent part should stay a GRU; critic privileged as now.

### A — Point/query policy: points stay points, memory is a point set

**Tokens.** Every return of the newest K = 5 frames (0, 0.1, …, 0.4 s old) becomes a point (x, y) in the current car frame through the chain of dead-reckoned SE(2) increments (what `rel_chain` + `Geo.points` already do in the probe — no raster), with age τ and range r. Per-point feature = MLP(Fourier(x, y) at scales 0.05 / 0.2 / 1 / 5 m, τ, r) → 64 (Sparse4D's "raw components through small MLPs"; Fourier scales so that 2 cm and 5 m are both resolvable). Points are grouped into runs of consecutive beams (8 in the newest frame → 135 tokens; 16 in older frames → 68 each): token = max-pool of the point features + the run's mean position. N ≈ 400 tokens, d = 96. Grouping by beam order is the 2D case of PTv3's serialisation; it is count-agnostic (a box spanning three runs is three tokens, a car ten, a wall fifty) and keeps exact coordinates inside each token. Frames before a restart are simply absent (variable N is fine).

**Queries.** 10 plan-time queries q_k = learned embedding + time encoding of t_k + Fourier PE of the anchor a_k = the previous plan's point k moved into the current frame (`PlanObs2.plan_xy` already keeps it). Plus one ego query (speed estimate, IMU, last commands). Keys carry **relative** position Fourier(token − a_k) and age, so each query's attention literally asks "what is near where I will be at t_k, and how was it moving" — the anchor-as-PE pattern of SparseDrive / Sparse4D / iPad's proposal corners.

**Attention pattern.** 2 blocks of [cross-attention queries → tokens, self-attention among the 11 queries, FFN], d = 128, 4 heads, pre-LN. **No self-attention among tokens**: cost is O(Q·N), the Perceiver/DETR-decoder shape, not O(N²). The ego query's output feeds the driving GRU; the GRU state is added back to all queries (the place code reaches the plan). Output per plan query: (turn, speed) → 20 numbers; zero-initialised so the fresh network drives as its anchors (the straight/previous plan).

**Memory (A's distinctive part): a bounded point memory.** M = 256 memory tokens, each (x, y, age, feature). Every decision: positions moved by the SE(2) increment (StreamPETR/Sparse4D propagation; dead reckoning only — **no re-registration against the current scan**, which would be scan matching, vetoed), ages += 0.1, tokens that leave a window (−20 … +20 m) are dropped, and the current frame's tokens with the top write scores replace the lowest-scored slots (StreamPETR's top-K push). The plan queries cross-attend to memory tokens with the same relative PE.

**cm precision:** yes for the present — the token keeps exact coordinates and the query reads relative position directly. **Moving cars:** the 5-frame trail with ages is the same evidence the BEV probe used (0.34 m) at exact coordinates; the test says whether exactness buys more. **Hidden boxes:** depends on the write score — and that is A's weak point: from PPO alone a learned write score has nothing to earn (the latent-GRU failure in a new coat); it needs the hindsight target of §4. **Any number of objects:** by construction.

**Cost.** Per decision ≈ 56 MFLOP per-point MLP (5 400 points) + ≈ 60 MFLOP for two cross-attention blocks (K/V projections dominate) + memory read ≈ 20 MFLOP ≈ **0.14 GFLOP** (draft BEV: 0.37 + 0.5 GFLOP). Orin: launch-bound, 1–2 ms in TensorRT FP16; the token build is a gather + SE(2) transform, no raster kernel. Training at 1000 envs: rollout +≈15 ms per control step on the 4070 (sim ≈ 30 ms today); PPO epochs 256 k decisions × 0.14 GFLOP × 3 × 4 ≈ 430 TFLOP ≈ 35 s → update ≈ 2–2.5× today's; tokens rebuilt inside the minibatch from the stored scans (one scan per step is already stored; the 4 older frames are gathered from earlier rows), odometry increments stored (3 floats/step); memory tokens stored only at chunk starts (`--mem-chunk` pattern), 256 × (3 + 96) × fp16 ≈ 50 kB per env.

**Failure modes.** (i) Attention under PPO from scratch has no 2024–26 precedent at this token count (gate 1). (ii) The write policy: without a self-supervised target it will not learn what to store. (iii) Beam-run grouping gives far objects few tokens: a car at 9 m spans ~2 runs — the probe's 6–10 m bin tells. (iv) Variable token count makes CUDA-graph capture awkward (pad to fixed N with masks).

### B — Dense BEV done properly: pillars, warped conv memory, continuous read-out

What it takes to make a grid honest rather than patched:
- **Pillars, not occupancy.** Each non-empty cell (0.125 m, −4 … +12 × ±6 m) encodes its returns through a per-point MLP on the exact in-cell offsets and age, max-pooled (PointPillars-style, learned, not a hand-written sub-cell offset). The cell is then a bucket, not a quantiser.
- **Read-out at the plan points, not from a pooled vector.** The plan head samples the feature map at the anchors a_k (bilinear) or through an implicit decoder (UnO) — the precision loss in the draft was in the flatten → 256 vector as much as in the cell.
- **Memory** = conv-GRU hidden grid warped every step by `grid_sample` with the SE(2) increment, 0.25 m cells (the probe's device, 0.95 / 0.78), trained with odometry noise.
- **Time** = K frames as channels, warped into the current frame (the probe's winning input).

**Can it reach cm precision at Orin cost?** Precision: probably yes with pillars + point read-out — the probe (§5) measures it. Cost: the honest problem. 0.4–0.9 GFLOP per decision; the rasteriser is 5–10 ms on Orin in today's PyTorch pieces (needs a fused kernel; ~1 ms then); cost ∝ area / cell², so the rear/side extent that the blind-sector cases need is paid at the same price as the nose; 90 % of cells are empty and still convolved; repeated bilinear warping blurs the memory grid every step (70 warps over a lap); fixed extent and fixed K frames. It is a sound design — just the camera-era one, and every one of its parameters is a resolution constant.

**Failure modes.** Blur of the warped memory; cost at the extent the blind sector needs; two code paths (raster kernel in the trainer and on Orin) to keep bit-equal.

### C — Hybrid, precision-matched: points for the present, an odometry-warped grid for the past, continuous queries for both

Front end and queries exactly as A. Memory as in B but at the precision the past actually has: an odometry-warped grid at **0.25 m**, −6 … +14 × ±8 m (80 × 64 cells), 16 channels, written by the newest frame's tokens (scatter to the cell of the token position, learned write), gated update, warped by the SE(2) increment each step, trained with the probe's odometry noise. Read by the same plan queries: the grid's non-empty cells are tokens (cell centre, age, feature) in the same cross-attention, with the same relative PE. 0.25 m is below the 0.4–0.7 m odometry drift of a lap-old memory, so nothing is lost; the grid's write is geometric (no learned selection), which is why it worked in the probe where every learned latent memory failed.

**Cost.** A's 0.14 GFLOP + ≈ 0.05–0.1 GFLOP for the 16-channel gated grid ≈ **0.2 GFLOP**; state at chunk starts 80 × 64 × 16 × fp16 = 164 kB per env (0.66 GB at 1000 envs, 4 chunks).

**Failure modes.** Same PPO-attention risk as A; the grid memory's long horizon is odometry-limited (0.78 recall at 1× noise is the ceiling we measured, not a design flaw, and the real car's wheel slip is a systematic error the random-walk model does not cover — a bag-replay check is in the plan).

---

## 4. Recommendation: C, with A's point memory as the measured alternative

**Decisive argument.** The plan head reads free space at exact (x, y, t_k). Only a design whose tokens carry exact coordinates and whose read-out is a query at those coordinates has no resolution constant at all; the cost of such a design is O(Q·N) with Q = 11 and N ≈ 400, which is cheaper than any grid that would meet the sensor's precision. For the past, exactness is not available — odometry sets it at 0.2–0.7 m — and the one memory device that worked in the probe (0.95 / 0.78 vs 0.16–0.29 for every learned latent) is the geometric one: content moved by the SE(2) increment with a write that needs no learned selection. C puts each piece of information at the precision it actually has and reads both through the same queries. A's point memory may match the grid; it must prove it on the same probe before it replaces a device that already works.

**What trains the representation.** PPO alone did not teach the latent memory anything the plan could read (trunk weights 0.05 vs 0.55; rare events). The 11-24 agreement already added a prediction head with its own optimizer. Its count-agnostic generalisation is **one** target, the quantity the plan reads: for each plan query, free space at (a_k, t_k) — signed clearance to the nearest object (wall, box, car) at t_k and occupancy at points sampled around the anchor — truth from the sim in hindsight, the car's own future scans on a real car (UnO's self-supervision). Own optimizer; its output (10 clearances + their confidence) reaches the trunk detached (the 11-24 pattern), and whether the head also trains the token encoder is decided by the oracle test in stage 2, not by taste. No object identity, no count, no slots: a car is where the field says "occupied at t_k", a box where it says so at every t_k.

**Why not attention over time.** Memory Gym: under PPO a TransformerXL window loses to a GRU on endless tasks and wins on finite ones only with an aux reconstruction loss. Our attention is over space within one decision; time enters as ages on tokens (0–0.5 s), as the geometric memory (seconds), and as the GRU (place code). Three devices for three time scales, each the one that measured best.

**Veto check.** No map, no global pose, no racing line, no lap position; memory positions move by dead reckoning only, never corrected against the scan; everything inside the network; the critic's truth field is critic-only.

---

## 5. First falsification test (half a day, lab PC, the recorded data)

Code: extend `tools/lidar_repr_test.py` on branch `repr-test` (it already turns scans into exact points — `Geo.points` — and composes noisy odometry chains — `rel_chain`); records in `~/F1tenth/reprtest/out` on the lab (opponent: 3 × 128 sessions × 90 s, 1v1; boxes: 2 × 128 × 60 s). Same splits, same budget (20 / 12 epochs, AdamW 1e-3 one-cycle, batch 128), RSS ≤ 3.5 GB as before. New kinds:

1. **`pts5` / `pts10`** — the A/C front end: point tokens with ages (beam runs of 8 / 16), 10 time-aligned queries + 1 car query, 2 cross-attention blocks, same 21 outputs as `OppNet`. Also `pts5_noise` (trained/evaluated with the 2 cm / 0.5° odometry noise).
   Pass line on the opponent at 1.0 s (held-out sessions): median ≤ 0.34 m (`bev10`) and ≤ 0.39 m in the pace < 0.7 bin; the 6–10 m bin not worse than `bev10` (0.51 m) by more than 0.1 m (tests the far-token sparsity). Fail → the point front end has no motion advantage; fall back to B's pillar grid with point read-out.
2. **Clearance-precision probe (new, all kinds).** Targets from the records: for each decision and each k = 1..10, the signed gap from the car's realised pose at t + 0.1 k (`st`; exact execution, so this is the plan's own point) to the nearest box edge (`pose` per box, footprint half-size from the layout bank — if it is not in the record, one line in `exam_record` and a 5-minute re-record) and to the wall (`edt`, `t_origin`, `t_res` are in the record). Models: `polar_gru` (today), `bev10` with the pooled vector, `bev10` with bilinear read-out at the plan points, `pts5`. Metrics: error median / p90 overall and in the bin gap < 0.3 m.
   Pass line: `pts5` median ≤ 3 cm and ≤ 5 cm in the < 0.3 m bin. Reading: if `bev10`-bilinear comes within 1 cm of `pts5`, the precision question is settled by the read-out, and grids are acceptable for the present too (choose by cost); if `bev10`-pooled is ≥ 2× worse, the draft's flatten was the loss.
3. **Hidden box — point memory arm.** `box_ptmem`: a dead-reckoned point memory (M = 1024 points, FIFO write of the newest frame subsampled 1/8 — no learned score, to test the device alone), read by the same grid-of-queries decoder as the grid memory so the recall/precision scoring is identical; noise 0 / 0.5× / 1× / 2×, trained with 1× noise.
   Pass line to be considered for C instead of the grid: recall ≥ 0.78 and precision ≥ 0.72 at 1× (the grid's numbers). Otherwise the grid stays.
4. **Latency**, batch 1 on the lab 4060 Ti as before (network only / with token build), scaled to Orin as the probe did (× 4).

Cost: ~3 h code, opponent models 3 × 15 min, clearance probe 4 × 10 min, box arms 4 × 5 min → within half a day.

---

## 6. Staged build with gates

| stage | build | gate (must be true to continue) | time |
|---|---|---|---|
| 0 | the probe above | pts ≤ bev on the car, ≤ 3 cm clearance, memory arm decided | ½ day |
| 1 | front end in `plan_rl` (point tokens, time-aligned queries, driving GRU on the ego query; no spatial memory); odometry noise in training; from scratch in the agreed mixed world to u150; Orin-proxy timing | vs scan31a at equal updates on exam3: laps/min within 30 %, early-visible box hits not higher, passes under 10 cm not more frequent than today's 10 %; forward ≤ 5 ms Orin-equivalent; bag replay of the token build with real `VescOdom` lines up with the map within 10 cm over 0.5 s | 1½ days |
| 2 | the free-space head at the plan queries (own optimizer; critic gets the truth field). **Oracle first**: true future clearances fed to the trunk for a 50-update fork | oracle: 0.6x overlap 13 % → clearly lower with passes ≥ 1.5; otherwise no head is built and the field goes to the critic only. Then the head: car position at 1 s ≤ 0.35 m in training, the same overlap fall as the oracle | 1½ days |
| 3 | spatial memory (grid 0.25 m warped, or the point memory if stage 0 said so) read by the queries; trained with odometry noise | wipe test on exam3: wiping each lap worsens recovery after a hit, side flips and box hits; re-hit rate falls vs the stage-2 policy; empty lap not slower | 1 day |
| 4 | the full run, both GPUs, ~1300 updates, exams every 25–50 on the lab | exam3 vs scan31b2 u600: lap ≤ 7.2 s, hard box ≤ 0.11; exam4 vs race33b: at-fault ≤ 0.97, passes ≥ 1.55, 0.6x overlap < 13 %; plan moves > 1 cm per 0.5 m shift of the predicted car (today 0.5–1 cm) | 3–4 days |

Each stage adds one structure and has a test that does not need the learned part to work first (oracle before head, wipe before claims). If stage 1 fails on speed, the one lever kept in reserve is a supervised **initialisation** of the token encoder from the stage-0 probe (representation init, not imitation; the plan head stays RL from scratch).

---
---

# Policy 네트워크 재설계 v2 — 표현을 원리에서 다시 (2026-11-25)

제안만 있고 만든 것도 돌린 것도 없습니다. 0.125 m 격자 초안(`net_redesign_draft.md`)을 대체합니다.

바꾸지 않는 것: 출력은 1초짜리 10점 plan(위치 + 속도), 학습은 plan대로 정확히 실행하는 PPO, 랩타임 보상, 우리 책임 접촉 비용, plan–차 겹침 비용; 네트워크 하나로 {혼자, 1대1} × {상자, 없음}; 입력은 우리 센서만(2D LiDAR 1081 빔 / 270° / 0.25° / 10 m — 코드상 policy가 1081 빔을 전부 읽고 3장을 겹침; IMU; 바퀴 주행거리)과 적분한 자기 이동; 레이싱라인·센터라인·랩 위치·전역 지도·스캔 매칭 위치추정 금지; Jetson AGX Orin, 0.1초 결정.

넘어야 할 것: 퀄리 scan31b2 u600 — 7.10 s, 상자 세션당 세게 닿음 0.11; race33b u975–u1075 — 1대1 세션당 우리 책임 접촉 0.97–0.99, 앞지름 ~1.55, 0.6배 차 상대로 우리 책임 접촉 1.84–2.19, plan과 그 차 미래의 겹침 ~13 %에서 정지.

---

## 1. 격자 계획이 왜 틀린 층의 생각이었나

초안은 "칸을 몇 cm로 하나"를 물었고 0.125 m라고 답했습니다. 설계 상수에 대한 질문이고, 그 뒤에 내놓은 세 가지(원 스캔을 격자 옆에 같이 넣기, 칸 안 위치 저장, 가까운 곳만 더 촘촘히)는 전부 그 상수가 깎아먹은 것을 다른 곳에서 메우는 방법이었습니다. 격자를 택하게 한 측정 세 가지는 사실 격자를 요구하지 않았습니다.

- 상대 차 예측이 좋아진 것(극좌표+GRU 0.50 m → 격자 10장 0.34 m)은 **우리 움직임을 지운 프레임에 시간 채널**을 준 데서 왔습니다. 지우지 않은 극좌표 묶음은 연산 12배에도 0.60 m, 지우는 법을 스스로 배워야 하는 극좌표+GRU는 0.50 m. 격자는 그 보정을 담은 그릇이었을 뿐, 원인이라는 증거는 없었습니다.
- 숨은 상자 결과(주행거리로 옮기는 격자 기억 재현율 0.95 대 GRU 0.16–0.29 = 기억 없음)는 **저장한 내용이 어디에 사느냐**의 문제입니다: 매 0.1초의 이동량으로 함께 옮겨지는 좌표 위에. 칸의 성질이 아니라 "좌표를 가진 내용"의 성질입니다.
- 지금 policy는 상자를 중앙값 30 cm, 10 %는 10 cm 아래로 지나가고, 센서는 1–4 cm로 잽니다(UST-10LX 사양 ±40 mm; 빔 간격 0.25° = 5 m에서 2.2 cm, 10 m에서 4.4 cm; 0.22 m 상자는 5 m에서 빔 10개). 12.5 cm 칸은 센서가 주는 것의 한 자릿수를, plan이 바로 읽는 양에서, 버립니다.

맞는 질문은 이것입니다: **plan 머리가 무엇을 읽고, 그 정보는 어느 정밀도로 존재하며, 각 정보가 그것이 관계된 plan 점까지 어떻게 닿는가.**

1. plan의 결정 변수는 연속량입니다: plan 점 k 자리의, 시각 t_k(0.1 k 초)의 빈 공간. 정확한 좌표(스캔 점, 적분한 우리 경로, 다른 차의 움직임)의 함수입니다.
2. 정보의 정밀도는 시간에 따라 다릅니다:
   - 가장 새 프레임의 점: 1–4 cm(센서 한계);
   - 0.1–0.5초 전 점을 지금 틀로 옮긴 것: 주행거리가 0.1초마다 ~2 cm, 0.5°를 더함(시험에 넣은 잡음 모형; 거기서는 손실 없음) → 0.5초에 4–5 cm, 1°, 즉 5 m에서도 몇 cm;
   - 한 바퀴 전(6–7초, 60–70번 적분)에 본 것: 17 cm, 4° 무작위 걸음 → 5–10 m 거리에서 0.4–0.7 m. 시험이 그대로 재어 줬습니다: 참 이동이면 0.95, 잡음 1배 0.78, 2배 0.49.
3. 그러니 모든 것에 한 가지 정밀도를 쓰는 것은 두 번 틀립니다: 0.125 m 격자는 현재에는 10배 거칠고 과거에는 가능한 것보다 2–5배 촘촘하며, 비용은 넓이/칸² 로 늡니다. 현재는 정확한 좌표를 그대로 두고, 과거는 주행거리가 허용하는 정밀도로 저장하고, 읽기는 **정확한 (x, y, t)에 대한 query**로 — 내용이 어떻게 저장돼 있든 읽기에는 해상도가 없습니다.
4. 단위가 "좌표와 나이를 가진 점"이고 읽는 쪽이 attention이면 개수 무관은 저절로 따라옵니다: 상자·차·벽이 몇 개든 token 수가 바뀔 뿐 모듈은 안 바뀝니다.

이것이 2024년 이후 이 분야가 옮겨간 틀입니다: 원 좌표를 MLP로 임베딩한 sparse token, 프레임 사이는 명시적 SE(2)/SE(3) 변환으로 옮기고, 후보 궤적 점에 anchor를 둔 planning query가 읽습니다. 밀집 BEV는 센서가 카메라라서 여러 시야를 합칠 자리가 격자밖에 없을 때 남아 있습니다. 우리는 2D LiDAR입니다: 점은 이미 있습니다.

---

## 2. 최근 연구가 하는 것 (직접 연 링크)

제가 연 것:
- SparseDrive — https://github.com/swc-17/SparseDrive (코드 2024-06, ICRA 2025; V2 2026-04-01). 밀집 BEV 없이 sparse 인지(검출 anchor 900 + 지도 polyline 100, instance 기억 큐) → 예측·계획 병렬; 9.0 FPS 대 UniAD 1.8, 학습 20 h 대 144 h.
- DiffusionDrive — https://github.com/hustvl/DiffusionDrive , https://arxiv.org/abs/2411.15139 (CVPR 2025 highlight). 궤적 anchor 20개를 2단계 denoising으로 다듬고, 궤적 좌표로 장면을 샘플링; 4090에서 45 FPS; NAVSIM PDMS 88.1.
- Sparse4D v3 — https://arxiv.org/html/2311.11722 : instance 900개 중 600개를 시간 너머로 끌고 감, 디코더 6층, 19.8 FPS; anchor는 자기 자세와 속도로 다음 프레임으로 옮김; anchor의 원 좌표를 작은 MLP로 임베딩해 내용 feature와 **이어 붙여**(concat) attention에서 위치가 씻겨 나가지 않게 함.
- StreamPETR — https://arxiv.org/html/2303.11926 : 기억 큐 4프레임 × 상위 256 객체; 저장한 중심을 자기 자세 변화 E_{t-1}^t로 변환(정지 가정; 속도 보정은 효과 없었다고); Motion-aware LayerNorm이 (자기 변환, 속도, Δt)로 query를 변조; 기억은 ~1초 넘으면 포화(표 7). 위의 2024–26 sparse 플래너들의 기반.
- Point Transformer V3 — https://arxiv.org/html/2312.10035v2 (CVPR 2024): 점을 공간 채움 곡선으로 정렬해 1024점 패치 안에서 attention; 상대 위치 인코딩(전방 시간의 26 %)을 버리고 sparse conv 위치 인코딩; 4090에서 nuScenes 한 장 44 ms / 1.2 GB. 2D 스캔은 빔 순서가 곧 그 정렬입니다.
- Memory Gym — https://arxiv.org/abs/2309.17207 (JMLR 2025), https://github.com/MarcoMeter/episodic-transformer-memory-ppo : PPO에서 과거 관측에 대한 TransformerXL은 유한 과제에서, 관측 복원 보조 손실이 있을 때만 GRU를 이기고, 끝없는 과제에서는 GRU가 크게 이김(정규화 점수 0.98 대 0.39). **시간**에 대한 attention은 PPO의 기억 장치로 채택할 것이 아니고, 한 결정 안의 **공간**에 대한 attention은 다른 이야기입니다.

조사 보조가 연 것(그들의 읽기에서 인용):
- SparseDriveV2 — https://arxiv.org/html/2603.29163v1 (2026-03): 생성 대신 경로 어휘(1024) × 속도 프로파일 어휘(256)를 채점; 경로–장면 상호작용은 deformable 집계가 cross-attention보다 좋았음; PDMS 92.0. 우리 출력(경로 + 속도 프로파일)과 같은 분해.
- iPad — https://arxiv.org/html/2505.15111v1 : 후보 64개, 각 후보의 네 모서리가 deformable attention의 참조점, 4번 반복; 4090에서 43 ms; 과거 프레임 없이 PDMS 91.7.
- UnO — https://arxiv.org/html/2406.08691v1 (CVPR 2024 oral): LiDAR 5–6장 → encoder → **임의의 (x, y, z, t)에 질의하는 암묵적 decoder**, 미래 스캔으로 자기 지도(빔 경로는 비어 있음, 반사점 0.1 m 안은 차 있음), 3초 지평, 종류·개수 무관. 해상도 없는 읽기의 본보기.
- CaRL — https://arxiv.org/html/2504.17838 (CoRL 2025): 대규모 PPO의 가장 가까운 선례(BEV 256×256, 3억 샘플, 미니배치 16 384, 추론 8 ms). 직접 제어 출력이라 우리 형식과는 다름.
- 최근 학습 레이싱: TinyLidarNet https://arxiv.org/html/2410.07447v1 (원 1081빔에 1D conv, 실차, 모방; 상대는 정지 장애물로); End2Race https://arxiv.org/html/2509.16894v1 (360빔 + GRU, 모방, 시뮬, 수동적 상대, 앞지름 59 %); AID-4 https://github.com/thdrbgur112/AID-4 (108 섹터 최소값, 기억 없는 MLP, Isaac Lab 8192 env, 동적 상대 미해결); Steiner 등 https://arxiv.org/html/2510.26040 (섹터 10개, TD3, 실차 1.5 m/s); Sony GT7 비전 에이전트 https://arxiv.org/html/2504.09021v1 (conv → GRU → MLP, "RNN을 빼면 완전 실패"). 2024–26년에 우리 속도로 움직이는 차 상대로 학습 LiDAR policy를 달리는 팀은 없고, 실차 LiDAR policy는 모두 스캔을 일찍 압축(섹터, 1D conv flatten)합니다 — 우리 움직임·정밀도가 사라지는 바로 그 자리.
- RL에서 2D 스캔에 attention을 쓴 것은 섹터 token ~30개(de Heuvel https://arxiv.org/html/2310.19670v2 , DDPG, Turtlebot) 또는 시점 token 10개(Transqer https://arxiv.org/html/2502.19892v1 , DQN, Orin Nano + UST-10LX에서 1.2 ms)뿐. 빔 ~1000개에 attention을 건 RL policy는 발표된 것이 없습니다 — 아래 A/C의 열린 위험이고 1단계 관문의 이유.
- RLBenchNet https://arxiv.org/html/2505.15040v1 : PPO에서 TransformerXL 기억은 GRU의 ~10배 GPU 메모리(1.77 대 0.19 GB) — 순환 부분을 GRU로 두는 또 하나의 이유.

---

## 3. 세 가지 구조

셋에 공통(바꾸지 않는 부분): plan 머리는 20개 숫자 → `plan_steps` 해석 → 10점 + 속도; 자기 상태 입력은 지금처럼(IMU, 마지막 명령, 지난 plan, 속도 추정 머리); GRU(128)가 PPO가 이미 배운 암묵적 위치 부호(0.32 m로 복원)를 들고 감 — Memory Gym과 GT7 에이전트가 말하듯 순환 부분은 GRU; critic은 지금처럼 특권 입력.

### A — 점/query policy: 점은 점으로, 기억은 점의 집합으로

**Token.** 최근 K = 5 프레임(0, 0.1, …, 0.4초 전)의 모든 반사점을, 적분한 SE(2) 이동량의 연쇄로 지금 차 틀의 점 (x, y)로 옮김(시험 코드의 `rel_chain` + `Geo.points`가 이미 하는 일 — 격자 없음), 나이 τ와 거리 r을 붙임. 점별 feature = MLP(Fourier(x, y) 척도 0.05 / 0.2 / 1 / 5 m, τ, r) → 64(Sparse4D의 "원 좌표를 작은 MLP로"; 2 cm와 5 m가 둘 다 구분되게 척도를 둠). 점을 연속한 빔 묶음으로 모음(새 프레임 8빔 → token 135개; 옛 프레임 16빔 → 각 68개): token = 점 feature의 max-pool + 묶음의 평균 위치. N ≈ 400 token, 폭 96. 빔 순서로 묶는 것은 PTv3 정렬의 2D 판이고, 개수 무관(상자가 세 묶음에 걸치면 token 셋, 차는 열, 벽은 쉰)이며 token 안에 정확한 좌표가 남습니다. 재시작 전 프레임은 그냥 없음(N이 변해도 됨).

**Query.** plan 시각 query 10개 q_k = 학습 임베딩 + t_k의 시간 인코딩 + anchor a_k의 Fourier 위치 인코딩, anchor = 지난 plan의 점 k를 지금 틀로 옮긴 것(`PlanObs2.plan_xy`에 이미 있음). 더하여 자기 상태 query 1개(속도 추정, IMU, 마지막 명령). key에는 **상대** 위치 Fourier(token − a_k)와 나이를 실어, 각 query의 attention이 그대로 "t_k에 내가 있을 자리 근처에 무엇이 있고 어떻게 움직였나"를 묻게 함 — SparseDrive / Sparse4D / iPad의 anchor-as-PE 패턴.

**Attention 패턴.** [query → token cross-attention, query 11개 사이 self-attention, FFN] 블록 2개, 폭 128, 헤드 4, pre-LN. **token 사이 self-attention 없음**: 비용 O(Q·N), Perceiver/DETR decoder 모양, O(N²)가 아님. 자기 상태 query의 출력이 주행 GRU로 들어가고 GRU 상태가 모든 query에 더해짐(위치 부호가 plan까지 닿음). plan query마다 (회전, 속도) → 20개; 0으로 초기화해 새 네트워크는 anchor(직진/지난 plan)대로 달림.

**기억(A의 고유 부분): 크기 제한된 점 기억.** 기억 token M = 256개, 각각 (x, y, 나이, feature). 결정마다: 위치를 SE(2) 이동량으로 옮김(StreamPETR/Sparse4D 전파; 적분만 — **지금 스캔에 다시 맞추지 않음**, 그건 스캔 매칭이라 금지), 나이 += 0.1, 창(−20 … +20 m)을 벗어나면 버림, 지금 프레임 token 중 쓰기 점수 상위가 점수 최하위 자리를 대체(StreamPETR의 top-K 밀어넣기). plan query가 같은 상대 PE로 기억 token에 cross-attention.

**cm 정밀도:** 현재는 예 — token이 정확한 좌표를 갖고 query가 상대 위치를 직접 읽음. **움직이는 차:** 나이 붙은 5프레임 꼬리는 격자 시험(0.34 m)과 같은 증거를 정확한 좌표로 가진 것; 정확함이 더 사 주는지는 시험이 답함. **숨은 상자:** 쓰기 점수에 달렸고 — 그게 A의 약점: PPO만으로는 학습 쓰기 점수가 배울 것이 없음(잠재 GRU 실패의 새 옷); §4의 사후 타깃이 필요. **객체 개수:** 구조상 무관.

**비용.** 결정당 점별 MLP ≈ 56 MFLOP(5 400점) + cross-attention 블록 둘 ≈ 60 MFLOP(K/V 투영이 대부분) + 기억 읽기 ≈ 20 MFLOP ≈ **0.14 GFLOP**(초안 BEV 0.37 + 0.5 GFLOP). Orin: 커널 실행 비용 위주, TensorRT FP16으로 1–2 ms; token 만들기는 gather + SE(2) 변환, 래스터 커널 없음. 1000 env 학습: 롤아웃 제어 걸음당 4070에서 +≈15 ms(지금 sim ≈ 30 ms); PPO 에포크 256 k 결정 × 0.14 GFLOP × 3 × 4 ≈ 430 TFLOP ≈ 35 s → update는 지금의 2–2.5배; token은 저장한 스캔에서 미니배치 안에서 다시 만들고(걸음마다 스캔 한 장은 이미 저장; 옛 4장은 앞 행에서 가져옴), 주행거리 이동량 저장(걸음당 3개), 기억 token은 chunk 시작에서만 저장(`--mem-chunk` 방식), env당 256 × (3 + 96) × fp16 ≈ 50 kB.

**실패 양상.** (i) 이 token 수로 PPO 처음부터 attention을 학습한 2024–26 선례 없음(1단계 관문). (ii) 쓰기 정책: 자기 지도 타깃 없이는 무엇을 저장할지 못 배움. (iii) 빔 묶음은 먼 물체에 token을 적게 줌: 9 m의 차는 묶음 ~2개 — 시험의 6–10 m 구간이 말해 줌. (iv) token 수가 변하면 CUDA graph 캡처가 번거로움(고정 N으로 채우고 mask).

### B — 밀집 BEV를 제대로: pillar, 옮기는 conv 기억, 연속 읽기

격자를 덧대기가 아니라 정직하게 만들려면:
- **점유가 아니라 pillar.** 비어 있지 않은 칸(0.125 m, −4 … +12 × ±6 m)마다 그 안의 점들을 칸 안 정확한 offset과 나이로 점별 MLP → max-pool(PointPillars식, 학습으로; 손으로 쓴 칸 안 offset이 아님). 그러면 칸은 양자화기가 아니라 바구니.
- **읽기는 pooled 벡터가 아니라 plan 점에서.** plan 머리가 anchor a_k 자리의 feature 지도를 샘플링(bilinear) 하거나 암묵적 decoder(UnO)로 읽음 — 초안의 정밀도 손실은 칸만큼이나 flatten → 256 벡터에 있었음.
- **기억** = 매 걸음 SE(2) 이동량으로 `grid_sample` 하는 conv-GRU 숨은 격자, 0.25 m 칸(시험의 장치, 0.95 / 0.78), 주행거리 잡음으로 학습.
- **시간** = K 프레임을 채널로, 지금 틀로 옮겨서(시험에서 이긴 입력).

**Orin 비용으로 cm 정밀도가 되나?** 정밀도: pillar + 점 읽기로 아마 됨 — §5 시험이 잼. 비용이 정직한 문제: 결정당 0.4–0.9 GFLOP; 래스터는 지금 PyTorch 조각으로 Orin에서 5–10 ms(합친 커널이 필요; 그러면 ~1 ms); 비용 ∝ 넓이/칸² 이라 사각지대 사례에 필요한 뒤·옆 범위를 코앞과 같은 값으로 치름; 90 %가 빈 칸인데 다 convolution; 옮길 때마다 bilinear로 기억이 번짐(한 바퀴 70번); 범위와 K가 고정. 건전한 설계지만 카메라 시대의 것이고, 모든 매개변수가 해상도 상수.

**실패 양상.** 옮기는 기억의 번짐; 사각지대에 필요한 범위에서의 비용; 두 코드 경로(학습기와 Orin의 래스터 커널)를 비트 단위로 맞춰야 함.

### C — 정밀도를 맞춘 혼합: 현재는 점, 과거는 주행거리로 옮기는 격자, 둘 다 연속 query로

앞단과 query는 A와 같음. 기억은 B처럼, 다만 과거가 실제로 가진 정밀도로: 주행거리로 옮기는 **0.25 m** 격자, −6 … +14 × ±8 m(80 × 64 칸), 16채널, 새 프레임의 token이 씀(token 위치의 칸에 scatter, 학습 쓰기), gate 갱신, 매 걸음 SE(2) 이동량으로 옮김, 시험의 주행거리 잡음으로 학습. 같은 plan query가 읽음: 격자의 비어 있지 않은 칸이 같은 cross-attention의 token(칸 중심, 나이, feature), 같은 상대 PE. 0.25 m는 한 바퀴 전 기억의 0.4–0.7 m 주행거리 드리프트 아래라 잃는 것이 없고, 격자의 쓰기는 기하학적(학습 선택 없음)이라서 모든 학습 잠재 기억이 실패한 시험에서 이것만 됐습니다.

**비용.** A의 0.14 GFLOP + 16채널 gate 격자 ≈ 0.05–0.1 GFLOP ≈ **0.2 GFLOP**; chunk 시작 상태 80 × 64 × 16 × fp16 = env당 164 kB(1000 env, chunk 4개면 0.66 GB).

**실패 양상.** A와 같은 PPO-attention 위험; 격자 기억의 긴 지평은 주행거리 한계(1배 잡음에서 재현율 0.78은 우리가 잰 천장이지 설계 결함이 아니고, 실차의 바퀴 미끄러짐은 무작위 걸음 모형이 안 덮는 체계 오차 — 계획에 bag 재생 점검을 넣음).

---

## 4. 추천: C, A의 점 기억은 측정으로 겨루는 대안

**결정적 논거.** plan 머리는 정확한 (x, y, t_k)의 빈 공간을 읽습니다. token이 정확한 좌표를 갖고 읽기가 그 좌표에 대한 query인 설계만이 해상도 상수가 하나도 없고, 그 비용은 Q = 11, N ≈ 400의 O(Q·N)으로 센서 정밀도를 맞출 어떤 격자보다 쌉니다. 과거에는 정확함이 없고 — 주행거리가 0.2–0.7 m로 정함 — 시험에서 된 유일한 기억 장치(0.95 / 0.78 대 모든 학습 잠재 기억 0.16–0.29)는 기하학적인 것: SE(2) 이동량으로 옮겨지는 내용과 학습 선택이 필요 없는 쓰기. C는 각 정보를 그것이 실제로 가진 정밀도에 두고 둘 다 같은 query로 읽습니다. A의 점 기억은 격자와 같을 수도 있지만, 이미 되는 장치를 대체하려면 같은 시험에서 먼저 증명해야 합니다.

**표현을 무엇이 가르치나.** PPO만으로는 잠재 기억에 plan이 읽을 수 있는 것을 가르치지 못했습니다(몸통 가중치 0.05 대 0.55; 드문 사건). 11-24 합의에 이미 자기 optimizer를 가진 예측 머리가 있습니다. 그것의 개수 무관 일반화는 **하나**의 타깃, plan이 읽는 양 그 자체: 각 plan query에서 (a_k, t_k)의 빈 공간 — t_k에 가장 가까운 물체(벽, 상자, 차)까지의 부호 있는 간격과 anchor 주변 표본점의 점유 — 시뮬에서는 사후의 참값, 실차에서는 우리 미래 스캔(UnO의 자기 지도). 자기 optimizer; 출력(간격 10개 + 확신)은 분리해(detached) 몸통에 들어가고(11-24 방식), 이 머리가 token encoder까지 가르칠지는 2단계의 oracle 시험이 정하고 취향으로 정하지 않습니다. 객체 정체도 개수도 칸도 없음: 차는 장이 "t_k에 차 있다"고 하는 곳, 상자는 모든 t_k에 그렇게 말하는 곳.

**왜 시간에 대한 attention이 아닌가.** Memory Gym: PPO에서 TransformerXL 창은 끝없는 과제에서 GRU에 지고 유한 과제에서는 보조 복원 손실이 있어야만 이김. 우리 attention은 한 결정 안의 공간에 대한 것이고, 시간은 token의 나이(0–0.5초), 기하학적 기억(몇 초), GRU(위치 부호)로 들어갑니다. 세 시간 척도에 각각 측정에서 가장 좋았던 장치 하나씩.

**금지 사항 점검.** 지도·전역 자세·레이싱라인·랩 위치 없음; 기억 위치는 적분으로만 움직이고 스캔에 맞춰 고치지 않음; 전부 네트워크 안; critic의 참값 장은 critic만.

---

## 5. 첫 반증 시험 (반나절, 연구실 PC, 이미 기록한 데이터)

코드: 브랜치 `repr-test`의 `tools/lidar_repr_test.py` 확장(스캔을 정확한 점으로 바꾸는 `Geo.points`, 잡음 주행거리 연쇄 `rel_chain`이 이미 있음); 기록은 연구실 `~/F1tenth/reprtest/out`(상대 차: 3 × 128 세션 × 90 s 1대1; 상자: 2 × 128 × 60 s). 같은 나눔, 같은 예산(20 / 12 epoch, AdamW 1e-3 one-cycle, batch 128), RSS ≤ 3.5 GB. 새 종류:

1. **`pts5` / `pts10`** — A/C 앞단: 나이 붙은 점 token(빔 8 / 16 묶음), 시각 맞춘 query 10개 + 차 query 1개, cross-attention 블록 2개, `OppNet`과 같은 출력 21개. `pts5_noise`(2 cm / 0.5° 주행거리 잡음으로 학습·평가)도.
   통과선(따로 둔 세션, 상대 차 1.0초): 중앙값 ≤ 0.34 m(`bev10`), 페이스 < 0.7 구간 ≤ 0.39 m; 6–10 m 구간이 `bev10`(0.51 m)보다 0.1 m 넘게 나쁘지 않음(먼 token 희소성 검사). 실패 → 점 앞단은 움직임에서 이점이 없음; B의 pillar 격자 + 점 읽기로 후퇴.
2. **간격 정밀도 시험(새로, 모든 종류).** 타깃은 기록에서: 결정마다 k = 1..10에 대해 t + 0.1 k의 실제 차 자세(`st`; plan대로 정확히 실행했으니 plan의 점 그 자체)에서 가장 가까운 상자 모서리(`pose`의 상자 자세, 크기는 배치 은행에서 — 기록에 없으면 `exam_record` 한 줄과 5분 재기록)와 벽(`edt`, `t_origin`, `t_res`가 기록에 있음)까지의 부호 있는 간격. 모형: `polar_gru`(지금), pooled 벡터의 `bev10`, plan 점에서 bilinear로 읽는 `bev10`, `pts5`. 지표: 오차 중앙/p90, 전체와 간격 < 0.3 m 구간.
   통과선: `pts5` 중앙값 ≤ 3 cm, < 0.3 m 구간 ≤ 5 cm. 해석: `bev10`-bilinear가 `pts5`와 1 cm 안이면 정밀도 문제는 읽기로 해결된 것이고 현재에도 격자가 허용됨(비용으로 선택); `bev10`-pooled가 2배 넘게 나쁘면 초안의 flatten이 손실이었음.
3. **숨은 상자 — 점 기억 arm.** `box_ptmem`: 적분으로 옮기는 점 기억(M = 1024점, 새 프레임을 1/8로 줄여 FIFO로 씀 — 장치만 보기 위해 학습 점수 없음), 격자 기억과 같은 query 격자 decoder로 읽어 재현율/정밀도 채점이 동일; 잡음 0 / 0.5 / 1 / 2배, 1배로 학습.
   C에서 격자 대신 고려될 통과선: 1배에서 재현율 ≥ 0.78, 정밀도 ≥ 0.72(격자의 수치). 아니면 격자 유지.
4. **지연 시간**, 연구실 4060 Ti batch 1(망만 / token 만들기 포함), 시험처럼 Orin으로 환산(× 4).

비용: 코드 ~3 h, 상대 차 모형 3 × 15분, 간격 시험 4 × 10분, 상자 arm 4 × 5분 → 반나절 안.

---

## 6. 단계별 구축과 관문

| 단계 | 만들 것 | 관문(다음으로 가려면 참이어야 할 것) | 시간 |
|---|---|---|---|
| 0 | 위 시험 | 차에서 pts ≤ bev, 간격 ≤ 3 cm, 기억 arm 결정 | ½일 |
| 1 | `plan_rl`에 앞단(점 token, 시각 맞춘 query, 자기 상태 query 위의 주행 GRU; 공간 기억 없음); 학습에 주행거리 잡음; 합의된 혼합 세계에서 처음부터 u150까지; Orin 대용 시간 | 같은 update의 scan31a 대비 exam3: 분당 랩 30 % 안, 일찍 보이는 상자 접촉 안 늘고, 10 cm 아래 통과가 지금의 10 %보다 안 잦음; 전방 ≤ 5 ms Orin 환산; 실제 `VescOdom`으로 bag 재생한 token 만들기가 0.5초 동안 지도와 10 cm 안에서 맞음 | 1½일 |
| 2 | plan query의 빈 공간 머리(자기 optimizer; critic은 참값 장). **oracle 먼저**: 참 미래 간격을 몸통에 넣고 50 update 가지치기 | oracle: 0.6배 겹침 13 % → 분명히 낮아지고 앞지름 ≥ 1.5; 아니면 머리는 안 만들고 장은 critic만. 그 뒤 머리: 학습 분포에서 1초 차 위치 ≤ 0.35 m, oracle과 같은 겹침 하락 | 1½일 |
| 3 | 공간 기억(0.25 m 옮기는 격자, 또는 0단계가 그렇다면 점 기억)을 query가 읽음; 주행거리 잡음으로 학습 | exam3 지우기 시험: 매 바퀴 지우면 접촉 뒤 복구·쪽 바꿈·상자 접촉이 나빠짐; 다시 박음 비율이 2단계 policy보다 낮음; 빈 트랙 랩 안 느려짐 | 1일 |
| 4 | 전체 run, GPU 둘, ~1300 update, 연구실에서 25–50마다 시험 | exam3 대 scan31b2 u600: 랩 ≤ 7.2 s, 상자 세게 ≤ 0.11; exam4 대 race33b: 우리 책임 ≤ 0.97, 앞지름 ≥ 1.55, 0.6배 겹침 < 13 %; 예측 차를 0.5 m 옮기면 plan이 1 cm 넘게 움직임(지금 0.5–1 cm) | 3–4일 |

단계마다 구조 하나를 더하고, 학습된 부분이 먼저 되지 않아도 되는 시험을 둡니다(머리보다 oracle 먼저, 주장보다 지우기 먼저). 1단계가 속도에서 실패하면 남겨 둔 지렛대 하나는 0단계 시험으로 token encoder를 **초기화**하는 것(표현 초기화일 뿐 모방이 아님; plan 머리는 처음부터 RL).
