# Driving console: visual design

**2026-09-12.** The console's look was rebuilt around how the major simulators present a scene, so
that someone who has used Isaac Sim, CARLA or an AV stack's replay tool finds nothing to relearn. The
references below were read for their *conventions*, not copied: our scene is a 1:10 track with duct
barriers, and the design has to serve that.

## References, and what was taken from each

| reference | convention adopted here |
| --- | --- |
| [Isaac Sim / Omniverse UI](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/gui/reference_user_interface.html) | graphite chrome with **no hue in the panels**, a large central viewport with docked panels either side, small monospace viewport chips (map / run in the corner, state badge), a ground plane with a fine and a coarse lattice that fades with distance, one calm selection accent, NVIDIA-style green reserved for "running" and the start action |
| [CARLA (Unreal Engine)](https://carla.readthedocs.io/en/latest/core_actors/) spectator and [sensor views](https://carla.readthedocs.io/en/latest/ref_sensors/) | a lit scene with a real sky instead of a void: directional sky gradient, horizon haze that the fog fades into, warm key light with cool sky ambient and a faint fresnel rim; LiDAR returns drawn as small soft discs coloured by what they hit |
| [NVIDIA Alpamayo](https://github.com/NVlabs/alpamayo) / DRIVE-style trajectory replay ([blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/)) | the planned trajectory as a **translucent flat ribbon with a bright core** rather than a wire, a ring on the ground under the ego vehicle, trails that fade out behind, neutral vehicle paint so the per-vehicle tint (ego / rival / collided) *is* the vehicle colour |

## Tokens

Defined once in `f1sim/viewer/console/theme.py` and reused by the GL scene (`gl_scene.py` carries
the matching `CLEAR_RGB` / `HORIZON_RGB` / `ZENITH_RGB`).

| token | value | used for |
| --- | --- | --- |
| `bg.window` / `bg.panel` / `bg.card` / `bg.raised` | `#161616` / `#1c1c1c` / `#222222` / `#2c2c2c` | four steps of graphite, never blue |
| `bg.viewport` | `#121316` | the viewport's own ground haze (`CLEAR_RGB`) |
| `line` / `line.strong` | `#303030` / `#454545` | hairlines, gauge tracks |
| `text.0` / `.1` / `.2` | `#e6e6e6` / `#a9a9a9` / `#747474` | primary, secondary, captions |
| `accent` / `accent.deep` | `#5aa9ff` / `#22415f` | selection, focus, measured value arcs |
| `primary` / `primary.line` | `#3d6f0b` / `#76b900` | the start button and the running badge only |
| `ego` | `#59e6ff` | the watched car: ring in the scene, marker in the policy panel |
| `rival` / `warn` / `danger` | `#f2a044` / `#f0a030` / `#ff5c5c` | a rival car and commanded values; preparing; collision and over-cap |

Type: body 12 px, labels 11 px, section headings 11 px with letter-spacing, metrics in
`DejaVu Sans Mono`. Radii 3 px on controls, 4 px on cards. Header 44 px.

## Scene

| element | before | now |
| --- | --- | --- |
| background | flat clear colour, black above the horizon | sky gradient by view direction (zenith `ZENITH_RGB` to horizon `HORIZON_RGB`), fog fades to the horizon tone, ground haze below |
| ground | light grey plate, 1 m lattice only | asphalt-toned apron (`0.135`) that dissolves into a backdrop disc of the same tone; 1 m and 5 m lattices, each fading with pixel footprint |
| duct barriers | mirror-like, ±26 % corrugation bands | satin (`metal` = 48 / 0.55), ±11 % bands |
| lighting | grey hemisphere, single key | cool sky ambient + warm ground bounce, warm key, fresnel rim |
| cars | tint multiplied a blue-grey paint so every car was teal | neutral paint; ego drawn true-colour with a cyan ring on the ground, rival amber, collided red, others dimmed |
| plan | 6 px wire | 0.11 m translucent ribbon + 2.5 px core, speed-coloured; tracker prediction stays a thin line |
| LiDAR | hard 22 px discs, saturated | 12 px soft discs, by hit type (duct amber, tall object magenta, floor teal, car yellow) |
| trails | constant alpha | fade out behind the car |
| chips | rounded cards | translucent graphite slabs with a hairline border, monospace |

## Instruments

The dash strip is an instrument cluster, not a dashboard prop: a flat ring gauge for speed (track,
measured sweep in the accent, over-cap zone red, commanded marker inside the ring, reading inside the
dial), a centred steering bar (fill from centre to the measured angle, marker for the commanded
angle, left steer to the left), and the g-g trace against the friction circle. Columns are separated
by hairlines. Everything a value stands for is unchanged from the previous drawing.

## The map card (2026-09-12)

The left panel's 맵 card was a group combo over five named groups and a list of **loader names** —
every direction, obstacle family and placement seed as its own row, two hundred rows of
`real:korea_2026_competition+rlobs213~mir~rev`. That is a list of runs, not a list of maps, and the
same map appeared in it twenty times.

| element | before | now |
| --- | --- | --- |
| groups | 기본 평가셋 / 장애물 (상자·궤짝·드럼) / 기본 학습셋 / 이전 실험 재현 (격자 장애물) / 전체 카탈로그 | 학습 / 검증 / 내 환경, each with a one-sentence hint saying what the split *is* |
| rows | one per loader name | one per **base map**: display name, the id beside it in grey, family chip |
| search | inside the selected group | the whole catalogue, so a map in neither split (another racetrack, a gym map, any generator seed) is reached by typing |
| direction | part of the name (`~rev`, `~mir~rev`) | 방향 segmented buttons: 정방향 / 역방향 / 거울 / 거울+역방향 |
| obstacles | part of the name (`+obs`, `+rlobs`, `+pinch`, `+props`) and a group of its own | 장애물 combo: 없음 / 가장자리 / 주행선 위 / 좁아짐 / 입체, narrowed to what the track's loader can carry |
| obstacle seed | part of the name, fixed at catalogue time | 시드: 무작위 (default) with 다시 뽑기, or 고정 N |
| what is selected | the loader name | the scenario in Korean (`Blackbox 2022 #1 · 역방향 · 주행선 위 (시드 44)`) with the spec under it, copyable |

Defaults are 정방향 / 없음 / 무작위, so **choosing a map is enough to start**. 무작위 does not draw in
the GUI: the worker draws one seed from the session seed when the session is built, and the facts
strip in the header then shows the concrete scenario — `real/bb22-1@rev#line:4417` — so a placement
worth keeping can be typed back in. 다시 뽑기 changes the session seed, which is a different map and
therefore a restart rather than a live command.

The training page's 학습할 맵 card is the same idea with checkboxes: the three groups as tabs, the
same 방향 / 장애물 / 시드 policy applied to whatever is ticked, and 학습 셋 전체 as one button that
emits `--tracks train` — the curated list itself, not a reconstruction of it. Above it, 지금 돌고 있는
학습 gives every training process a summary card (recipe, tracks, race size, controller arm, init
checkpoint and its short sha, lr → lr-end, total steps, elapsed and ETA, W&B link, log path), all
read from the argv the process was started with and the log it already writes.

See [Tracks](tracks.md) for the ids, the grammar and the splits.

## 기본 vs 없음 in the 장애물 selector (2026-09-15)

The 맵 card's 장애물 combo had one entry for "nothing added": `없음`. That is true of a measured
floor or a generated layout, which carry no obstacles of their own. It is false of an editor scene,
which carries whatever its author placed — so picking 없음 on a custom map produced a map full of
boxes. The user said so:

> 맵 고르고 장애물 정도 여부 선택할 때 없음을 선택하면 기본 맵이 되는데, 내가 커스텀해서 장애물을
> 놨으면 그 맵 자체가 기본으로 나와서 장애물 없음이라는 말과 안맞아.

One word was doing two jobs, so it is two entries now.

| entry | id | what it builds | label on a scene with 3 placed obstacles |
| --- | --- | --- | --- |
| 기본 | *(none)* | the map as authored | `기본 (배치된 장애물 3개)` |
| 없음 | `#bare` | the placed obstacles removed, walls only | `없음 (배치 장애물 제거)` |
| the families | `#edge:*` … | added **on top** of whatever the map has | unchanged |

On a map with nothing placed the two are the same track, so 기본 carries no count and 없음 is listed
**greyed** with `이 맵은 배치 장애물이 없음 — '기본'과 같습니다.` in its tooltip. Greyed rather than
hidden: "this map has nothing on it" is an answer, and removing the entry would leave 기본 looking
like the only thing there is. Switching to such a map while 없음 is selected falls back to 기본
rather than leaving a selection that means nothing.

Beside the combo is **배치 장애물 먼저 제거**, enabled only when both halves of it are true — the map
has placed obstacles, and a family is selected. It is the `+bare` composition: `scene:hall+bare+hard3`
is "the author's boxes taken off, then the hard patterns put on", which is how a custom scene is used
as a bare track for a family. With 기본 or 없음 selected the box is disabled and simply shows the
state (unchecked / checked), because there it *is* the selection.

The hint under the row changes with the choice: 기본 says the map is used as authored, 없음 says what
it removes, and a family adds the one sentence the old label contradicted — *장애물 종류는 맵이 이미
가진 것 위에 더합니다.* The session header names what was actually built (`장애물 없음`, or
`장애물 기본 (3개)`), from the worker's facts rather than from the control's selection.

The 학습 page's track picker is the same vocabulary as checkboxes — 기본 and 없음 are two boxes, and
ticking both makes two variants of every selected map, which is what that row has always meant.

One loader change came with it: `+props` used to *replace* a track's props, so `scene:hall+props3`
quietly threw away the author's boxes as well. It adds now, and places the new ones clear of the old.
Only an editor scene can arrive there carrying props, so that is the only kind of id whose meaning
moved.

## The 상대차 table (2026-09-15)

The 주행 page's 고급 설정 had one control for the other cars: a combo, `상대차 주행 방식`, with two
entries (`teacher`, `policy`). That is the right control for a two-car race and the wrong one for any
other — it says the same thing about every other car at once, and a race of three is interesting
precisely when the two other cars are *not* the same car. The user asked for the other way round:
*"대상차를 티쳐/정책 선택여부 뿐만 아니라 각 대상차에서 적용할 속도 프로파일링 및 체크포인트 등의
설정이 가능했으면 해."*

It is now a **table**: `레이스당 차량 수 - 1` rows, one per grid slot.

| column | control | notes |
| --- | --- | --- |
| 차량 | the slot number | 1 … `race_size - 1`; the learner is car 0 and is not in the table |
| 종류 | raceline 티처 / interactive 티처 / 정책 체크포인트 / 자기 자신 | `interactive` is worker 17's opponent-aware teacher: **listed and greyed, labelled `[병합 후 활성]`**, with the whole reason in its tooltip. Dropping it would make "no such kind" and "not merged yet" look the same, and greying alone reads as "not applicable to this row" |
| 체크포인트 | file picker | the button shows the basename; its tooltip is the path plus the file's **controller arm** and **memory kind**. An oracle (`opp_token`) or conditional checkpoint is refused here, in red, with the loader's own sentence — before the session is built rather than a minute into it |
| 속도 배율 | two spins, 하한–상한 | equal = a fixed multiplier; different = a band drawn per reset |
| 그립 라벨 | 참값 / 공칭 / 보수적 | the friction this teacher's speed profile assumes |
| 속도 cap | spin, `없음` at 0 | this car's own cap in m/s |
| 이벤트 | 제동 / 정지 / 차선 / 지그 | the timed events this car may be dropped into |
| 빈도 | spin | per 10 s, for this car |
| 반응형 확률 | four spins | defend / yield / line / oblivious, per race, for this car |
| 스폰 | 앞 / 뒤 / 나란히 / 무작위 | where this car starts **relative to the learner** |

Above the table: a 프리셋 combo — 기본 (티처 1.0), 학습 레시피 (0.6–1.15, 전체 이벤트, teacher + self
+ checkpoints), 느린 선두 (0.6), 막는 상대 (defend 1.0) — and 동일하게, which copies row 1 into every
other row. Under it, one line describing the whole table, which turns into the objection when the
table has one; 시작 is disabled while it does, because a checkpoint the loader will refuse is a start
that fails after a minute of loading.

Cells a kind cannot carry are disabled rather than accepted and ignored: a `자기 자신` row has no
speed profile to label and no teacher to script, so its 그립 라벨, 이벤트, 빈도 and 반응형 cells grey
out. The columns are fixed-width and the table scrolls horizontally in the 420 px panel; that is the
price of being a table rather than four identical forms, and it is what lets someone see at a glance
that row 2 is the only one with events.

The **same widget** (`viewer/console/opponent_table.py`) is the 학습 page's, behind a
`차량별 상대차 설정` switch in the 레이스·제어기 section. On, it emits one `--opp-slots` JSON and drops
the flags that table replaces; off, the recipes emit exactly the command they always did. One widget
and not two because the two pages must not be able to disagree about what a slot is — the driving
page builds a `SessionConfig` out of it, the training page builds a command line out of it, and a
table that produced something the trainer could not parse would be a second configuration language
with no way to diff it against the first.

The session header names the mix (`상대차 2x raceline, 1x policy`) and its tooltip lists one line per
slot; `SessionConfig.opponent_slots` carries the table, so 세션 저장/불러오기 round-trips it. The
viewport's rival colouring and the 정책 입·출력 panel are untouched — they read `sim.other_idx` and the
focus car, neither of which a slot table moves — and the ROS 2 link still drives car 0 only.

## Verification

Rendered headlessly with the same capture path as the README screenshots (Xvfb + llvmpipe, real
console, real worker, checkpoint `cl_origrecipe_legacy_s701` under the legacy arm). Console and
viewport test files pass (190). The GLSL for the sky and scene programs compiles in a standalone
context; `Scene` constructs headless.
