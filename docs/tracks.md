# Tracks: the catalogue, the scenario grammar, and the three splits

A **track** is a base map — a measured floor, a real circuit at 1:10, a generated layout, or
something built in the environment editor. A **scenario** is a track plus the three choices you make
about it: which way round you drive it, what is standing in the way, and where those obstacles are.

Those used to be one string. `real:korea_2026_competition+rlobs213~mir~rev` is a map, a mirroring, a
reversal, an obstacle family and a placement seed glued together, and a catalogue of two hundred of
them is a list of *runs*, not a list of maps. So there are two names now:

| | what it names | example | who reads it |
| --- | --- | --- | --- |
| id | one base map | `real/korea26` | every list a person looks at |
| scenario | a map and what is done to it | `real/korea26@rev#line:213` | `--tracks`, the console's facts strip |
| loader name | the same scenario, old spelling | `real:korea_2026_competition+rlobs213~rev` | `maps.load`, manifests, frozen benchmark files |

`f1sim.tracks` converts freely between them, and `maps.load` takes either. **Nothing was renamed on
disk and no recorded name changed meaning**: every checkpoint manifest, W&B config and frozen
benchmark suite still says `real:blackbox2022_1+rlobs44~rev`, and still means what it always did.

## Bringing your own map

A ROS `map_server` pair — `map.yaml` plus `map.pgm` or `map.png`, which is what `slam_toolbox` and
`map_saver` write — is a track. Give the path anywhere a track name is taken:

```bash
python3 -m f1sim.slam_map ~/maps/venue.yaml --preview venue.png   # measure and look before training
python3 -m f1sim.learn.evaluate CKPT --tracks ~/maps/venue.yaml   # use it by path
python3 -m f1sim.slam_map ~/maps/venue.yaml --install venue       # or catalogue it: user:venue
```

The centerline is traced from the free space and cached (`~/.cache/f1sim/centerlines`), so a map
that arrives with no `_centerline.csv` still gets a raceline, a teacher and a lap-time reference.
The console's environment editor imports the same pair (see
[environment editor](environment_editor.md)), which is where to fix a map the tracing got wrong.

## The scenario grammar

```
<track id> [ @<direction> ] [ #<obstacle>:<seed> ]

real/bb22-1                 the map as recorded
real/bb22-1@rev             driven the other way round
real/bb22-1@mir             mirrored (left and right swapped)
real/bb22-1@mir+rev         mirrored and reversed
real/bb22-1#line:44         boxes on the racing line, placement seed 44
real/bb22-1@rev#edge:3      reversed, boxes against the lane edge, seed 3
real/bb22-1#line:*          boxes on the racing line, seed drawn at run time
```

### Direction

| `@` | UI | means | loader suffix |
| --- | --- | --- | --- |
| *(none)* | 정방향 | the map as recorded | |
| `rev` | 역방향 | the same map driven the other way round | `~rev` |
| `mir` | 거울 | left and right swapped | `~mir` |
| `mir+rev` | 거울+역방향 | both, mirror first | `~mir~rev` |

`mir+rev` is one choice rather than two flags because the loader applies mirror and *then* reverse;
the other order is a different track.

### Obstacles

| `#` | UI | what stands in the way | loader suffix |
| --- | --- | --- | --- |
| *(none)* | 없음 | nothing | |
| `edge` | 가장자리 | boxes against a lane edge; the racing line stays clear | `+obs<seed>` |
| `line` | 주행선 위 | boxes **on** the racing line; the car has to plan around them | `+rlobs<seed>` |
| `pinch` | 좁아짐 | the lane closes down at a few places | `+pinch<seed>` |
| `props` | 입체 | modelled boxes, crates and a drum — finite convex sections rather than stamped cells | `+props<seed>` |

`edge` / `line` / `pinch` are rasterised into the occupancy grid: rotated rectangles of one height
with no top. `props` are modelled solids. They are different things and both are kept — every
checkpoint in the catalogue was trained against the rasterised ones, and redefining a name would
move the ground under those runs without saying so.

Not every family carries every obstacle: `rt/` (racetracks) and `scene/` (editor) understand `props`
only, which is the loader's own limit. The picker offers what the track can actually take.

### Seeds, and what 무작위 does

`#line:44` is one specific placement. `#line:*` is "this map, with boxes on the line, anywhere".
What `*` does depends on who is reading it:

* **The viewer / console.** One seed is drawn when the session starts, from the session seed, in the
  worker. The facts strip at the top of the window then shows the concrete scenario —
  `real/bb22-1@rev#line:4417` — so a placement worth keeping can be typed back in as 고정. **다시 뽑기**
  draws a new one; because a different placement is a different map, that restarts the session.
* **Training.** `--tracks 'real/bb22-1#line:*'` with `--obstacle-draws 8` becomes eight rasterised
  variants of that map, drawn from `random.Random(--seed)`. The same `--seed` gives the same eight,
  and the run's manifest records the concrete loader names.

This is the **rasterised** obstacle story: many seeds per map, baked once at load. The grids stay
rasterised once and shared by every environment on that track, which is what makes several hundred
environments fit on one GPU, and re-baking them per reset is still not a thing that happens.

What *is* implemented, since 2026-09-13, is the other way round: `--procedural-obstacles` draws a
fresh layout for each environment at every reset as **props**, which need no grid at all. It uses
the same six patterns as `#hard:*` and the same sizes, and it is a training flag only — it places
nothing on a catalogue track and changes no loader name, so `real/bb22-1#hard:44` means exactly what
it meant. The two are complementary rather than alternatives:

| | `#hard:<seed>` / `+hard<seed>` | `--procedural-obstacles` |
| --- | --- | --- |
| what an obstacle is | cells in `occupancy` and `tall` | a convex prism (`f1sim.props`) |
| when it is drawn | once, at load | at every reset, per environment |
| how many layouts a map has | one per seed | a new one every episode |
| passability | proved by eroding the free space and checking the lane still connects | guaranteed by construction, so no proof is needed per draw |
| seen by the grid rewards (proximity, plan clearance) | yes | **no** — see [training.md](training.md#obstacle-layouts-redrawn-at-every-reset) |
| evaluation and the frozen suites | yes | never |

Use `#hard:*` when the layout should be part of the map -- a scenario to score, a picture to show,
a benchmark cell. Use `--procedural-obstacles` when the point is that the layout is *not* learnable.

## The catalogue

Every base track the registry names. The `gen/` rows are the generated seeds the splits use;
any other seed of any style is a valid id too (see below).

| id | 이름 | 옛 이름 (로더) | 출처 |
| --- | --- | --- | --- |
| `real/icra22` | ICRA 2022 | `real:icra2022` | bag |
| `real/bb21-1` | Blackbox 2021 #1 | `real:blackbox2021_1` | bag |
| `real/bb21-2` | Blackbox 2021 #2 | `real:blackbox2021_2` | bag |
| `real/bb21-3` | Blackbox 2021 #3 | `real:blackbox2021_3` | bag |
| `real/bb22-1` | Blackbox 2022 #1 | `real:blackbox2022_1` | bag |
| `real/bb22-2` | Blackbox 2022 #2 | `real:blackbox2022_2` | bag |
| `real/bb22-3` | Blackbox 2022 #3 | `real:blackbox2022_3` | bag |
| `real/korea26` | Korea 2026 | `real:korea_2026_competition` | bag |
| `real/lab16x07` | 실측 16×7 m | `real:map16x07` | bag |
| `real/lab12x16` | 실측 12×16 m | `real:map12x16` | bag |
| `real/iccas25` | ICCAS 2025 | `real:korea_2025_iccas` | bag |
| `rt/austin` | Austin | `rt:Austin` | racetracks |
| `rt/brandshatch` | BrandsHatch | `rt:BrandsHatch` | racetracks |
| `rt/budapest` | Budapest | `rt:Budapest` | racetracks |
| `rt/catalunya` | Catalunya | `rt:Catalunya` | racetracks |
| `rt/hockenheim` | Hockenheim | `rt:Hockenheim` | racetracks |
| `rt/ims` | IMS | `rt:IMS` | racetracks |
| `rt/melbourne` | Melbourne | `rt:Melbourne` | racetracks |
| `rt/mexicocity` | Mexico City | `rt:Mexico City` | racetracks |
| `rt/montreal` | Montreal | `rt:Montreal` | racetracks |
| `rt/monza` | Monza | `rt:Monza` | racetracks |
| `rt/moscow` | MoscowRaceway | `rt:MoscowRaceway` | racetracks |
| `rt/nuerburgring` | Nuerburgring | `rt:Nuerburgring` | racetracks |
| `rt/oschersleben` | Oschersleben | `rt:Oschersleben` | racetracks |
| `rt/sakhir` | Sakhir | `rt:Sakhir` | racetracks |
| `rt/saopaulo` | SaoPaulo | `rt:SaoPaulo` | racetracks |
| `rt/sepang` | Sepang | `rt:Sepang` | racetracks |
| `rt/shanghai` | Shanghai | `rt:Shanghai` | racetracks |
| `rt/silverstone` | Silverstone | `rt:Silverstone` | racetracks |
| `rt/sochi` | Sochi | `rt:Sochi` | racetracks |
| `rt/spa` | Spa | `rt:Spa` | racetracks |
| `rt/spielberg` | Spielberg | `rt:Spielberg` | racetracks |
| `rt/yasmarina` | YasMarina | `rt:YasMarina` | racetracks |
| `rt/zandvoort` | Zandvoort | `rt:Zandvoort` | racetracks |
| `gen/control-1400` | 생성 control 1400 | `gen:control:1400` | generator |
| `gen/control-1401` | 생성 control 1401 | `gen:control:1401` | generator |
| `gen/control-1402` | 생성 control 1402 | `gen:control:1402` | generator |
| `gen/control-1403` | 생성 control 1403 | `gen:control:1403` | generator |
| `gen/control-1404` | 생성 control 1404 | `gen:control:1404` | generator |
| `gen/control-1405` | 생성 control 1405 | `gen:control:1405` | generator |
| `gen/control-1406` | 생성 control 1406 | `gen:control:1406` | generator |
| `gen/control-1407` | 생성 control 1407 | `gen:control:1407` | generator |
| `gen/comp-1000` | 생성 competition 1000 | `gen:competition:1000` | generator |
| `gen/comp-1001` | 생성 competition 1001 | `gen:competition:1001` | generator |
| `gen/comp-1002` | 생성 competition 1002 | `gen:competition:1002` | generator |
| `gen/comp-1003` | 생성 competition 1003 | `gen:competition:1003` | generator |
| `gen/comp-1004` | 생성 competition 1004 | `gen:competition:1004` | generator |
| `gen/comp-1005` | 생성 competition 1005 | `gen:competition:1005` | generator |
| `gen/comp-1006` | 생성 competition 1006 | `gen:competition:1006` | generator |
| `gen/comp-1007` | 생성 competition 1007 | `gen:competition:1007` | generator |
| `gen/circuit-1200` | 생성 circuit 1200 | `gen:circuit:1200` | generator |
| `gen/circuit-1201` | 생성 circuit 1201 | `gen:circuit:1201` | generator |
| `gen/circuit-1202` | 생성 circuit 1202 | `gen:circuit:1202` | generator |
| `gen/circuit-1203` | 생성 circuit 1203 | `gen:circuit:1203` | generator |
| `gen/hall-1100` | 생성 hallway 1100 | `gen:hallway:1100` | generator |
| `gen/hall-1101` | 생성 hallway 1101 | `gen:hallway:1101` | generator |
| `gen/hall-1102` | 생성 hallway 1102 | `gen:hallway:1102` | generator |
| `gen/hall-1103` | 생성 hallway 1103 | `gen:hallway:1103` | generator |
| `gen/control-1408` | 생성 control 1408 | `gen:control:1408` | generator |
| `gen/control-1409` | 생성 control 1409 | `gen:control:1409` | generator |
| `gen/control-1410` | 생성 control 1410 | `gen:control:1410` | generator |
| `gen/control-1411` | 생성 control 1411 | `gen:control:1411` | generator |
| `gen/control-1412` | 생성 control 1412 | `gen:control:1412` | generator |
| `gen/control-1413` | 생성 control 1413 | `gen:control:1413` | generator |
| `gen/comp-1008` | 생성 competition 1008 | `gen:competition:1008` | generator |
| `gen/comp-1009` | 생성 competition 1009 | `gen:competition:1009` | generator |
| `gen/comp-1010` | 생성 competition 1010 | `gen:competition:1010` | generator |
| `gen/comp-1011` | 생성 competition 1011 | `gen:competition:1011` | generator |
| `gen/control-1414` | 생성 control 1414 | `gen:control:1414` | generator |
| `gen/control-1415` | 생성 control 1415 | `gen:control:1415` | generator |
| `gen/control-1416` | 생성 control 1416 | `gen:control:1416` | generator |
| `gen/control-1417` | 생성 control 1417 | `gen:control:1417` | generator |
| `gen/control-1418` | 생성 control 1418 | `gen:control:1418` | generator |
| `gen/control-1419` | 생성 control 1419 | `gen:control:1419` | generator |
| `gen/comp-1012` | 생성 competition 1012 | `gen:competition:1012` | generator |
| `gen/comp-1013` | 생성 competition 1013 | `gen:competition:1013` | generator |
| `gen/comp-1014` | 생성 competition 1014 | `gen:competition:1014` | generator |
| `gen/comp-1015` | 생성 competition 1015 | `gen:competition:1015` | generator |
| `gen/comp-0` | 생성 competition 0 | `gen:competition:0` | generator |
| `gen/control-9100` | 생성 control 9100 | `gen:control:9100` | generator |
| `gen/comp-9200` | 생성 competition 9200 | `gen:competition:9200` | generator |
| `gen/control-9102` | 생성 control 9102 | `gen:control:9102` | generator |
| `gym/berlin` | berlin | `gym:berlin` | gym |
| `gym/levine` | levine | `gym:levine` | gym |
| `gym/skirk` | skirk | `gym:skirk` | gym |
| `gym/stata` | stata basement | `gym:stata_basement` | gym |
| `gym/vegas` | vegas | `gym:vegas` | gym |

Generated tracks are not enumerated: `gen/<style>-<seed>` is any seed of any style, with the style
spelled `comp` (competition), `control`, `circuit`, `hall` (hallway) or `serp` (serpentine). Editor
scenes are `scene/<name>`, the name you saved it under. Slugs are lowercase and at most twelve
characters, so the id column of a narrow list never elides.

## The three splits

Groups in the UI are three, and the line between them is one question: **has the policy seen this
floor?**

| group | what it is | what a score on it means |
| --- | --- | --- |
| **학습** | the maps a policy trains on, in every direction and with every obstacle family that was trained against | how well a known circuit was learned |
| **검증** | maps that have never been used for training in any form | the only generalisation number there is |
| **내 환경** | whatever you built in the environment editor | whatever you built it to test |

The five groups the console used to show — 기본 평가셋, 장애물 (상자·궤짝·드럼), 기본 학습셋,
이전 실험 재현 (격자 장애물), 전체 카탈로그 — named *where a list came from*, not what it means, and
two of them were the same maps with different obstacles stamped in. Direction and obstacles are an
option on a map now, so they are not groups. The rest of the catalogue (the other twenty racetracks,
the gym maps, any generator seed) is reached by typing in the search box, which searches everything.

The group label is a claim about the **project's** split, not about the checkpoint you happen to have
selected: runs here were resumed with different `--tracks`, so what a checkpoint actually saw is in
its own manifest. The UI says so wherever a group label appears.

### How the lists are built

`learn/common.py` no longer holds the names. `tracks.py` holds one `SplitRule` per base track —
which directions, which obstacle family, which seeds — and `TRAIN_TRACKS`, `HELDOUT_TRACKS` and
`HELDOUT_OBSTACLE_TRACKS` are generated from them:

```python
SplitRule("real/korea26", ("", "rev", "mir"), "line", (211, 212, 213, 214))
# -> real:korea_2026_competition+rlobs211, ...+rlobs211~rev, ...+rlobs211~mir, ...+rlobs212, ...
```

`tests/test_tracks.py` holds the previous literal lists as a frozen oracle and compares them string
for string, in order. That comparison is the whole licence for the refactor: a checkpoint manifest, a
frozen benchmark suite and `heldout_leakage` all name their tracks in the loader's grammar, and a
reordered or dropped variant would quietly redefine what "trained on" and "held out" mean in all of
them. When the split is deliberately changed, the oracle changes in the same commit and the diff is
the statement of what moved.

`real/korea26` — the venue this car actually raced at — is a **training** venue and nothing else.
Twenty-five obstacle variants of that floor are trained on, so its geometry, its folds and its sight
lines are all in the weights; a clean lap on it says a known circuit was memorised. The two
pre-competition floors (`real/lab16x07`, `real/lab12x16`) are the ones the simulator had never held
at all, which is what makes a generalisation number possible. See [benchmark.md](benchmark.md).

## Using it

```bash
# the curated training split, exactly as the recipes were measured with
python3 -m f1sim.learn.ppo --tracks train ...

# five maps, both directions
python3 -m f1sim.learn.ppo --tracks 'real/icra22,real/icra22@rev,rt/spielberg,rt/spielberg@rev' ...

# one map with eight random box placements on the racing line
python3 -m f1sim.learn.ppo --tracks 'real/bb22-1#line:*' --obstacle-draws 8 --seed 701 ...

# the held-out split, for evaluation only
python3 -m f1sim.learn.evaluate --tracks heldout ...
```

```python
from f1sim import maps, tracks

maps.load("real/bb22-1@rev#line:44")            # same object as maps.load("real:blackbox2022_1+rlobs44~rev")
tracks.resolve("real/korea26@mir+rev")          # 'real:korea_2026_competition~mir~rev'
tracks.short("real:korea_2025_iccas+obs101")    # 'real/iccas25#edge:101'
tracks.split_summary("heldout")["n_tracks"]     # 8
tracks.expand("real/bb22-1#line:*", 8, random.Random(701))
```

In the console, the map card lists base tracks and the 시나리오 row underneath builds the rest:
방향, 장애물, 시드. On the training page the same three controls apply to whatever is ticked in the
picker, and 학습 셋 전체 emits `--tracks train` — the curated list itself, not a reconstruction of it.
