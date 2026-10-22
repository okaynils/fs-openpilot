# fs-openpilot

Fine-tune [openpilot](https://github.com/commaai/openpilot)'s open driving model for **Formula Student
Driverless**. It covers procedurally generated FSG-rule tracks, a cone-track simulator, sim + FSOCO training, open-
and closed-loop evaluation with an FSG referee, ONNX export, and Rerun visualization throughout.

```
 tracks (FSG 2027 rules) ─▶ simulator (Panda3D, model-frame cameras, expert + noise) ─▶ dataset (5 Hz, YUV)
        │                                                                                    │
        ▼                                                                     feature cache (frozen trunk)
 validator / previews                                                                        │
                                       openpilot supercombo (ONNX ─▶ torch) + cone heads ◀───┘ + FSOCO
                                                         │
                               open-loop metrics · closed-loop suite (referee) · ONNX export · Rerun
```

## What the model is

openpilot `driving_supercombo.onnx` @ `b361e952` (MIT, sha256 `659727c4…`), converted to PyTorch with
parity to the FP32 graph of 5e-7 relative (`fsop model verify`).

- **Trunk.** A FastViT/MobileOne-style reparameterized trunk sees both cameras channel-fused: road
  (medmodel, fl 910) and wide (sbigmodel, fl 455). Each camera input is 2 frames 0.2 s apart in openpilot's
  6-channel YUV packing.
- **Policy.** A one-layer temporal transformer over the current plus 8 past 512-d hidden states (5 Hz).
  The plan is a 33×15 Laplace head (position, velocity, acceleration, euler, rates over 0–10 s).
- **Frame convention.** openpilot plans are in x fwd / **y right / z down**. fsop works in x fwd / y left /
  z up (`fsop.geometry.op_frames`). This was verified empirically: the untouched model's lateral plan
  correlates at −0.89 with ego-frame ground truth in the sim.

Formula Student additions (`fsop.model`):

| head | input | output |
|---|---|---|
| plan (kept, fine-tuned) | policy | track-centre trajectory (33×15, mean + log-std) |
| `cone_bev` (new) | s16 + final trunk features, BEV-query cross-attention | 4-class cone heatmap + offsets, 64×64 grid, 0–48 m × ±24 m |
| `cone_img` (new) | s4/s8/s16 taps, FPN-lite | per-camera CenterNet (bottom-centre key point, w/h); trained on sim **and FSOCO** |

## Quickstart (Apple Silicon or Linux; Python 3.12, uv)

```bash
uv sync --extra sim                      # torch, onnx2torch, rerun, panda3d (+ MetaDrive assets)
uv run python -m metadrive.pull_asset    # optional: asphalt/grass textures reused by the renderer
uv run fsop model fetch && uv run fsop model prepare && uv run fsop model verify

# tracks + simulator
uv run fsop track gen --out data/tracks/td_v1 --n 160 --seed 1000
uv run fsop track validate data/tracks/td_v1
uv run fsop track preview data/tracks/td_v1 --png artifacts/previews/td_v1.png   # or --rerun
uv run fsop sim drive --track skidpad              # expert lap in the Rerun viewer
uv run fsop sim drive --track gen:7 --noise --save runs/demo.rrd

# dataset + training
uv run fsop sim record --tracks data/tracks/td_v1 --out data/sim/v1 --workers 6
uv run fsop data cache-features --dataset data/sim/v1
uv run fsop train --cfg configs/train/mac_stage1.yaml       # frozen trunk: plan + BEV heads (~40 min on M1)
uv run fsop data fsoco-import --src /path/to/fsoco          # Supervisely or YOLO export
uv run fsop train --cfg configs/train/mac_stage2.yaml -o fsoco_root=data/fsoco

# evaluation, visualization, export
uv run fsop eval open-loop  --dataset data/sim/v1 --ckpt runs/mac_stage1/ckpt/best.pt --out runs/ol.json
uv run fsop eval open-loop  --dataset data/sim/v1                          # untouched openpilot baseline
uv run fsop eval closed-loop --ckpt runs/mac_stage1/ckpt/best.pt --out runs/closed_loop/stage1
uv run fsop eval closed-loop --baseline geometric_gt --out runs/closed_loop/geometric_gt
uv run fsop viz dataset data/sim/v1/<episode> --ckpt runs/mac_stage1/ckpt/best.pt
uv run fsop export --ckpt runs/mac_stage1/ckpt/best.pt
uv run pytest
```

Every command takes `--help`. Training configs are OmegaConf dataclasses. Override any field with
`-o key=value`, and use `defaults: [base.yaml]` in a config for simple inheritance (see `configs/train/cuda_*.yaml`).

## Pipeline pieces

**Tracks** (`fsop.track`)
- Trackdrive and autocross loops are generated in the curvature domain: random corner/straight
  sequences, closed by least squares, with optional slaloms. A Fourier-loop generator adds variety.
- Cones are placed with curvature-adaptive spacing. Each track gets big orange start/finish gates,
  a staging pose and a stop zone.
- Skidpad and acceleration use fixed layouts per D 4/D 5.
- `validate()` checks:
  - lap length 200–500 m;
  - straights ≤ 80 m;
  - turning diameter ≥ 9 m;
  - width ≥ 3 m;
  - no self-intersection and clearance between sections;
  - cone spacing and sides (blue left, yellow right);
  - gates.
- IO: JSON, plus EUFS/FSDS cone CSV import and export.

**Simulator** (`fsop.sim`)
- Rendering is offscreen Panda3D, directly in openpilot's model-frame intrinsics: about 215 FPS for both
  cameras on an M1. Pixel geometry is exact (`tests/test_reprojection.py`).
- The car is a kinematic → dynamic bicycle model with saturating tires and actuator lag, at 100 Hz.
- The expert is pure pursuit on the centerline with a curvature speed profile and randomized
  aggressiveness. OU steering-noise injection produces recovery data.
- Plan labels are **counterfactual**: a noise-free expert rollout from each state.
- Domain randomization covers:
  - environment: lighting, fog, asphalt textures, distractors;
  - cones: jitter, missing or extra cones;
  - camera: mount height and angles;
  - image effects: exposure, noise, blur, JPEG;
  - dynamics: tire μ, mass, actuator lag.
- The referee implements laps, DOO, OC, stop zone, unsafe stop and the DQ speed rules, with penalties
  per D 10.1.7.
- Skidpad route choice (right×2, left×2, exit) is given to the model through openpilot's desire input.

**Data** (`fsop.data`)
- Episodes are stored on the 5 Hz grid. The model consumes frames t−4/t at 20 Hz plus 5 Hz history,
  so this loses nothing and saves 4× space (about 80 KB per frame for both cameras).
- Targets are derived on the fly:
  - BEV heatmaps masked to the camera frustum;
  - per-camera image boxes.
- FSOCO images are mapped into both model-frame slots, with a randomized assumed FOV and horizon, a
  blue↔yellow-swapping mirror, and a team-level split.

**Training** (`fsop.train`)
- Stage 1 freezes the trunk and trains the policy and BEV head on cached features. About 170
  samples/s on M1, IO bound.
- Stage 2 fine-tunes from images:
  - the last trunk stage and summarizer are unfrozen, regularized by L2-SP towards the pretrained weights;
  - the image cone head is added;
  - FSOCO batches are mixed in.
- MPS runs in FP32; CUDA uses bf16 autocast.
- TensorBoard logs go to `runs/<name>/tb`. Each validation also writes a Rerun snapshot to
  `runs/<name>/viz/*.rrd`.

**Evaluation** (`fsop.eval`)
- Open-loop metrics:
  - plan: lateral and longitudinal error at 0.5/1/2/4 s, ADE, heading, speed, Laplace calibration;
  - BEV cones: P/R/F1 per class and range, blue↔yellow swap rate;
  - image head: COCO mAP.
- Closed-loop suite on held-out tracks × DR levels, with this report from the referee:
  - completion, pass, DOO, OC, lap time;
  - lateral RMS.
- Closed-loop policies:
  - `model`;
  - `openpilot` (untouched);
  - `geometric_gt` (cone-midline planner on ground-truth cones);
  - `geometric_model` (midline planner on the model's own cone detections);
  - `expert`.

**Visualization**: [Rerun](https://rerun.io) is the primary tool (`fsop.viz.rr_log`).
- It provides a world 3D view (track, cones, car, GT/predicted plans with ±σ band, detected cones) and
  both camera views. The cameras are logged as Pinholes with the model intrinsics, so 3D plans and cones
  are **projected into the images automatically**.
- Other panels: BEV heatmaps, telemetry, the race event log and a referee summary.
- `.rrd` files are written for every closed-loop run and every validation. Open them with `rerun file.rrd`.

## First results (M1 MacBook, dataset v1)

**Data and training.**
- Dataset v1: 160 tracks, 204 episodes, 39k frames on the 5 Hz grid.
- Stage 1 was stopped at 3k of 6k steps once it plateaued; the frozen trunk was the bottleneck.
- Stage 2 trained for 3k steps at batch 12.
- There are no FSOCO images yet, so the image head is sim-only so far.

**Open loop** on 17 held-out test tracks (3,315 frames, `runs/eval/ol_*.json`):

| metric (ego frame) | openpilot (untouched) | stage 1 (frozen trunk) | stage 2 (fine-tuned) |
|---|---|---|---|
| lateral error @0.5 s / 1 s / 2 s [m] | 0.32 / 1.28 / 4.64 | 0.19 / 0.75 / 2.69 | **0.12 / 0.43 / 1.57** |
| longitudinal error @1 s / 2 s [m] | 2.79 / 5.81 | 1.52 / 3.27 | **1.19 / 2.55** |
| ADE 0–4 s [m] | 6.28 | 3.74 | **2.60** |
| heading error @1 s [deg] | 13.0 | 7.9 | **4.1** |
| BEV cones F1, learned BEV head | – | 0.12 | 0.13 |
| BEV cones F1, image head + ground projection | – | – | **0.40** (pos err 0.17 m, blue/yellow swaps 2.7 %) |

For reference, constant-yaw-rate extrapolation with the *true* yaw rate gets a 0.82 m lateral error at 1 s.

**Closed loop**, quick suite with light DR and a standing start (`configs/eval/closed_loop_quick.yaml`,
reports in `runs/closed_loop/quick_*/report.md`). Each trackdrive run is 1 lap on each of 3 held-out tracks:

| policy | trackdrive pass | DOO / run | acceleration | skidpad |
|---|---|---|---|---|
| openpilot (untouched) | 0/3 (off course in the first 10–25 % of the lap) | – | pass (8.6 s) | fail |
| **stage 2 (end-to-end plan)** | **3/3** | 7.7 (0 on one track, 14 on the worst) | **pass (6.2 s)** | fail |
| geometric midline, GT cones (perfect perception) | 3/3 | 0.7 | pass (7.0 s) | fail |
| geometric midline, model's cones | 0/3 (66 % completion) | 0.3 | pass (12.0 s) | fail |

**Findings and next steps**
- **Train the trunk.** Fine-tuning it (stage 2) is what makes the plan work; frozen road features
  carry little cone geometry.
- **Use the image head for cones.** The learned BEV lifting head is weak. Image head + ground projection
  is three times better and is the default cone source for closed loop (`ModelPolicy(cones="img")`).
- **Skidpad fails for every policy.** There are only about 10 skidpad episodes so far. Record more
  route-conditioned skidpad data: `fsop sim record --skidpad 60`.
- **Reduce cone hits with more data:**
  - run DAgger rounds (`fsop eval dagger --ckpt …`, then re-cache and train);
  - add more tracks and episodes on a CUDA box (`configs/train/cuda_*.yaml`);
  - mix in FSOCO for the image head.
- **Launch is handled by the mission layer.** A still frame pair doesn't say whether to start (staging
  and finished look the same), so the mission layer gives a launch assist on "go". It also handles the
  final braking into the stop zone (`fsop.eval.closed_loop.mission_brake`).

## Design decisions

- **Renderer: pure Panda3D instead of MetaDrive.** The M0 spike ran MetaDrive (commaai `minimal`
  branch, installed in the `sim` extra) on macOS. It reset in 8 s and rendered 2 cameras at about 20 FPS,
  and its road-map coupling did not fit cone tracks. The Panda3D backend renders at 500 FPS (1 camera)
  and reuses MetaDrive's textures when installed. `scripts/spikes/spike_metadrive.py` keeps the
  comparison reproducible.
- **ONNX graph surgery** (`fsop.model.onnx_prep`):
  - FP16 → FP32;
  - batch-1 Reshape constants made batchable;
  - `allowzero` dropped;
  - constant pad removed from the policy output;
  - vision/policy split;
  - converters for Gelu-20, Reduce*-18, Cast/Reshape/Identity-19.
- **Learned BEV lifting.** The trunk fuses both cameras channel-wise, so geometric per-camera lifting
  isn't available.

## Rules to verify (`fsop.track.rules`, `verified=False`)

These values come from the FSG 2027 rules PDF section D, except for these, which need the Event
Handbook / DV spec:
- cone dimensions (small 228×228×325 mm, large 285×285×505 mm);
- maximum cone spacing on straights (5 m);
- section clearance used by the generator.

The EUFS/FSDS CSV layouts and the FSOCO watermark border (140 px) are also to be checked against the
upstream tools.

## Licenses

- openpilot code and model: MIT (commaai). Ported functions are marked in their docstrings.
- MetaDrive: Apache-2.0.
- Ultralytics (optional `yolo` extra): AGPL-3.0.
- FSOCO data is not redistributed; follow the FSOCO terms of use.
