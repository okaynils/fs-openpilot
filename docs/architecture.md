# Architecture

## Package map

| module | what it does |
|---|---|
| `fsop.track` | FSG rule constants (`rules.py`, flags for unverified values), track schema, generators (curvature-domain + Fourier), fixed skidpad/acceleration layouts, validator, JSON/EUFS/FSDS IO |
| `fsop.sim` | vehicle model (kinematic → dynamic bicycle), expert (pure pursuit + speed profile), counterfactual plan labels, noise injection, referee, domain randomization, Panda3D renderer, recorder, parallel dataset generation |
| `fsop.geometry` | frames and camera models (openpilot model-frame intrinsics), YUV packing (bit-exact port), plan frame conversion, real-camera → model-frame warp |
| `fsop.model` | ONNX prep (FP32, batchable, taps, vision/policy split), onnx2torch converters, parity check, `SupercomboFS` (+ cone heads), ONNX export |
| `fsop.data` | sim dataset (cached-feature and image modes), on-the-fly targets, feature cache, FSOCO import/dataset |
| `fsop.train` | trainer (stages 1/2, L2-SP, FSOCO mixing), losses |
| `fsop.eval` | open loop, closed-loop suite + mission layer, controllers, geometric baseline, BEV/COCO metrics, DAgger, image → ground projection |
| `fsop.viz` | Rerun schema/blueprints, dataset inspector, MP4/GIF rendering, track previews |
| `fsop.profile`, `fsop.runtime`, `fsop.bench` | team profiles, deployment runtime, benchmark records |

## The openpilot model (commit `b361e952`)

- **Inputs:**
  - `img` and `big_img` (uint8 [12,128,256]): road (medmodel, fl 910) and wide (sbigmodel, fl 455)
    cameras. Each holds two frames 0.2 s apart in 6-channel YUV420 packing.
  - `features_buffer` [24,512]: past hidden states at 5 Hz.
  - `desire_pulse` [25,8].
  - `traffic_convention`.
  - `action_t` (unused by the graph).
- **Trunk:** FastViT/MobileOne-style and reparameterized. Both cameras are channel-fused into one trunk.
- **Policy:** a one-layer causal transformer over the **last 9** history slots (current + 8 past = 1.6 s),
  then a ResMLP plan head: 33 time steps × 15 values (position, velocity, acceleration, euler, rates),
  as mean + log-std of a single Laplace.
- **Plan frame:** x forward, **y right, z down**. fsop uses x forward / y left / z up internally
  (`fsop.geometry.op_frames`). This was verified empirically, with a correlation of −0.89 between the
  untouched model's lateral plan and ground truth.
- **Graph surgery for training** (`fsop.model.onnx_prep`):
  - FP16 → FP32;
  - batch-1 Reshape constants → batchable;
  - `allowzero` dropped;
  - the constant pad is dropped from the policy output;
  - converters added for Gelu-20, Reduce*-18 and Cast/Reshape/Identity-19.
- **Parity:** torch matches the FP32 ONNX graph to about 5e-7 relative (`fsop model verify`).

## Heads

| head | input | output |
|---|---|---|
| plan (kept, fine-tuned) | policy | 33×15 trajectory, mean + log-std |
| `cone_img` (new) | trunk taps s4/s8/s16, FPN-lite | per camera slot: CenterNet heatmap (bottom-centre key point), w/h, offsets |
| `cone_bev` (new) | s16 + final features, BEV-query cross-attention | 4-class 64×64 grid (0–48 m × ±24 m) |

**Cones used downstream** come from the image head plus ground projection with the known camera mount
(`fsop.eval.img_to_bev`). This is about 3× better F1 than the learned BEV lifting, which suffers because
both cameras are fused in the trunk.

## Simulator

- **Rendering.** Offscreen Panda3D renders directly with the model-frame intrinsics, so no warp is
  needed in simulation and pixel geometry is exact (`tests/test_reprojection.py`).
  - The camera sits at the profile's mount, randomized per episode.
  - The real car uses the same virtual camera through `fsop.geometry.warp`.
- **Expert labels.** The expert is pure pursuit with a curvature-limited speed profile and randomized
  aggressiveness. Labels are **counterfactual**: a noise-free expert rollout from every visited state.
  Combined with OU steering-noise injection this yields recovery labels, in the spirit of DAgger.
- **Recording.** Data is stored on the 5 Hz grid. The model only needs frames t and t−4 at 20 Hz plus
  5 Hz history, so this loses nothing and saves 4× space.
- **Referee.** Implements D 2.6, D 4–8 and D 10, with penalties per D 10.1.7.
- **Mission layer** (`fsop.eval.closed_loop.mission_brake`): launch on "go" and braking into the stop
  zone. It currently uses the simulator's track progress, standing in for odometry and lap counting
  on the car.

## Design decisions and findings

- **Renderer.** MetaDrive (commaai `minimal` branch) was evaluated first, as planned. On macOS it reset
  in about 8 s, rendered 2 cameras at about 20 FPS, and its road maps don't fit cone tracks. Pure
  Panda3D renders at about 500 FPS (1 camera) and still reuses MetaDrive's textures when installed
  (`scripts/spikes/`).
- **Frozen trunk is not enough.** Stage 1 (frozen road features) plateaus well above a constant-yaw-rate
  baseline. Fine-tuning the last trunk stage (stage 2) is what makes the plan work.
- **Launch.** A still frame pair carries no information about whether to start, because staging and
  finishing look identical. Go/stop belongs to the AS state machine, not the model.
- **Inference precision.** ONNX Runtime with the CoreML provider runs internally in fp16 and drifts by
  up to 3 m at the 10 s plan horizon. The runtime therefore defaults to the CPU or CUDA providers.
