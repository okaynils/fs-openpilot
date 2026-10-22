# Team guide: using fs-openpilot with your car

This guide walks through adapting the pipeline to your vehicle and cameras, from profile to an ONNX
model running in your stack.

> Safety: fs-openpilot is research software. It does not implement the AS state machine, EBS, RES
> handling or any safety monitoring, and it has not been validated on a real vehicle. Treat its
> outputs like any other untested component in your software stack.

## 1. Team profile

A profile holds everything car-specific. Copy `configs/profiles/example_team.yaml`; unset keys keep the
defaults from `configs/profiles/default.yaml`.

| section | keys | used by |
|---|---|---|
| `vehicle` | `mass`, `lf`/`lr` (CoG → axles), `track`, `width`, overhangs, tyre stiffness `cf`/`cr`, `mu`, `max_steer_deg`, accel/brake limits, actuator lags | simulator dynamics, referee footprint, controllers |
| `mount` | camera position `x, y, z` [m] relative to the CoG on the ground (x fwd, y left, z up), `roll/pitch/yaw_deg` | simulator camera (± DR jitter), ground projection of cones |
| `cameras` | per physical camera: resolution, `fx, fy, cx, cy`, `dist` (OpenCV radtan, or fisheye with `fisheye: true`), `slots` it feeds, small `roll/pitch/yaw_deg` offsets from `mount` | `fsop.geometry.warp`, runtime |

`fsop camera show configs/profiles/<team>.yaml` parses the profile and rejects unknown keys.

### Cameras: what the model expects

The model has two input slots:
- **road:** the medmodel, about 31° horizontal FOV, sharp at range;
- **wide:** the sbigmodel, about 59° horizontal FOV, for the near field and the sides.

Both are virtual pinhole cameras at your `mount`.

- **One camera, both slots (most FS cars).** Set `slots: [road, wide]`. The camera needs at least
  about 60° horizontal FOV to fill the wide slot; the road slot is a 2× zoom of the centre.
- **Two cameras** (narrow and wide). Give each camera one slot.
- **Mount height.** Train with your real height. The simulator randomizes ±0.1 m (light DR) to
  ±0.25 m (heavy DR) around it. Higher mounts see the ground closer to the car: at 1.1 m the first
  visible ground is about 5 m ahead in the road slot.
- **Calibration.** Use standard OpenCV intrinsic calibration (checkerboard). For orientation, put the
  car on flat ground and adjust `pitch_deg` and `roll_deg` until
  `fsop camera warp --profile <p> --image front=frame.jpg` shows the magenta horizon line on the real
  horizon and the 10/20 m lines on cones you placed at those distances.

## 2. Data

- **Simulation.**
  - Run `fsop track gen` (presets in `configs/track/`: `trackdrive`, `autocross`, `tight`).
  - Then run `fsop sim record --profile <p>`. Every episode uses your dynamics and camera mount.
  - Plan labels are counterfactual expert rollouts, so they encode how to recover from the
    noise-injected states.
- **FSOCO** (real images, for the cone head). Use `fsop data fsoco-import --src <supervisely or yolo export>`.
  - Images are mapped into both slots with a randomized assumed FOV, and the split is by team.
  - Verify the watermark border (`border`, default 140 px) against your export.
- **Your own logs (not yet supported end to end).**
  - The intended path: record synchronized camera, wheel-speed and IMU data, warp the images with
    `fsop.geometry.warp`, and derive future-trajectory labels from odometry.
  - A log importer is on the [roadmap](ROADMAP.md); contributions are very welcome.

## 3. Training

| config | what | M1 time |
|---|---|---|
| `configs/train/mac_stage1.yaml` | trunk frozen, cached features → plan + BEV heads | ~20 min (3k steps) |
| `configs/train/mac_stage2.yaml` | last trunk stage unfrozen (L2-SP), image cone head, optional FSOCO | ~25 min (3k steps) |
| `configs/train/cuda_stage{1,2}.yaml` | same recipe, bigger batches, full trunk | GPU |

Override any key with `-o key=value`. Each run writes TensorBoard logs to `runs/<name>/tb` and Rerun
validation snapshots to `runs/<name>/viz/*.rrd`.

## 4. Evaluation

- `fsop eval open-loop --dataset … --ckpt …` gives trajectory errors and cone P/R/F1 on held-out tracks.
- `fsop eval closed-loop --ckpt … --profile <p>` runs the suite with the FSG referee. The suites are
  `configs/eval/closed_loop_quick.yaml` and `closed_loop_suite.yaml`.
- `fsop bench run --name <id> --ckpt …` does both and adds a record to `benchmarks/results/`.
- `fsop viz video` renders an MP4 like the one in the README.

## 5. Deployment

```python
from fsop.runtime import FSDriver

drv = FSDriver("artifacts/export/fs_supercombo.onnx", profile="configs/profiles/my_team.yaml")
# 20 Hz loop (the model's rate; the temporal queues assume it)
out = drv.step({"front": rgb_uint8}, v_ego=wheel_speed_mps)
out.plan        # [33, 15] trajectory in the ego frame (x fwd, y left), times 10*(i/32)^2 s
out.cones       # [N, 4] x, y [m], class (0 blue, 1 yellow, 2 orange, 3 big orange), score
out.steer, out.accel   # from a pure-pursuit plan follower (fsop.eval.controller); or use the plan directly
```

- **Car requirements:** `numpy`, `opencv`, `onnx`, `onnxruntime`.
  - The default providers are CPU or CUDA. CoreML/fp16 drift on the long horizon, so treat them as
    opt-in.
  - The M1 CPU takes about 45 ms/step. Jetson-class GPUs should be well below that, but this hasn't
    been measured yet.
- **What your stack must provide:**
  - the AS state machine: go/stop, lap counting (D 8.3.4), stop-zone braking, EBS, RES;
  - for skidpad, route commands via `desire`: 2 = right circle, 1 = left circle, 7 = exit, each a
    pulse about 2 s before the segment.
- **ROS 2:** wrap `FSDriver.step` in a node subscribing to your image topic(s) and wheel speed. A
  reference node is on the roadmap.

## Known gaps

- **Skidpad** is not solved yet. More route-conditioned data is needed.
- **No real-car validation.** The sim-to-real gap is mitigated by domain randomization and FSOCO,
  but not measured.
- **Rule values to verify** (`fsop.track.rules`, marked `verified=False`): cone dimensions, maximum cone
  spacing, and the generator's section clearance. They need checking against the FSG Event Handbook
  and the FS Driverless specification. FSAE/FS-AI variants are not modelled separately.
