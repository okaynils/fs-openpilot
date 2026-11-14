# Architecture

## Package map

| module | what it does |
|---|---|
| `fsop.track` | FSG rule constants (`rules.py`, flags for unverified values), track schema, generators (curvature-domain + Fourier), fixed skidpad/acceleration layouts, validator, JSON/EUFS/FSDS IO |
| `fsop.sim` | vehicle model (kinematic → dynamic bicycle), expert (constrained racing reference + feedback control), counterfactual plan labels, noise injection, referee, domain randomization, Panda3D renderer, recorder, parallel dataset generation |
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
- **Expert labels.** The default expert optimizes a racing reference within the track corridor,
  then tracks it with pure pursuit (or Stanley feed-forward). Smooth lateral offsets minimize predicted
  travel time using a speed envelope with combined tire grip, acceleration/braking, drag and steering
  rate limits. Constraints include body clearance, nearby cones and maximum steering curvature.
  A hard 0.35 m body margin includes the cone base; the desired tracking reserve is 0.60 m.
  These are configurable through `ExpertParams.path_min_clearance` and `path_clearance`.
  The solver first seeks a path with the full desired reserve. Only when it cannot find a feasible
  candidate within its budget does it use the hard minimum margin. In that case, permitted speed
  scales with available/desired clearance, and lateral acceleration with the square of that ratio.
  Braking is propagated backwards before the narrow section. This is an explicit planning heuristic,
  not a proven tracking-error bound. The result is a local numerical solution under a simplified
  model, not a globally optimal or real-car-validated trajectory.
  Labels remain **counterfactual**: a noise-free vehicle rollout from each visited state toward this
  reference. Recovery excursions affect recording actions, not the teacher's reference. Mission
  progress stays on the original route so skidpad segments, timing lines and lap counts retain their
  meaning. Start and final stopping remain mission constraints outside the flying speed envelope.
  `ExpertParams(path_mode="centerline")` retains the legacy teacher for matched comparisons.
- **Label provenance.** New episode metadata includes `label_planner` (version, mode, solver result,
  reference time and centerline comparison with the same speed model). `reference.npz` stores route
  station, reference positions, offsets and speeds. Each retained iterate satisfies the chosen clearance
  constraints and improves a feasible baseline. `clearance_required` records which margin was used.
  An infeasible centerline is not a valid time target: a necessary feasible detour may take longer.
  Iteration-limited improvements are kept with `converged=false`, and the solver status and iteration
  count are recorded. If no improvement exists, a feasible centerline is retained; infeasible
  fallback raises an error. The optimizer uses the full privileged
  route and actual episode cones (including jitter and extra cones); camera visibility and perception
  uncertainty are not yet planner constraints.
  Counterfactual racing labels also check every integrated vehicle pose against scene cones.
  If the nominal tracked trajectory intersects a cone, version 6 compares slower speed schedules
  over the full horizon and chooses the clear candidate with greatest route progress. Clear nominal
  trajectories stay unchanged. If no candidate is clear, it delays first contact and then reduces
  contact duration; `rollout_safety` records adjusted and unresolved frames explicitly. This finite
  trajectory search accounts for tracking dynamics and recovery state but is not a global optimum
  or a continuous swept-volume guarantee. Recorded behavior still uses the configured feedback driver.
  Legacy datasets and checkpoints contain the old centerline targets. Generate a
  fresh dataset directory, rebuild caches, retrain and benchmark before claiming learned improvement.
  To reuse existing observations, run `fsop data relabel-racing --source data/sim/v1 --out data/sim/racing_v6`.
  This copies states and geometry, regenerates counterfactual plans, and references source images
  **read-only** through `frames_source`. The source dataset must remain available at that path.
  Destination feature caches are independent; rebuild history from the initialization checkpoint.
  Relabeling commits episodes atomically, resumes only matching source/code fingerprints, and publishes
  the dataset index only when all episodes are complete. Recorded behavior and label expert parameters
  are kept separately. `configs/train/cuda_racing_v6.yaml` requires racing labels and refuses legacy
  targets before loading the model. Its best checkpoint minimizes validation ADE over the first
  four seconds; the initialization checkpoint is eligible, so improved cone losses cannot hide a
  regression in trajectory accuracy.
  Run `modal run --detach scripts/modal_app.py::racing` to relabel the cloud recovery dataset,
  enforce the sampled-clearance gate, build history caches on L4/A10, and train on A100.
  Training stages indexed episodes on local cloud SSD and materializes read-only source frames there.
  Use `modal app logs` to monitor the detached job; training remains on Modal.
  `uv run python scripts/benchmark_expert.py` compares teachers without rendering or model weights.
  The matched suite is recorded in [racing_v5.json](../benchmarks/expert/racing_v5.json):
  all 20 runs passed with no cone hits, with six faster racing runs, two slightly slower runs (at most 0.9%) and two
  unchanged acceleration runs. This is teacher evidence, not learned-model improvement.
  The version-5 sampled label audit on all 204 original episodes (39,354 frames) found two
  frames with predicted body/cone contact, versus seven under the original labels. The cloud
  held-out subset had zero versus six across 17 episodes. These checks sample plan poses over
  0.5–4 seconds; they do not prove continuous swept-volume clearance or learned driving quality.
  The broader version-5 recovery dataset failed its gate (416 contact frames versus 251 across
  284,172 frames), so no model was trained on it. Version 6 adds the trajectory correction above;
  its full audit found 187 versus 251 contact frames, a 25.5% reduction. Its held-out
  subset had zero versus six contact frames across 17 episodes.
  The [version-6 learned-model report](../benchmarks/racing/v6.json) records A100 training
  for 4,000 steps, selecting step 3,500 by validation ADE (1.346 → 1.270 m on the fixed subset).
  Full validation geometry error at 10 m fell from 0.349 to 0.248 m; on substantially changed
  paths it fell from 0.470 to 0.275 m. The final 17-episode test comparison confirmed
  0.306 → 0.229 m overall and 0.402 → 0.235 m on changed paths, with full 10 m coverage.
  Test ADE over four seconds fell from 1.574 to 1.455 m.
  Both checkpoints then drove the quick suite in the same A10 container. Both passed all
  three trackdrive scenarios; cone hits fell from four to two, with no off-course events.
  Mean penalty-adjusted time was essentially unchanged (48.52 → 48.65 s), while raw time
  increased (45.85 → 47.32 s). This demonstrates learned geometry and fewer contacts,
  **not a learned lap-time improvement**. Acceleration passed for both (8.75 → 9.05 s).
  Skidpad improved to route completion but still failed with 19 cone hits and one off-course
  event. This small simulation suite does not establish general driving or real-car readiness.
  The selected checkpoint is `runs/cuda_racing_v6/ckpt/best.pt` locally and on the Modal volume;
  the report records its SHA256, source hashes, hardware, and complete metrics.
  `scripts/evaluate_racing.py predict` caches held-out predictions for a checkpoint; `compare`
  checks that both models saw identical recorded states and image locations, then reports racing-label
  errors and lateral error on targets that moved by more than 0.20 m. A separate geometric metric
  compares positions 10 metres along each path, including the subset whose geometry changed by
  more than 0.20 m. Short predictions clamp to their endpoint and report missing coverage, so stopping
  cannot discard difficult frames. This distinguishes path geometry from timing. Use validation tracks for model
  selection and reserve the test split for the final comparison.
  For background on the distinction between minimum curvature and minimum time, see
  [TUM's racing trajectory optimization](https://github.com/TUMFTM/global_racetrajectory_optimization).
- **Recording.** Data is stored on the 5 Hz grid, matching the model’s 0.2 s frame-pair/history
  spacing and saving 4× stored frames. This does not replace evaluation on actual
  20 Hz sequences or capture every faster sensor/mounting disturbance.
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


### Racing-label clearance audit

`uv run python scripts/audit_racing_labels.py --dataset data/sim/racing_v6 --out runs/racing_eval/clearance.json`
checks vehicle footprints against the actual recorded cones at sampled plan poses between 0.5 and 4 s.
It also checks the original labels on the same observations. This catches mismatches between nominal
map geometry and scene randomization; it is not a continuous swept-volume guarantee. Planner version 5
prioritizes the desired reserve, uses clearance-dependent speed limits when it must fall back to the
hard margin, and distinguishes an infeasible centerline from a valid time baseline. Safety and
closed-loop completion must be assessed alongside travel time and imitation error.


## Experimental transfer contract

[Camera/vehicle transfer](transfer.md) adds optional versioned geometry inputs to
`SupercomboFS.plan`. A zero-initialized adapter combines calibrated geometry with the
current visual feature before the existing temporal policy. Vision history remains
unconditioned, so cache identity is tied to the vision checkpoint, episode metadata and
inference precision; geometry is loaded from episode metadata for every sample.

The physical-sensor path renders rectilinear source pixels and calls the deployment
warper before YUV packing. Per-frame `camera_poses.npy` drives moving-camera targets;
`calibrated_mount` is the separate estimate available to the policy. Labels stay in the
ground-projected CoG frame. Body visibility uses the same ray/box geometry in images
and targets. Distribution manifests, configuration-level split assignments and source
snapshots accompany generated datasets.

Training checkpoints now include optimizer/RNG state and an input contract. Explicit
resume continues the optimizer and random state, but restarts the shuffled data loader;
it is not a promise of bitwise identical sample order after interruption. Existing
checkpoint loading remains strict and infers legacy head presence from the state dict.

Inference and newly cached features disable TF32 to agree with the FP32 export reference.
BF16 training autocast is still intentional. ONNX CUDA inference sets `use_tf32=0`, as
documented by the [ONNX Runtime CUDA provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32).
TensorRT/CoreML reduced-precision execution remains opt-in and needs its own parity checks.

Context v1 retains the experimental 16 fields. Context v2 removes camera-angle
channels after validation intervention, retaining 13 calibrated translation/vehicle
fields. The anchored residual subtracts its response at a checkpoint-stored reference
configuration while the original policy remains frozen. Specialization removes the
three first-layer columns at zero angle, with recorded-sequence equivalence tests.
Profiles retain full angles for rendering, warping and cone projection. `FSDriver`
reads the input version from the artifact; training selects it with `context_version`.
