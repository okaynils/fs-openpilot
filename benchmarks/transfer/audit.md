# Transfer audit and experiment chronology

See the [final experiment report](report.md) for the recommendation and comparisons.

Starting revision: `bd408fa54b1961719688a8f06471ad02c52311bb`.
Selected baseline: `runs/cuda_racing_v6/ckpt/best.pt`, SHA256
`fdf1b3d56ebd110658dace49a8c9ab25fa50c42ce076b09c0b0b5d418746f507`.
Existing working tree was clean; changes are on `camera-vehicle-transfer-v1`.

## Before changes

- [Environment](../../src/fsop/sim/env.py): height/pitch/roll/yaw perturbations sampled once at construction; x/y only configurable in a profile. Mount drives render transforms. No mounting motion or calibration error distinction. Reset retains installation. Labels remain at ground-projected centre of gravity.
- [Randomization](../../src/fsop/sim/randomization.py): scene lighting, cones, exposure, blur, noise, JPEG and optional venue degradation. No sampled vehicle families or physical intrinsics.
- [Renderer](../../src/fsop/sim/render/panda3d_backend.py): ideal fixed road/wide model slots, no physical sensor preprocessing or ego bodywork. All cameras share one position. Visibility targets check frusta, not occlusion.
- [Profiles/warping](../../src/fsop/profile.py), [warp](../../src/fsop/geometry/warp.py): physical resolution/intrinsics/distortion and orientation offsets supported at deployment, bypassed in training simulation. Rotation remapping retains camera translation; it does not produce a common viewpoint.
- [Vehicle](../../src/fsop/sim/vehicle.py): profile-specific bicycle dynamics, footprint, limits and first-order response. Defaults otherwise fixed; no vehicle speed cap. [Expert](../../src/fsop/sim/expert.py) draws car-scale speed/acceleration preferences independently of profile. Racing planner receives vehicle and recovery rollouts integrate its dynamics.
- [Model](../../src/fsop/model/build.py): images/history/desire/traffic only; wheel speed is available to controller but not trajectory policy. Calibration and vehicle geometry are absent. Monocular scale and still-image start/stop ambiguity cannot in general be resolved by randomization.
- [Dataset](../../src/fsop/data/sim_dataset.py): track hash splits, not complete configuration holdouts; static camera transform in labels. [Feature cache](../../src/fsop/data/feature_cache.py) resumes based on file presence, without content identity.
- [Trainer](../../src/fsop/train/trainer.py): validation-based selection includes initialized baseline, but checkpoint has no optimizer/RNG resume state. Existing run directories can be reused.
- [Evaluation](../../src/fsop/eval/closed_loop.py): shared seeds/tracks available, but direct policy graph call bypasses future conditioning. Privileged simulator speed, route desire, mission progress and launch/finish braking are used. Ground projection uses actual simulated mount. No deployed ONNX temporal comparison.

## Experiment gate

Preserve baseline and all prior runs. Inspect rendered extremes, verify coordinate/label contracts and bounded pilot behavior before larger training. Compare unchanged baseline, image-only diverse-data adaptation and calibrated conditioning on the same examples. Reserve complete final test configurations and seeds until checkpoint selection. Simulation results alone cannot establish real-vehicle readiness.

## Initial implementation decisions

- Calibration context v1 uses camera xyz/roll/pitch/yaw and vehicle wheelbase, width, front/rear extents, steering/speed/acceleration/braking limits and response time constants. A zero-initialized residual adapter on current vision features leaves unconditioned history caches and legacy checkpoints usable. These are candidate inputs, pending ablation evidence; no transfer claim is made yet.
- The simulator supports physical rectilinear camera rendering followed by the same `ModelFrameWarper` used at deployment. Distorted/fisheye simulation is rejected explicitly rather than silently rendered incorrectly. Distortion remains supported at deployment.
- Chassis occlusion uses an opaque axis-aligned body box in vehicle coordinates. Ray/box intersection drives both image masking and target visibility. This is intentionally a coarse approximation, without wheels, suspension, aerodynamic wings, reflections, or shadows from the vehicle.
- Mount flex is deterministic analytic motion in simulation time, independent of frame capture cadence. Calibration metadata estimates the fixed installation and does not reveal instantaneous flex to the policy.
- Initial pilot dataset uses straight acceleration courses only to check mechanics and learnability; it cannot establish cornering transfer or justify replacing the default.

## Reproduced baseline

Modal app `ap-OOSS58RG3IJwuEvKadP2ee`, function call `fc-01M7AGZZY52A5XDS951K9KSZ0C`.
[Open-loop results](default_baseline_open_loop.json): 17 legacy test episodes, 3,315 scored frames,
ADE through 4 s = 1.390793 m. [Closed-loop results](default_baseline_closed_loop.json):
trackdrive 3/3 passes, two total cone contacts; acceleration passes without contacts;
skidpad completes but fails with 19 contacts and one off-course event. Launch/finish assistance
and simulator wheel speed/progress remain enabled exactly as disclosed above.

## Predeclared decision criteria

Before looking at held-out final configurations: require at least 15% lower mean
configuration-balanced 0–4 s trajectory ADE than the selected baseline on interpolation
configurations, no reduction from 3/3 default trackdrive passes, default ADE within 10%
of 1.390793 m, and no more than one additional default trackdrive cone contact.
Also inspect transfer contacts, recovery and steering stability; these can veto replacement
regardless of ADE. Extrapolation is reported separately and cannot establish support outside
the documented distribution. This small simulator suite does not establish statistical or
real-world safety guarantees.

Pilot data generation: 12 episodes, 611 frames, 1129.786 s in an 8-physical-core/16-GiB Modal
container (`ap-O6e4JZ9DDsxlOP2Uelgh91`). Manifest SHA256
`f8445f95c621e63adce32edeb1a4b3b3db941590d2d1001b3a4808dfdc5cab78`.
The original distribution snapshot is [pilot_distribution.yaml](pilot_distribution.yaml).
One slow rover episode had an optimizer convergence in one iteration, no adjusted or
unresolved recovery-label frames, and no contacts. Those diagnostics do not explain its
wall-time outlier; subsequent generation records setup/environment/record timing separately.

Geometry gate: 13 tests passed in Modal's GPU environment (8.3 s test execution),
app `ap-rTliORbyn5a2rRrEF0C1sP`; [extreme-placement previews](extremes.jpg) inspected.
Two CPU-only verification attempts crashed in Triton import, including after preloading
PyTorch. The successful GPU run does not establish import-order as the cause.

## Pilot and numerical checks

Pilot training (`ap-YmjvzqemK96zmH93U7kEIx`) ran 200 steps per arm, with the
vision trunk frozen and identical data/order seeds. Image-only selection was step
100; conditioned selection was step 150. Training validation ADE was respectively
3.461 m and 3.016 m (initialized baseline 5.480 m). The separate
[paired temporal evaluation](pilot_paired_validation.json) excludes five warm-up
frames: baseline 5.118 m, image-only 3.525 m, conditioned 3.113 m.
These short straight-course episodes establish a working pipeline, not cornering transfer.

The [pilot context intervention](pilot_context_ablation.json) gives 3.113 m with
calibrated context, 3.073 m with default translation, 2.855 m with default camera
context and 4.147 m with default vehicle context. Thus this pilot supports use of
vehicle context but **does not establish benefit from camera conditioning**.
Validation histories are [image-only](pilot_image_validation.jsonl) and
[conditioned](pilot_conditioned_validation.jsonl).

An initial export comparison failed: maximum trajectory-element discrepancy was
0.3845 over 48 runtime ticks. A single-frame GPU/CPU intervention isolated cuDNN
TF32: CUDA-versus-ONNX plan difference fell from 0.10249 to 0.0000229 when TF32 was
disabled; hidden-state difference fell from 0.000393 to 0.000000313. Strict inference
precision now applies to feature caching, training validation and runtime. Old
TF32 caches have a distinct identity and remain preserved. The 48-tick integration
gate now requires plan difference below 0.001 and hidden difference below 1e-5.
These checks use ONNX Runtime's CPU provider; CUDA-provider parity is not established.

A physical-sensor reprojection test exposed an independent horizontal lens-offset
sign error. With cx=312 in a 640-pixel image, the old Panda lens projected 16 pixels
to the wrong side. The corrected film offset and an analytical lens regression
pass; physical-sensor rendering and runtime parity passed together on Modal (three
tests, 14.08 s, `ap-YiN5Cm7XqT7EAH0wrHpqQi`). The training distribution has centred
principal points, so existing generated observations were unaffected by this fix.

The [strict baseline open-loop rerun](default_baseline_strict_open_loop.json)
measured 1.390478 m ADE. Its [closed-loop rerun](default_baseline_strict_closed_loop.json)
still passes 3/3 trackdrive cases with two contacts and passes acceleration with
zero contacts. Skidpad remains a failure: 23 contacts and eight off-course events.
The change from the earlier skidpad result demonstrates closed-loop sensitivity to
small numerical differences; neither run establishes acceptable skidpad behavior.

## Generation performance investigation

Full generation began on eight CPU shards (`ap-lFjoDAxu0sEE5Zc6XrYJKA`). Two shards
were dominated by low-camera software rendering. Native sampling of a slow worker
showed Mesa `swrast_dri` rendering/readback, not planner or file writing. A one-second,
five-frame replay of `train_1_start` took 69.53 s with one software-rendering thread;
the same specification on a Modal GPU took 2.72 s (`ap-NfYstLdEFDWzDiSIueCiWT`).
The remaining CPU workers were stopped and unfinished episodes resumed on GPU
(`ap-SUdNjBR9gCIySZhPoiDutj`) using the dataset's archived source and original specs.
Completed episodes are preserved; `gpu_resume.json` records the backend change.
This dataset mixes CPU and GPU rasterization. Future transfer generation uses at
most three GPU workers to avoid this pathological CPU cost.

Cost reporting uses measured execution seconds and resource settings, not an
account invoice. [Modal's published rates](https://modal.com/pricing), checked
2026-11-13, are $0.000222/s for L4, $0.000306/s for A10, $0.000583/s for A100-40GB,
$0.0000131/physical-core-second and $0.00000222/GiB-second. Startup, idle retention,
failed probes and orchestrator containers can add charges beyond model execution.

Full training data: [manifest](training_manifest.json), [quality audit](training_quality.json),
[recorded observations with projected trajectories](training_observations.jpg), and
[GPU continuation record](gpu_resume.json). There are 86 episodes / 6,664 frames:
16 training configurations, four validation configurations, four scenarios each,
and six legacy replay episodes. Validation configurations are disjoint from training.
The label audit found 69 unresolved recovery frames (1.04%); these are retained and
reported, not represented as collision-free expert supervision. Difficult recovery
and legacy replay episodes also contain actual cone contacts. The observation sheet
was inspected before launching the larger training runs.

Full repository test suite on Modal: **68 passed**, 77.21 s, app
`ap-aWGLtLGSDYjmlSAOXd9h8O`. This includes the real recorded-sequence export gate
and a fresh-versus-reused-vision context-ablation equivalence check. Subsequent
mission-brake output clamping passed its targeted local geometry/controller tests;
physical dynamics already clamped that command, so recorded training states are unchanged.

The finite draws do not fill the nominal distribution. [Exact sampled profiles](training_profiles.json)
show training camera x/y/z ranges of [−0.775, 0.420]/[−0.279, 0.266]/[0.864, 1.586] m
for Formula Student and [−0.232, 0.157]/[−0.109, 0.043]/[0.312, 0.622] m for rovers.
In the predeclared suite, “interpolation” means inside the **configured sampling
bounds**, not inside the coordinate-wise envelope of these 16 finite draws. Low,
far-forward and some lateral test placements therefore also probe extrapolation
beyond observed training coordinates. Report that distinction alongside the nominal
in-distribution/out-of-distribution comparison; a successful nominal-range test
must not be described as dense coverage of every combination.

Modal `cpu` requests denote physical cores, not vCPUs; billing uses the higher of
requested or actual CPU/memory usage ([resource documentation](https://modal.com/docs/guide/resources)).
Resource-based cost calculations therefore use the requested physical-core count
and remain estimates rather than invoice totals.

The image-only full run selected step 1,200 at validation ADE 1.801030 m
([history](image_validation.jsonl)); 2,000 training steps took 164.64 s
([timing](image_training_timing.json)) on A100-40GB. The two arms use the same
global seed and data splits, but adapter initialization consumes additional RNG;
minibatch order is not guaranteed identical. This is a single-run comparison,
not a multi-seed estimate of architecture variance.

## Bounded revision before final transfer tests

Full paired validation (episode-mean ADE, after five warm-up frames) was baseline
6.0403 m, image-only 1.6856 m and conditioned 1.7768 m
([paired results](paired_validation.json)). The original conditioned adapter's
[interventions](context_ablation.json) gave 1.9823 m with default translation,
4.3571 m with default vehicle parameters, 2.0670 m with default footprint,
2.5311 m with default speed limit, 2.3740 m with default dynamic limits and 1.9969 m
with default response parameters. Defaulting all camera inputs instead gave
1.6405 m, showing that the combined camera context was not uniformly beneficial.

The original adaptations failed default retention: image-only ADE 1.890974 m and
conditioned ADE 1.640430 m; both passed 3/3 trackdrive cases, but their total contacts
were ten and five respectively, versus the strict baseline's two. See
[image open-loop](default_image_open_loop.json), [image closed-loop](default_image_closed_loop.json),
[conditioned open-loop](default_conditioned_open_loop.json) and
[conditioned closed-loop](default_conditioned_closed_loop.json).

One additional 2,000-step run froze the original policy and learned
`adapter(hidden, context) - adapter(hidden, baseline_context)`. At the reference
configuration this correction is exactly zero, even after training. The reference
vector is stored in the checkpoint; this is a continuous correction, not a switch
between named vehicles. [The configuration](../../configs/train/transfer_anchored_v1.yaml)
uses the same dataset and seed, with no new final-test observations.
The selected checkpoint is step 800 (validation ADE 1.985 m).
[History](anchored_validation.jsonl), [timing](anchored_training_timing.json).

Default retention improved to ADE 1.395257 m, 3/3 trackdrive passes and one contact
([open-loop](default_anchored_open_loop.json), [closed-loop](default_anchored_closed_loop.json)).
Skidpad remains a failure (29 contacts, seven off-course events); preserving a
reference policy does not repair its pre-existing mission/prediction failures.

The anchored [camera-angle-only intervention](anchored_context_ablation.json)
improved episode-mean validation ADE from 2.2083 m to 1.8651 m, while defaulting
translation worsened it to 2.4292 m. Consequently, a deterministic specialization
removes the three angle columns of the adapter's first layer and removes those
inputs from context version 2 (13 inputs). This reproduces the zero-angle
intervention without another training run. Actual angles still drive rendering,
projection and calibrated physical-camera offsets. This result is not evidence
that orientation is universally irrelevant; it is evidence against retaining
those learned angle inputs in this candidate.

The original anchored checkpoint and optimizer state remain intact. The specialized
checkpoint is marked inference-only with its source checkpoint hash/step and can
initialize a fresh training run, but cannot resume the original optimizer.
