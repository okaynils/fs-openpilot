# Camera and vehicle transfer (experimental)

Keep `cuda_racing_v6` as the default. The selected candidate improves held-out
trajectory error and preserves the default trackdrive benchmark, but camera-transfer
failures prevent recommending replacement. See the [experiment report](../benchmarks/transfer/report.md).

Teams supply camera calibration and a vehicle profile; the new conditioned policy
contract does not require a team-specific model. The reusable candidate distribution
is [transfer_v1.yaml](../configs/distributions/transfer_v1.yaml), with generic
[Formula Student](../configs/profiles/generic_formula.yaml) and
[Ackermann rover](../configs/profiles/generic_rover.yaml) examples.
These are simulation envelopes, not claims of hardware validation. Failures also
occur inside the nominal bounds; this is an experimental capability, not a verified
configuration range for unattended deployment.

| Parameter | Formula Student family | Small Ackermann family |
|---|---:|---:|
| Wheelbase | 1.395–1.705 m | 0.36–0.54 m |
| Width | 1.26–1.54 m | 0.32–0.48 m |
| Camera forward offset | −0.8–1.1 m | −0.25–0.35 m |
| Camera left offset | −0.3–0.3 m | −0.12–0.12 m |
| Camera height | 0.75–1.65 m | 0.22–0.65 m |
| Configured maximum speed | 7–18 m/s | 1–3 m/s |
| Maximum steering angle | 20–28° | 24–35° |
| Steering response time | 0.04–0.12 s | 0.08–0.25 s |
| Acceleration response time | 0.08–0.2 s | 0.12–0.35 s |
| Acceleration capability | 3.5–7 m/s² | 0.6–2 m/s² |
| Braking capability | 6–13 m/s² | 1–3.5 m/s² |

Both families sample pitch −3–8°, roll ±3°, yaw ±6°, sensor horizontal FOV
62–90° and 640×360, 960×540 or 1280×720 resolution. Length, axle distances,
overhangs, mass, inertia and tire stiffness change coherently around each
representative vehicle. A protruding camera support is included conservatively in
the longitudinal footprint. The current vehicle model is a bicycle model with
first-order actuator response; it does not model skid steering, articulation,
rough-terrain suspension, wheel lift, actuator deadband or network transport delay.
It is not a distribution over arbitrary four-wheeled vehicles.

## Coordinates and calibration

All vehicle geometry uses the ground projection of the centre of gravity as its
origin: x forward, y left, z up. Predicted trajectory positions remain relative to
this origin even when the camera moves. Measure axle offsets and footprint from
the same reference. Camera mount translations are in metres; profile angles are
in degrees. Positive pitch points the camera downward. Rotation order is
`Rz(yaw) @ Ry(pitch) @ Rx(roll)`.

`mount` is the calibrated virtual-camera installation. Each physical camera supplies
its resolution, intrinsics, distortion model and angular offset relative to that
installation. The current slots share one optical centre: one physical camera can
supply both road and wide slots. Separated stereo camera centres are not supported
by this contract. Intrinsics and angular offsets are removed through the deployment
remap, but camera translation and height remain. A homography cannot remove general
translation-induced parallax without scene depth.

The final candidate uses geometry context version 2 (13 inputs). `FSDriver` builds
it from the profile: camera translation, wheelbase, width, front/rear extents,
steering/speed/acceleration/braking limits and actuator time constants. Camera
angles remain required for calibration and projection, but were removed from the
learned policy input after a validation ablation. Earlier experimental checkpoints
use version 1 (16 inputs, including angles); the runtime supports both contracts.
The exact ordering and normalization are in
[conditioning.py](../src/fsop/model/conditioning.py). Exported models expose an
additional `geometry` input; use the runtime or this documented encoder, not an
unversioned hand-built tensor. Checkpoints without conditioning retain the legacy
input contract. Unknown conditioned contracts are rejected. The anchored candidate
freezes the original policy and subtracts its adapter response at a stored baseline
configuration; the correction is zero at that configuration. Nearby configurations
still require evaluation. The 13-input candidate is a deterministic specialization
of the selected anchored checkpoint, not an additional team-specific training run.

Installation is fixed per episode. Small calibration errors perturb the estimate
available to the policy; smooth mounting motion perturbs the actual rendered pose
without revealing instantaneous simulator pose to the policy. The distribution
uses ±5 mm and ±0.25° calibration errors and up to 2 mm/0.2° sinusoidal mounting
motion with base frequency 0.5–2 Hz (axis multipliers 0.67–1.0). These are experimental tolerances, not established deployment
acceptance limits. Episode files store actual per-frame camera transforms for
supervision and the distinct estimated installation for policy inputs.

## Preprocessing and visibility

Transfer simulation renders a physical rectilinear sensor, applies image degradation,
and uses the same `ModelFrameWarper` and packed YUV conversion as deployment.
Source FOV gaps remain black and are excluded from detection targets. Deployment
supports calibrated OpenCV radtan/fisheye remapping, but distorted sensor simulation
is not yet implemented and is explicitly rejected. The distribution therefore
covers rectilinear sensors only; distortion robustness is not established.

A low rectangular chassis approximates visible bodywork. The same ray/box geometry
masks images and target visibility. It does not reproduce wheels, wings, suspension,
reflections or vehicle shadows. Knocked cones are removed from upright detection targets using recorded contact events.
Visibility labels test the cone base, so a partially visible tip can be excluded when its base is hidden.
Cone-on-cone occlusion and realistic fallen-cone geometry remain simulator limitations. Low rear-mounted cameras can lose most of
the narrow slot to the vehicle: see [extreme-placement examples](../benchmarks/transfer/extremes.jpg).
No amount of conditioning can recover a track boundary hidden from both cameras.

## Operation and evidence limits

The runtime expects 20 Hz calls and maintains the original frame-pair and feature
history queues. Reset it between independent sequences. It uses measured wheel
speed for control; no retraining is needed to supply a new vehicle profile.
The launch/stop/lap state machine is external to deployment. Simulation comparisons
must disclose their use of simulator speed, desire, progress and mission assistance.

A model may still need adaptation when appearance, course geometry, dynamics,
calibration error or visibility lies outside the measured range. Validate on recorded
real data and in controlled vehicle tests before drawing conclusions about physical
transfer. Simulation success does not establish real-vehicle readiness.

## Reproducing the experiment

Run from the repository root with the existing Modal account and `fsop` volume.
The selected baseline is never overwritten. Dataset manifests record seeds,
configuration specifications, source hashes and the baseline hash. New training
runs archive their source and save `best.pt`, `last.pt`, optimizer/RNG state and
validation history. Keep run and dataset names distinct when changing inputs.

```bash
modal run scripts/modal_app.py::transfer_verify
modal run scripts/modal_app.py::transfer_generate
modal run scripts/modal_app.py::transfer_pilot
modal run scripts/modal_app.py::transfer_full
modal run scripts/modal_app.py::transfer_audit
modal run scripts/modal_app.py::transfer_train
modal run scripts/modal_app.py::transfer_analyze \
  --dataset data/sim/transfer_train_v1 \
  --image-run transfer_image_v1 --conditioned-run transfer_conditioned_v1 \
  --name transfer_analysis_v1
```

Inspect validation and freeze the selected checkpoints before running
`transfer_test_data` and `transfer_closed`. `transfer_closed --runtime --name
transfer_runtime_v1` evaluates the exported runtime on a bounded subset.
`transfer_finish` resumes an interrupted training-data generation job using its
saved source archive; stop its original workers first. The original full dataset
was partly CPU-rendered; the continuation and future transfer generation use GPU
rendering, as recorded in the experiment report.

To resume interrupted model training, use the general `train` entrypoint with the
same config and run name plus `resume=runs/<name>/ckpt/last.pt` in `--o`.
The optimizer and RNG states resume; shuffled loader position is not bitwise
reconstructed. Use the run's archived source and preserve its dataset manifest.
A completed existing run is not automatically retrained or overwritten.

Download artifacts with `modal volume get fsop runs/<name>/ckpt/best.pt
runs/<name>/ckpt/`. Exports and parity reports live in the named analysis directory.
The [experiment report](../benchmarks/transfer/report.md) records the actual runs,
validation decisions, failures and measured limitations; these commands alone do
not establish model quality.

The current finite training sample does not densely cover those bounds. In
particular, low and far-forward held-out placements extend beyond the observed
training coordinates even when they lie inside the configured distribution.
Consult the report's exact profile list and per-configuration results before
interpreting a range as supported behavior.

The specialized `transfer_minimal_v1` checkpoint is inference-only: its first-layer
columns were pruned to reproduce the selected angle ablation. Its provenance links
to the resumable `transfer_anchored_v1` training run. To train from the specialized
weights, use `init_ckpt` in a fresh run with `context_version=2`, not `resume`.
Vision caches remain reusable because geometry is encoded separately per sample.

### Selected revision and final evaluation commands

These are the commands used after the first two full-training arms. Existing run
names are immutable; use archived sources to inspect/resume an original run and
new names for a new experiment. The full-generation/training orchestration uses
fixed dataset/run names, so change those explicitly in a new experimental version
rather than silently reusing `transfer_train_v1` with different geometry.

```bash
modal run scripts/modal_app.py::train --cfg configs/train/transfer_anchored_v1.yaml
modal run scripts/modal_app.py::sh --cmd "python -c \"import runpy; runpy.run_path('scripts/transfer_experiment.py')['derive_minimal']()\""
modal run scripts/modal_app.py::transfer_analyze \
  --dataset data/sim/transfer_train_v1 --image-run transfer_image_v1 \
  --conditioned-run transfer_minimal_v1 --name transfer_minimal_analysis_v1
# Only after validation selection is frozen:
modal run scripts/modal_app.py::transfer_test_data
modal run scripts/modal_app.py::transfer_open \
  --conditioned-run transfer_minimal_v1 --extra-run transfer_conditioned_v1 \
  --name transfer_open_v1
modal run scripts/modal_app.py::transfer_closed \
  --conditioned-run transfer_minimal_v1 --extra-run transfer_conditioned_v1 \
  --name transfer_closed_isolated_v1
modal run scripts/modal_app.py::transfer_closed \
  --conditioned-run transfer_minimal_v1 --runtime \
  --export-run transfer_minimal_analysis_v1 --name transfer_runtime_isolated_v1
modal run scripts/modal_app.py::bench --name transfer_default_minimal_v1 \
  --ckpt runs/transfer_minimal_v1/ckpt/best.pt
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_diagnostics.py'
modal run scripts/modal_app.py::sh --cmd 'python -m pytest tests -q'
```

The standard `train` and `bench` entrypoints spawn work and print its app/call ID;
wait for that app to complete before a dependent command. The transfer entrypoints
wait for their stage to finish. Generation and analysis identities reject incompatible
reuse; existing derived checkpoints are never overwritten.

The candidate is available locally at `runs/transfer_minimal_v1/ckpt/best.pt`, with
export `artifacts/export/transfer_minimal_v1/conditioned.onnx`. The same export lives
on Modal volume `fsop` at `runs/transfer_minimal_analysis_v1/conditioned.onnx`.
[Artifact hashes](../benchmarks/transfer/artifacts.json) identify the exact files.
For downloads, pass an explicit destination filename:

```bash
modal volume get fsop runs/transfer_minimal_v1/ckpt/best.pt /tmp/transfer_minimal_best.pt
modal volume get fsop runs/transfer_minimal_analysis_v1/conditioned.onnx /tmp/transfer_minimal.onnx
```

Training source archives, optimizer checkpoints and manifests remain on the volume
under their original run/dataset directories. `transfer_anchored_v1/ckpt/last.pt` is
the resumable source run; the derived candidate does not contain optimizer state.

For a fresh repeat against the preserved dataset/checkpoints from the current
working tree, use distinct output names, for example:

```bash
modal run scripts/modal_app.py::train --cfg configs/train/transfer_anchored_v1.yaml \
  --o 'name=transfer_anchored_repro_v1'
modal run scripts/modal_app.py::transfer_closed \
  --conditioned-run transfer_minimal_v1 --extra-run transfer_conditioned_v1 \
  --name transfer_closed_recheck_v1
```

This trains a new run or rechecks frozen models; it does not recreate the original
source snapshot. For historical reproduction, restore the corresponding archived
source/configuration first. The final test configurations are already disclosed
and must not be reused for selecting a new candidate; reserve new complete
configurations for any further iteration.
