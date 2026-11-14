# Camera and vehicle transfer experiment

**Recommendation: retain `cuda_racing_v6` as the default.** The selected shared
candidate improves held-out metric trajectory error and retains the default
trackdrive benchmark, but does not establish reliable closed-loop transfer across
the proposed camera/vehicle envelope. No team-specific weights or integration were
added. This is simulator evidence, not real-vehicle validation.

The candidate records **0/12 transfer referee passes**, versus **7/12** for the
baseline and **12/12** for the privileged expert. Baseline transfer runs still
incur 160 contacts, so its seven penalized passes do not establish safe transfer
either. The robust shared-model outcome was **not achieved** in this experiment.

Implementation is on `camera-vehicle-transfer-v1`, based on
`bd408fa54b1961719688a8f06471ad02c52311bb`, with no commit or release. The original
checkpoint remains unchanged. [Chronological audit](audit.md) records the initial
inspection, gates, discovered defects and validation decisions.

## Experiment and selection

[The versioned distribution](../../configs/distributions/transfer_v1.yaml) covers
Formula Student and small Ackermann vehicles, coherent geometry/dynamics, physical
sensor intrinsics and quality, fixed installations, calibration error and smooth
mount motion. The [public guide](../../docs/transfer.md) gives explicit bounds,
coordinates, preprocessing and limitations. All predictions use the ground-projected
vehicle centre of gravity, independent of camera position.

A 12-episode/611-frame straight-course pilot preceded the full 86-episode/6,664-frame
training/validation dataset. Full data contains 16 training and four validation
configurations, each with driving, disturbed recovery, start and stop scenarios,
plus six legacy replay episodes. Configuration partitions are explicit. The
[label audit](training_quality.json) retains and discloses 69 unresolved recovery
frames (1.04%). [Rendered training examples](training_observations.jpg) were inspected
before full training. The frozen vision trunk uses provenance-checked FP32 caches.

Two 2,000-step runs compared image-only adaptation and a 16-input calibrated context
adapter. They improved transfer validation but regressed the default benchmark.
One bounded 2,000-step revision froze the original policy and learned an anchored
context residual, exactly zero at the stored reference configuration. Validation
selected step 800; an angle-input ablation improved validation further. Pruning the
three angle columns produced the 13-input inference-only candidate, with recorded
sequence equivalence error below 0.000018. No final-test data selected weights or
inputs. [Selection record](selection.json), [specialization](specialization.json).

Image-only monocular observations cannot generally distinguish metric scale and
camera translation across this range. Warping removes calibrated sensor rotation
and intrinsics, but cannot synthesize a translated viewpoint without depth. Context
is therefore a plausible mechanism, not proof of successful transfer. The retained
inputs are camera xyz, wheelbase, footprint, steering/speed/acceleration/braking
limits and response constants. [Encoder and versions](../../src/fsop/model/conditioning.py).
The [adapter](../../src/fsop/model/build.py) modifies the current vision embedding;
its temporal history remains unconditioned, which preserves existing feature caches.
It does not implement an explicit camera-to-vehicle geometric conversion. This is
a bounded conditioning experiment, not evidence that this architecture is sufficient.

Single global seeds and model-dependent RNG consumption mean these are bounded
single-run comparisons, not estimates of architecture variance across training seeds.

## Held-out trajectory prediction

[Final paired results](paired_test.json) use six complete unseen configurations,
two tracks/seeds each, identical recorded observations for every model, and the
same warm-up exclusion. ADE is mean Euclidean position error through four seconds,
first averaged per episode and then equally across configurations.

| Model | Selected step | Within nominal bounds ADE (m) | Outside bounds ADE (m) | Predicted-contact frame fraction, within bounds |
|---|---:|---:|---:|---:|
| Original baseline | existing | 5.7893 | 8.9070 | 0.6356 |
| Diverse image-only | 1200 | 2.3660 | 4.1270 | 0.5881 |
| Unanchored context v1 | 400 | 2.7072 | 4.7441 | 0.5384 |
| Selected anchored context v2 | source 800 | 3.1131 | 3.7595 | 0.5083 |

The selected candidate reduces within-bounds ADE by **46.2%**, exceeding the
predeclared 15% error criterion. Image-only adaptation has lower aggregate transfer
ADE but fails default retention. Predicted contacts measure planned swept footprint
intersection with cones; they are not actual closed-loop contacts. Expert trajectories
have zero predicted-contact frame fraction on these final recorded sequences.

| Held-out configuration | Baseline ADE | Image-only ADE | Context v1 ADE | Selected v2 ADE |
|---|---:|---:|---:|---:|
| Formula low/forward/lateral | 4.5075 | 4.2013 | 5.2978 | 7.2479 |
| Formula high/rear/lateral | 3.6029 | 3.6779 | 3.6775 | 3.6799 |
| Rover low/forward/lateral | 9.5760 | 0.8696 | 1.1130 | 0.5985 |
| Rover rear/lateral | 5.4709 | 0.7154 | 0.7406 | 0.9262 |
| Formula outside high/forward | 6.8768 | 4.7701 | 5.2452 | 6.9967 |
| Rover outside low/rear | 10.9372 | 3.4839 | 4.2430 | 0.5223 |

“Within bounds” is the suite's nominal `interpolation` category. The finite training
sample does **not** cover the full sampling bounds: Formula x tops out at 0.420 m
and rover x at 0.157 m; minimum heights are 0.864 and 0.312 m. Thus low/forward test
placements also extrapolate beyond observed training draws. The high/rear Formula
case is within the observed coordinate ranges, but uses an unseen combination.
[Exact training profiles](training_profiles.json), [final configurations](../../configs/eval/transfer_v1.yaml),
[final data manifest](test_manifest.json). No universal coverage claim follows from
these six configurations.

## Existing default benchmark

| Model | Legacy ADE (m) | Trackdrive passes | Trackdrive contacts | Acceleration contacts | Skidpad contacts / off-course |
|---|---:|---:|---:|---:|---:|
| Strict FP32 baseline | 1.3905 | 3/3 | 2 | 0 | 23 / 8 |
| Image-only | 1.8910 | 3/3 | 10 | 0 | 3 / 5 |
| Context v1 | 1.6404 | 3/3 | 5 | 0 | 14 / 5 |
| Anchored v1 | 1.3953 | 3/3 | 1 | 0 | 29 / 7 |
| Selected v2 | 1.3866 | 3/3 | 2 | 0 | 18 / 1 |

[Baseline](default_baseline_strict_closed_loop.json), [image](default_image_closed_loop.json),
[context](default_conditioned_closed_loop.json), [anchored](default_anchored_closed_loop.json),
[selected](default_minimal_closed_loop.json), [selected open-loop](default_minimal_open_loop.json).
All skidpad runs fail. The selected candidate satisfies the predeclared default
retention criteria, but preservation of an imperfect baseline is not a safety claim.

## Closed-loop transfer results

[Corrected paired results](closed_loop.json) contain 60 unique evaluations. Completion below means referee pass; cone contacts are reported separately because a penalized finish can still pass.

| Nominal regime / model | Passes | Contacts | Mean progress | Worst clearance (m) | Mean RMS lateral (m) | Mean steering-rate RMS (rad/s) |
|---|---:|---:|---:|---:|---:|---:|
| interpolation / Baseline | 7/8 | 96 | 0.922 | -0.850 | 7.414 | 0.091 |
| interpolation / Image-only | 2/8 | 67 | 0.854 | -0.849 | 18.369 | 0.066 |
| interpolation / Context v1 | 1/8 | 78 | 0.666 | -0.849 | 15.188 | 0.095 |
| interpolation / Selected v2 | 0/8 | 55 | 0.625 | -0.849 | 13.767 | 0.108 |
| interpolation / Expert | 8/8 | 0 | 1.000 | 0.150 | 0.660 | 0.171 |
| extrapolation / Baseline | 0/4 | 64 | 0.750 | -0.849 | 3.372 | 0.090 |
| extrapolation / Image-only | 0/4 | 4 | 0.424 | -0.641 | 40.830 | 0.061 |
| extrapolation / Context v1 | 0/4 | 23 | 0.260 | -0.849 | 87.481 | 0.081 |
| extrapolation / Selected v2 | 0/4 | 24 | 0.264 | -0.840 | 18.482 | 0.110 |
| extrapolation / Expert | 4/4 | 0 | 1.000 | 0.150 | 0.788 | 0.167 |

| Configuration | Baseline passes / contacts | Candidate passes / contacts | Baseline / candidate mean progress |
|---|---:|---:|---:|
| formula_low_forward | 2/2 / 35 | 0/2 / 26 | 1.000 / 0.166 |
| formula_high_rear | 2/2 / 52 | 0/2 / 27 | 1.000 / 1.000 |
| rover_low_forward | 1/2 / 9 | 0/2 / 2 | 0.686 / 0.491 |
| rover_rear_offset | 2/2 / 0 | 0/2 / 0 | 1.000 / 0.841 |
| formula_outside_high | 0/2 / 45 | 0/2 / 20 | 1.000 / 0.073 |
| rover_outside_low_rear | 0/2 / 19 | 0/2 / 4 | 0.500 / 0.455 |

| Initial-recovery cases, all six configurations | Mean lateral deviation at 4–6 s (m) | Mean 1 s own-plan tracking error (m) | Maximum yaw rate (rad/s) | Mean steering saturation fraction |
|---|---:|---:|---:|---:|
| Baseline | 1.167 | 3.629 | 0.935 | 0.000 |
| Image-only | 0.994 | 1.072 | 0.820 | 0.000 |
| Context v1 | 1.102 | 1.144 | 0.810 | 0.000 |
| Selected v2 | 5.321 | 0.545 | 0.884 | 0.000 |
| Expert | 0.462 | n/a | 1.142 | 0.000 |

Lateral deviation is relative to the track centreline, not the optimized racing line; even the expert can intentionally run off-centre. Low steering rate or few contacts can accompany driving straight away from the course, so neither alone demonstrates stability. The raw report also provides per-case visibility, body-mask fraction, online expert-plan error, latency and stop/referee outcomes.

## Conditioning evidence and failure diagnosis

Validation interventions replace one calibrated parameter group with the original
vehicle's values, holding images and unconditioned vision history fixed. They are
counterfactual input tests, not retrained architecture ablations; correlated groups
and out-of-distribution combinations limit causal interpretation.

| Anchored v1 validation intervention | ADE (m) |
|---|---:|
| Calibrated context | 2.2083 |
| Default camera translation | 2.4292 |
| Default camera angles | 1.8651 |
| Default vehicle parameters | 5.8684 |
| Default footprint | 2.7213 |
| Default speed limit | 2.8408 |
| Default dynamics limits | 3.1096 |
| Default response constants | 1.8958 |

[Anchored interventions](anchored_context_ablation.json), [original interventions](context_ablation.json),
[final 13-input interventions](minimal_context_ablation.json). Translation and vehicle
context have evidence of use; camera angle inputs were harmful in this candidate.
Response constants are inconclusive: defaulting them helped anchored v1 but hurt the
unanchored model. They remain experimental physical inputs, not established benefits.

[Baseline rear-camera montage](rear_baseline.jpg) and [candidate montage](rear_candidate.jpg)
show visible cones but predicted trajectories outside the expert corridor
(red prediction, green expert). In a fixed track/scene diagnostic, nominal-camera
baseline completes with zero contacts, rear-camera baseline incurs 25, and the
candidate incurs 23. Feeding expert trajectories through the **same plan follower**
completes with one contact; the privileged expert controller has zero.
[Nominal-camera trace](reference_camera_diagnostic.json), [oracle tracking](oracle_tracking.json),
[reproduction source](../../scripts/transfer_diagnostics.py). These are diagnostic
probes after selection, not additional selection data. They support policy/viewpoint
error as a major failure source, without claiming to exclude every controller effect.

Failure attribution is deliberately narrower than the aggregate scores:

- **Visibility:** the high/rear candidate still sees a mean 6.54 eligible cones per frame in the diagnostic seed, yet incurs 23 contacts. It is not completely blind. Low/rear placements can lose near-ground pixels to bodywork/FOV; per-case mask and visibility statistics are retained, but low visibility after leaving the course can be an effect of prediction failure rather than its initial cause.
- **Calibration:** final closed-loop cases use exact installed calibration with no flex or injected error. Their failures therefore cannot be attributed to the sampled calibration-error distribution. Validation perturbations probe sensitivity, but do not establish closed-loop calibration tolerances.
- **Prediction:** the same-controller expert-plan intervention and the visible-corridor montages identify a substantial policy/viewpoint problem. Formula 1 s and 4 s lateral errors also worsen in the selected candidate.
- **Tracking:** high/rear seed 99001 has 1 s own-plan tracking error 0.964 m for the candidate versus 0.781 m for baseline. This measures the vehicle's response to its own changing plan, not an isolated controller accuracy test. The oracle-plan probe is evidence for that one configuration, not proof that tracking is solved throughout the envelope.
- **Mission logic:** `unsafe_stop` means stopping outside the referee's stop zone after mission progress reaches the finish. A `finished` result can still lack a valid counted lap. Launch and stopping assistance use simulator progress and speed in every learned-policy comparison, so starts/stops are not autonomous-policy-only successes.

## Export and numerical behavior

The selected [48-tick parity test](minimal_export_parity.json) has maximum plan-element
difference 0.0000954, hidden difference 2.50e-7, steering difference 6.75e-8 rad and
acceleration difference 5.07e-6. Recorded 5 Hz images are repeated four times through
the real 20 Hz queues. A separate physical-sensor closed-loop subset tests fresh
20 Hz observations through `FSDriver`, including remapping and temporal state.
ONNX uses CPU execution in these tests; CUDA ONNX parity is not established.

An early failed parity gate isolated TF32 as a material error source; inference,
validation and cache generation now disable it. Old caches have distinct identities.
Training and runtime both use 0.2 s image-pair and history spacing; there is no
fourfold frame-pair interval mismatch. The 5 Hz recordings omit the intermediate
20 Hz control phases, which limits what recorded-sequence evaluation alone can prove.
[Queue implementation](../../src/fsop/model/temporal.py), [dataset pairing](../../src/fsop/data/sim_dataset.py).

Small residual numerical differences can still alter cone collisions and closed-loop
trajectories; deployment parity does not imply bit-identical closed-loop outcomes.

## Exported runtime closed-loop check

[Raw-sensor runtime results](runtime_closed_loop.json) use two held-out configurations
and two seeds each. Torch and exported ONNX agree on the failure categories:

| Configuration / seed | Torch contacts | ONNX contacts | Torch / ONNX progress | Valid completion |
|---|---:|---:|---:|---|
| Formula high/rear 99001 | 23 | 23 | 1.000 / 1.000 | neither |
| Formula high/rear 99002 | 4 | 4 | 1.000 / 1.000 | neither; unsafe stop |
| Rover low/forward 99001 | 2 | 3 | 0.818 / 0.824 | neither |
| Rover low/forward 99002 | 0 | 0 | 0.164 / 0.165 | neither; off course |

The privileged expert completes all four with zero contacts. In the last case,
RMS lateral deviation exceeds 46 m: zero contacts reflects driving away from the
course, not safe driving. Torch steady-state policy p95 is 15.1–19.2 ms; CPU ONNX
including raw-sensor preprocessing is 36.4–50.6 ms. The latter marginally exceeds
the 50 ms budget in two cases, even before external capture/control overhead.
This is not an established real-time deployment guarantee.

An orchestrator preemption caused automatic retries in this app. The final report
contains 12 unique model/configuration/seed records, not repeated statistical
samples. Future reused workers explicitly reload the volume before checking the
immutable completed-case identity, to avoid stale cache checks after retries.

## Evaluation integrity

The first final closed-loop attempts (`transfer_closed_v1`, `transfer_runtime_v1`)
were stopped after discovering that renderer instances shared stale scene nodes
when Modal reused a worker. Their partial metrics are **excluded**. The renderer now
isolates each camera's scene and releases buffers/cameras in `finally`; a regression
test renders one scene before and after creating another and requires pixel identity.
Models, final configurations and seeds were unchanged for corrected runs
`transfer_closed_isolated_v1` and `transfer_runtime_isolated_v1`.

The same bug does not invalidate recorded training/final observations: data workers
reuse one renderer and explicitly remove it when camera signatures change. The
legacy benchmark uses one renderer per fresh benchmark process. Diagnostic captures
explicitly released each renderer before the next capture.

## Limitations and recommendation

Keep the baseline default and treat the candidate as a research artifact. The
bounded revision repaired default retention but did not solve Formula camera
transfer. Training at 5 Hz with history spacing and evaluating fresh 20 Hz frames,
a frozen vision trunk, sparse installation coverage and limited recovery exposure
are plausible contributors, not experimentally isolated causes. Future work should
first establish dense vehicle/camera configuration coverage and denser temporal
sequence coverage on new validation partitions; these final cases must not become a hidden
training-selection loop.

Low cameras can lose road pixels to the coarse chassis mask or physical FOV. Neither
conditioning nor image warping restores unobserved boundaries. Bodywork is a box,
not a CAD vehicle; rough terrain, suspension, wheel lift, latency/deadband and
realistic occlusion/materials remain unmodeled. Distorted sensor rendering and
separated physical camera centres are unsupported. The evaluator supplies true
wheel speed, route desire and mission progress for launch/stop. The expert additionally
uses the full racing line, true dynamics and cone map. No result here demonstrates
unassisted mission logic or real-world readiness.

## Artifacts and reproduction

[Artifact manifest](artifacts.json) records SHA256 and size of the unchanged baseline,
both first-round candidates, anchored source checkpoint, selected derived checkpoint
and ONNX export. Local paths are `runs/<run>/ckpt/best.pt` and
`artifacts/export/transfer_minimal_v1/conditioned.onnx`. Full resumable checkpoints,
source archives, exact configs, validation history and data manifests remain on
Modal volume `fsop`. The derived checkpoint is explicitly inference-only; use the
anchored source's `last.pt` to resume its optimizer, or initialize a fresh v2 run.

[Modal app ledger](modal_runs.json) identifies successful, stopped and excluded runs.
[Public reproduction commands](../../docs/transfer.md#selected-revision-and-final-evaluation-commands)
cover training, specialization, export, paired tests and diagnostics. Dataset identity
includes distribution/source hashes and exact configuration seeds; current generation
also hashes referenced profiles. Earlier generated data retains its exact archived
profiles and manifests rather than silently adopting a new hash schema. Feature
cache identity includes checkpoint, precision mode and episode metadata.

Final evaluation uses the same two tracks/seeds per configuration, including a
0.5 m / 0.08 rad initial recovery disturbance in the second case. It uses a
100-second cap, real 20 Hz observation/queue updates, fixed installed calibration
and light photometric variation. The final closed-loop suite does not vary mount
flex or calibration error; these are exercised by training/validation, interventions
and geometry tests, not established as robust closed-loop tolerances. Loop timing is
measured but does not delay simulated actuator commands when inference exceeds the
50 ms budget. Clearance is against the original cone positions, including cones
already knocked down; it is a conservative geometry diagnostic, not a count of new
contacts. Progress reaching 100% is distinct from referee-valid completion.

Final verification: **71 tests passed** in 80.97 s on Modal, including analytical
and rendered projection, translations/rotations, temporal mounting continuity,
profile/input contracts, legacy checkpoints, anchored identity and recorded runtime
parity. Six warnings are from upstream onnx2torch indexing deprecation. Ruff and
`git diff --check` pass. [Verification record](verification.json).
A prior test command without an explicit directory was stopped during discovery
because it traversed the mounted dataset volume; no tests had executed. The
corrected command is `python -m pytest tests -q`.

A final provenance review added a separate passing local regression test: rejected
resume attempts preserve the original run configuration, and same-path datasets
with changed manifests are rejected before model construction. Successful resumes
record a separate configuration file. This safeguard was added after model training;
it changes neither weights nor evaluation behavior.

The open-loop improvement is not just a change in one scalar score. On nominal
rover configurations, the candidate reduces mean 1 s longitudinal/lateral error
from 5.654/1.724 m to 0.500/0.119 m. On Formula configurations, the corresponding
errors worsen from 1.934/1.030 m to 2.034/1.687 m; 4 s lateral error rises from
9.182 to 14.589 m. Even the rover improvement on expert-recorded states does not
ensure recovery from the states reached by its own closed-loop predictions.
This distinction prevents selecting a shared default solely by average ADE.

## Compute costs

Modal's metered usage snapshot for the 32 identified experiment apps totals **$7.8565**, retrieved at `2026-11-13T15:53:37.396297+00:00`. This includes stopped attempts, automatic retries and resource overhead attributed to those apps. It is metered usage before account credits/adjustments, not an invoice; late-settling usage, untracked initial probes and shared volume storage may add costs.

| Category | Metered USD |
|---|---:|
| Default benchmark reproductions and regressions | 1.0614 |
| Data generation / audit, including CPU continuation | 0.6695 |
| Successful geometry / compatibility tests | 0.1404 |
| Training pipelines (pilot + three full arms) | 0.9092 |
| Validation, export, specialization and final open-loop | 0.1384 |
| Failure and rendering diagnostics | 0.0686 |
| Stopped evaluation / discovery attempts | 1.6118 |
| Final corrected closed-loop evaluations | 3.2572 |

Resource breakdown: A10G $4.5596, Memory $0.9404, CPU $1.6911, Network Egress $0.0009, A100-40GB $0.4608, L4 $0.2037.

[Filtered metered records](modal_metered_costs.json), [app ledger](modal_runs.json), and [execution-time estimates](compute_costs.json) retain the accounting inputs. The estimates cover only measured stage execution and must not be added to the metered total.

The three full fits took 164.64, 165.36 and 151.66 seconds on A100-40GB (2,000 steps each); the full cache took 90.55 seconds. The original two-arm orchestration took 662.46 seconds including staging/cache/validation. GPU data continuation took 305.90 seconds of pipeline wall time. Corrected final closed-loop worker time is recorded per case and summed in `compute_costs.json`.

Prices used for execution-only estimates come from [Modal pricing](https://modal.com/pricing): L4 $0.000222/s, A10 $0.000306/s, A100-40GB $0.000583/s, CPU $0.0000131/physical-core-second and memory $0.00000222/GiB-second. Requested physical cores, not vCPUs, determine the CPU estimate ([resource documentation](https://modal.com/docs/guide/resources)).
