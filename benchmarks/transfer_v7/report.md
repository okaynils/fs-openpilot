# Second-round correction pilot

The original collection pilot failed its quality gate. A dynamics-aware collection
safeguard subsequently passed; the current-collector training arm completed, while
the paired control was interrupted and its retry remains blocked by Modal spending
capacity. The completed supplemental CPU comparison rejects the current candidate:
it passes 0/2 rover cases versus 1/2 for v1. Keep the existing default model.

The sections below retain the experiment chronology and intermediate results.

| Configuration | Collector | Valid labels | Contacts | Expert control |
|---|---|---:|---:|---:|
| train_v2_00 | previous | 62/71 | 3 | 57.2% |
| train_v2_00 | current | 64/71 | 5 | 75.7% |
| train_v2_03 | previous | 200/200 | 0 | 39.9% |
| train_v2_03 | current | 200/200 | 0 | 41.9% |
| train_v2_04 | previous | 79/83 | 1 | 60.3% |
| train_v2_04 | current | 102/102 | 0 | 41.7% |
| train_v2_13 | previous | 200/200 | 0 | 70.8% |
| train_v2_13 | current | 200/200 | 0 | 55.1% |

[Raw results](collection_report.json), [manifest](collection_manifest.json),
[predeclared plan](plan.md), [collector](../../src/fsop/sim/transfer_collection.py).
All starts were stationary. The four rover episodes lasted the full 40 seconds;
full-size episodes ended earlier. The current rover mean expert-control fraction
was 48.5%, below the 75% quality ceiling. These privileged expert interventions
are data collection assistance, not autonomous driving success.

The exact source archive is retained at
`artifacts/experiments/transfer_refresh_pilot_v7/source.tar.gz`.
Worker time was 388.58 seconds, excluding orchestration/startup. Metered cost
is not yet retrieved. Expert-only controls and saved-frame inspection follow
the gate failure; no checkpoints were changed and final holdouts remain sealed.

## Inspection and safeguard diagnosis

All eight saved-frame previews were visually inspected, including initial, early,
midpoint, final and first invalid-label observations where present. Low-camera
rover views contain nearby large cones and, for rear mounting, substantial body
occlusion. Later track sections can leave the image despite valid privileged
expert targets; valid labels do not imply the full trajectory is visually observable.
[Inspection traces and gates](inspection.json) confirm validity/contact gates fail
while the rover intervention-fraction gate passes.

The expert-only control for train_v2_00 produced zero contacts and 70/70 valid
labels. At t=4.0 the current collector let the policy act with steering −0.0398 rad
versus expert −0.2014 rad at 8.69 m/s. This fell below the fixed 0.20-rad safeguard.
By t=4.4 the expert target was invalid, followed by a contact at t=4.5. A new
optional speed/wheelbase-aware discrepancy trigger is now under a paired replay
test, as specified in [the plan](plan.md). Defaults preserve old behavior.

The two targeted guard tests failed before implementation; the guard and existing
reference/reuse tests subsequently passed (**6 passed**). This verifies trigger
semantics and compatibility, not collection quality or improved autonomous driving.

## Dynamic safeguard result

The [paired replay](guard_probe_report.json) passed: both episodes now have 70/70
valid labels and zero contacts. Previously they had 62/71 and 64/71 valid labels,
and 3 and 5 contacts. Only the additional dynamic trigger changed. Expert-control
fractions rose to 86.3% and 97.5%, so this fast full-size case contributes little
model-driven exploration. This demonstrates improved collection safety on the
case, not a better autonomous policy.

The remaining six episodes are being recollected with the same trigger to check
the full gate, including rover expert-control fraction. The two completed probe
episodes are copied byte-for-byte with metadata hashes, not rerun. Training still
requires the whole pilot and rendered inspection to pass.

## Guarded pilot accepted for bounded training

[Guarded results](guarded/report.json) and [inspection](guarded/inspection.json):
all eight episodes have 100% valid labels and zero contacts. Current-rover expert
control averages 47.375%, leaving a majority of ticks under learned-policy control.
All eight guarded previews were visually inspected. The same low-view/bodywork
limitations remain; some turn states have no visible track even though the
privileged expert target is valid. This does not establish observability of every
label. Both training arms retain those cases consistently.

The paired datasets add four matched correction episodes each to identical old
data. Surplus labels are masked at matching frame indices; images/history are
retained. Staging verifies exact baseline/initialization vision tensor equality
before allowing feature-cache reuse. The paired-mask unit test passes.

Staging completed: 552 new labels per arm (70, 200, 82, 200), with identical
metadata hashes for all retained episodes. All 270 vision tensors match the
baseline used to build caches. [Previous-arm manifest](previous_staging_manifest.json),
[current-arm manifest](current_staging_manifest.json). The paired 300-step
training job has been launched with the fixed configs in the plan.

The [completed expert-only control](expert_control_00.json) is preserved. Its
remaining CPU batch was deliberately cancelled after over ten minutes without
another episode result; this batch is partial, not a four-case success. The
completed control plus the paired safeguard replay support the narrower diagnosis.

The paired training workers each report 10,351 training samples (120 episodes)
and 1,910 validation samples. The unchanged initialization has identical baseline
validation ADE 1.948 in both arms. Current-refresh reached step 300 with ADE
1.933; this small offline change does not authorize scaling. The fixed selection
script rejects ties and unchanged-initialization winners; its targeted tests and
the existing selection test pass (2 tests).

## Compute interruption

Current-refresh completed 300 steps in 35.50 fit seconds, peak CUDA allocation
654,723,584 bytes ([timing](current_timing.json), [validation log](current_val.jsonl)).
The previous arm stopped after logging step 175. Its preserved last checkpoint
is loadable at **step 100**, with optimizer/RNG state
([checkpoint inspection](interrupted_checkpoint.json)); no step-300 or timing
artifact exists. No explicit worker-failure cause was available in server logs.
The stale orchestrator was stopped after Modal reported zero workers.

A single technical retry was prepared under a distinct name to preserve the
paired sample stream (the Trainer does not save its in-epoch sampler cursor).
Modal rejected that launch because the **workspace spending limit was exceeded**
([interruption record](compute_interruption.json)). No retry job was created.
This service rejection does not prove why the earlier worker stopped.

Driving validation has not run. The current model must not be promoted based on
its small offline ADE change. The goal remains unmet. Metered v7 usage currently
reports **$0.89820784** ([partial billing](costs_partial_2130.json)); recent usage
may be unsettled, and this is not an invoice or the account's total spending.

## Local export and compatibility verification

The completed current checkpoint is preserved locally at
`artifacts/experiments/transfer_refresh_current_pilot_v7/step_000300.pt`
(SHA256 `7fb74cd1e8df6401b8b3cf935859951981d0f30260e113d6fed4176ac48db27c`).
Its unselected deployment export is
`artifacts/export/transfer_refresh_current_pilot_v7/candidate.onnx`
(SHA256 `9d6ca63cb376582a34675a519ac4cda9580de9cf6239ae6a91c4d5f4b72a199f`).

[CPU parity artifact](../../artifacts/export/transfer_refresh_current_pilot_v7/parity.json),
[reproduction script](../../scripts/transfer_v7_local_export.py): 40 synthetic
raw-camera frames per generic profile pass through actual warping, measured speed,
independent temporal queues and controllers. Formula/rover maximum trajectory
output differences are 8.44e-5/4.01e-5; steering differences below 3.46e-7 rad.
This establishes backend compatibility on those inputs, not transfer performance.

Full local test invocation: 111 passed, 2 skipped, 4 failed because macOS could
not create a graphics pixel format inside the sandbox. Rerunning exactly the four
rendering tests with local graphics access passed all four (3.72 seconds). This
is 115 passing tests across the two invocations, not a claimed all-green single
sandboxed run. No tests were weakened or newly skipped.

[Fresh rendered runtime parity](local_rendered_parity.json),
[script](../../scripts/transfer_v7_rendered_parity.py): two existing rover
validation configurations, 120 ticks each at 20 Hz on expert-driven states, pass
CPU Torch/ONNX comparison. Maximum trajectory-output differences are 7.44e-5 and
9.73e-5; steering differences are below 4.92e-7 rad. These observations are
physically rendered and warped by the simulator; driver histories evolve
independently. Exact simulator speed/desire and expert control are privileged.
This is numerical temporal parity, not autonomous completion or GPU validation.

## Supplemental local driving (partial)

The [first complete paired case](local_driving/val_v2_0.json) passes the evaluator's
mission criterion for every model, but is not clean driving: baseline has 6 cone
contacts/0 off-course events, v1 9/0, current v7 Torch 6/1 and current v7 ONNX 7/1.
Expert has zero of both. Current Torch/ONNX finish valid laps, yet both have worse
off-course counts than the baseline here. Numerical parity on shared observations
does not guarantee identical contact outcomes once each backend drives its own
trajectory. The remaining configurations are still running; no aggregate claim
is made from this first case.

The [same-window billing refresh](costs_partial_2138_full_hour.json) still totals
$0.89820784 for v7. A shorter-window query returned no rows because it omitted the
active hourly interval; [that snapshot](costs_partial_2138.json) is preserved and
**must not be interpreted as zero experiment cost**.

Timing limitation: [the evaluator](../../src/fsop/eval/transfer.py) calls prediction
then [advances a fixed simulation timestep](../../src/fsop/sim/env.py), independent
of elapsed inference time. On the first local case, Torch p95 is about 95 ms
and ONNX about 34 ms. Neither simulated completion nor configured 20 Hz input
rate proves a real system can meet its deadlines; measured wall-clock latency is
not injected as an extra control delay in this benchmark.

Both full-size cases are now complete ([paired partial summary](local_formula_partial.json)).
All models finish two valid laps, but learned policies are not clean: baseline
30 contacts/0 off-course events, v1 18/0, current Torch 27/1, current ONNX 28/1;
expert 0/0. On the second recovery case, current Torch's mean lateral error during
4–6 s is 1.899 m versus v1 1.271 m and baseline 2.198 m. The candidate has not
shown superiority to v1 on these full-size configurations. Rover results remain
pending; this is not the aggregate transfer result.

The [first rover case is complete](local_driving/val_v2_2.json). All learned
policies fail; expert finishes with zero contacts/off-course events. V1 achieves
41.5% progress, zero contacts and 1 off-course event. Current Torch and ONNX each
report zero progress, 1 contact and 1 off-course event. Their online trajectory
ADEs are 0.715/0.648 m versus v1 0.506 m; one-second tracking errors are
0.248/0.274 m versus v1 0.107 m. Mean visible cone counts fall below one for
all three adapted policies, while expert averages 9.24. These aggregates alone
do not establish whether visibility loss preceded prediction failure.
One rover case remains; the original paired Modal ablation is still incomplete.

Mission accounting detail: in val_v2_2 the baseline has three reported start-line
crossing intervals and `complete=true`, but `result=running`, zero net progress,
and `pass=false`. The mission did not finish. V1 and both candidate backends have
zero reported lap intervals. Preserve this distinction: raw referee completion
or line crossings alone cannot establish successful driving.

## Completed supplemental CPU comparison

All four configurations and five arms completed. [Full raw report](local_driving/report.json),
[validated aggregate](local_driving_summary.json), and [source identity](local_driving/identity.json)
preserve the exact configurations, seeds, checkpoints, and evaluator. Earlier partial
results above are superseded by this complete comparison.

| Model | All mission passes | Rover passes | Rover contacts | Rover off-course | Mean rover progress |
|---|---:|---:|---:|---:|---:|
| Original baseline | 3/4 | 1/2 | 9 | 3 | 50.0% |
| v1 calibrated model | 3/4 | 1/2 | 0 | 1 | 70.6% |
| v7 current refresh, Torch | 2/4 | 0/2 | 1 | 2 | 1.9% |
| v7 current refresh, ONNX | 2/4 | 0/2 | 1 | 2 | 1.8% |
| Privileged expert | 4/4 | 2/2 | 0 | 0 | 100% |

V1 completes val_v2_3 cleanly and reaches 41.5% progress on val_v2_2 before
failing. This is useful simulator behavior but not reliable rover transfer. The
current v7 candidate regresses and must not replace v1 or the original default.
Its lower aggregate online ADE does not imply better driving; each policy's ADE
is measured on its own visited states. Both candidate backends fail both rover
cases despite good numerical parity on shared observations.

This local CPU validation uses simulator speed, route desire, and mission launch/stop
assistance. The expert additionally uses privileged geometry. It is not real-world
validation, the missing paired Modal collector ablation, or an independent final
test. The v3 final configurations remain untouched. Required Modal work remains
blocked by the workspace spending limit; the latest requested retry also launched
no job (see [interruption record](compute_interruption.json)).
