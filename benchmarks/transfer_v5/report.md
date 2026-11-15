# V5 measured-speed conditioning experiment

**Do not replace the default model.** The best new checkpoint passes 2/4 validation
configurations, versus 3/4 for both baseline and v1. All new candidates fail both
rover cases. [Predeclared comparison](plan.md), [paired results](validation.json),
[fixed ranking](selection.json).
The unchanged baseline/v1 references and sealed v3 final suite remain preserved.

The only new measured quantity is current forward speed in m/s. A separate,
zero-initialized adapter receives current visual features, the existing calibrated
geometry vector and speed normalized by 15 m/s. The capacity-matched control feeds
zero into the same scalar slot. Neither adapter changes the trajectory origin,
image preprocessing or visual history. Unlike the geometry-only anchored adapter,
the state adapter can change predictions at default geometry after training; default
retention must therefore be measured, not assumed.

Training consumes the already-recorded speed alongside cached visual features.
The new checkpoint contract is validated on load; old checkpoints default to no
state input. Runtime forwards its existing `v_ego` measurement to speed-conditioned
prediction and rejects missing/nonfinite speed before changing temporal queues.
[Model](../../src/fsop/model/build.py), [training](../../src/fsop/train/trainer.py),
[export](../../src/fsop/model/export.py), [runtime](../../src/fsop/runtime.py),
[compatibility tests](../../tests/test_speed_conditioning.py).

All [108 regression tests](junit.xml) passed. An actual [Torch/ONNX contract probe](contract_probe.json)
used nonzero synthetic adapter weights, eight sequential frames and varying speeds.
Maximum plan differences were 0.0000172 m (speed) and 0.0000114 m (constant), with
steering differences below 0.0001 rad. This checks input plumbing and temporal
numerical parity, not learned usefulness. Both 300-step pilots completed after the [read-only cache preflight](cache_check.json)
verified exact equality of 270 vision tensors and all 138 episode cache identities.
The matched full fits completed after both pilots passed the short driving gate.
All 28 complete-configuration validation runs finished with fresh baseline/v1 references.
Final-test observations remain untouched.


## Pilot results

| Arm | Validation ADE (m), step 300 | Tail absolute lateral offset (m) | Contacts / off-course | Fit seconds |
|---|---:|---:|---:|---:|
| Unchanged v1 initialization | 1.850 | 1.189 | 0 / 0 | — |
| Constant input | 2.073 | 1.033 | 0 / 0 | 35.26 |
| Measured speed | 1.994 | 1.128 | 0 / 0 | 35.41 |

[Constant driving](constant_pilot_probe.json), [speed driving](speed_pilot_probe.json),
[initialization driving](../transfer_v4/init_probe.json). These are matched 12-second
validation probes, not full-course completion results. Speeds average 1.782/1.771 m/s
for constant/speed versus initialization 1.398 m/s. The unchanged initialization
was measured under the immediately preceding implementation; 108 regression tests
cover legacy behavior, and full validation reran baseline/v1 under the same evaluator code as all candidates.

Both pilots met the no-gross-degradation driving gate, despite worse full-validation
ADE than initialization. The constant control does better on this short driving
metric, so the evidence does not yet justify measured-speed conditioning. Peak
allocated GPU memory was 0.610 GiB in each arm. [Training logs](speed_pilot_val.jsonl),
[timing](speed_pilot_timing.json), [provenance](speed_pilot_provenance.json).

## Does the model use speed?

The [controlled reset-state comparison](state_information.json) uses one existing
training configuration, identical pixels, geometry and empty history, with four
feasible speeds 0/0.5/1/1.5 m/s. Targets at 0.35 seconds are
0.023/0.195/0.363/0.528 m forward. Initialization predicts 0.256 m for all four;
the constant control predicts 0.431 m for all four. The measured-speed arm predicts
0.420/0.428/0.435/0.442 m: it responds in the correct direction but much too weakly.
Mean absolute error is 0.168/0.202/0.197 m for initialization/constant/speed.
Thus the new input reaches prediction, but this pilot **does not resolve the ambiguity**.
This is a single controlled training-state diagnostic, not unseen transfer evidence.

## Trained export/runtime verification

[Physical-camera parity](runtime_parity.json) covers each trained pilot on two rover
validation configurations, including recovery, for 120 fresh 20 Hz frames each.
Both inference implementations see identical expert-driven observations. The expert
chooses the states solely for this parity test; this is not autonomous completion.
Maximum plan errors are 0.0000458 m (constant) and 0.0000362 m (speed); steering
errors remain below 0.00000024 rad. Checkpoint/export hashes are recorded per row.
Exports are preserved remotely at `runs/transfer_state_runtime_v5/{constant,speed}.onnx`
and retrieved under `artifacts/export/transfer_state_pilot_v5/`. They are research
artifacts, not replacements for the default model.

The full comparison used only fixed step-250 and step-1000 snapshots from one fresh
1000-step fit per arm. No default-retention or final evaluation was launched because
complete-configuration validation failed the promotion gate.


## Full fits and complete-configuration driving results

The constant/speed 1000-step fits completed in 76.58/77.63 seconds, with equal
0.610 GiB peak allocated GPU memory. Validation ADE at step 1000 is 2.074/1.799 m,
versus initialization 1.850 m. The measured-speed input improves this offline metric
relative to both the matched control and initialization. This offline improvement did **not** translate into better transfer driving.

[Constant validation log](constant_full_val.jsonl), [speed validation log](speed_full_val.jsonl),
[constant provenance](constant_full_provenance.json), [speed provenance](speed_full_provenance.json).
The fixed validation job evaluated four complete configurations against baseline,
v1, both step-250 snapshots, both step-1000 snapshots and expert controls (28 runs).
The same tracks, seeds, recovery flags, time limits and mission assistance apply
to all models. No validation results have been substituted from earlier code versions.

Combined training fit time for the four v5 fits is 224.88 seconds; this excludes
staging, startup, tests, exports and driving evaluation. The refreshed
[billing query](costs_reproducible_2042.json) reports **$2.29104657** in metered
usage, including the previously unsettled 17:00 interval. The earlier
[partial snapshot](costs_reproducible_1707.json) is preserved. These are metered
hourly records, not an invoice.
The shared transfer goal remains active and unproven. No default replacement,
additional commit/push, or final-test opening has occurred.


## Read-only training coverage audit

[Actual sampled states](state_coverage.json), [audit source](../../scripts/transfer_v5_coverage_audit.py).
The 9683 training samples contain no empty-history observations because the historical
loader starts at frame 1. Small-wheelbase episodes contribute 5072 samples, of which
32 are at or below 0.1 m/s and 39 are in (0.1, 0.5] m/s: only 1.40% at low speed.
There are 56 valid omitted initial rover observations, with recorded initial speeds
ranging from zero to 2.966 m/s. This is a coverage limitation, not a demonstrated cause
of steady driving failures. The audit did not modify the dataset or read final images.

The training configuration now supports opt-in `include_initial_frame: true`. It uses
the actual recorded first frame, measured speed, feasible-label mask and zero history;
it repeats that image in the initial frame pair exactly as runtime does. The flag is
recorded in run configuration and changing it on resume is rejected. Existing v5 fits
used the historical default `false`, and their data/checkpoints are unchanged.
[Tests](../../tests/test_initial_frame_training.py) cover cached and fresh inputs,
trainer forwarding and compatibility; 21 targeted local tests pass including resume
and temporal-history tests. No new fit with this option has been launched.


## Why progress is not a valid completion

The [two completed Formula Student cases](formula_partial.json) give valid-lap
passes 2/2 for baseline/v1, 2/2 for constant-250 and speed-250, and 1/2 for
constant-1000 and speed-1000. Their contact totals are 30/19/34/31/23/29
respectively. That preserved partial snapshot was not used for selection; the complete four-case
ranking above determines the outcome.
The speed-1000 model reaches `completion_fraction: 1.0` on `val_v2_0` but has no
valid lap (`complete: false`, `pass: false`). Thus route progress and a mission
`finished` flag must not be interpreted as valid completion. The case-level
referee records and fixed ranking use the valid-lap pass flag.


## Final validation ranking and failure interpretation

| Model / snapshot | Valid passes | Off-course events | Cone contacts | Mean route progress |
|---|---:|---:|---:|---:|
| Expert control | 4/4 | 0 | 0 | 1.000 |
| Previous v1 candidate | 3/4 | 1 | 19 | 0.853 |
| Original baseline | 3/4 | 1 | 33 | 0.750 |
| Speed, step 250 | 2/4 | 4 | 32 | 0.543 |
| Constant, step 250 | 2/4 | 5 | 35 | 0.504 |
| Constant, step 1000 | 1/4 | 3 | 24 | 0.295 |
| Speed, step 1000 | 1/4 | 7 | 33 | 0.540 |

Selection chooses speed step 250 among the new candidates, but it loses to both
references. Both fresh references reproduce their previous ranking exactly.
All four case files were checked against the aggregate report; no records were
mixed across source versions. [Case records](validation_cases/val_v2_2_98011_5101.json)
include exact configuration, checkpoint/source hashes and all clearance, prediction,
tracking, recovery and stability metrics.

For the selected new candidate, the two rover cases reach 0.000/0.172 route progress,
with RMS lateral deviation 28.00/61.66 m and only 0.81/1.19 mean visible cones.
One-second trajectory tracking errors are 0.145/0.176 m and steering-rate RMS is
0.040/0.042 rad/s. Small tracking error and smooth steering while driving away from
the course are not recovery success. These aggregates mix the initial departure
with long periods after visibility is lost and cannot, by themselves, isolate the
first cause. The earlier [paired oracle intervention](../transfer_v3/departure_report.json)
is evidence for prediction-driven departure in that earlier candidate, not proof
of every v5 failure. The expert completes both rover cases without contacts.

Every learned-policy driving run uses simulated speed, route desire and mission
progress for disclosed launch/stop assistance. The speed-conditioned arm additionally
feeds exact simulated longitudinal speed into prediction; sensor noise/delay robustness
is not established. Expert controls additionally use the full racing line, true state,
dynamics and cone map. No final-test configurations were rendered or evaluated.

The comparison supports a small advantage of measured speed over its matched
constant-input control, but neither shared model achieves useful transfer improvement
over baseline/v1. The reset diagnostic also shows that merely wiring the input does
not make the model learn the required dependence. The sampling coverage issue is
now documented and an opt-in correction is tested; it has not yet been trained or
shown to fix steady driving. The goal remains active and unmet.


[Shutdown audit](shutdown_audit.json) confirms all seven v5 apps stopped after completion.
No worker was left running and no additional training was launched.
