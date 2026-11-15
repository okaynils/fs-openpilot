# Transfer v3 investigation — in progress

The [v2 negative result](../transfer_v2/report.md) is preserved. No default or deployed
controller has changed. The previous goal turn was progress: it completed controlled
training/evaluation and narrowed the failure, but did not achieve transfer improvement.
[Predeclared v3 plan](plan.md); new final metadata is frozen in
[transfer_v3_test.yaml](../../configs/eval/transfer_v3_test.yaml) and remains unrendered.

## Diagnostics

The [spatial-lookahead probe](spatial_probe.json) reproduces low-forward rover drift
on validation seed 98011/5101: tail lateral deviation 1.482 m with the existing follower
and 1.490 m with the single spatial alternative. Both have no contacts over 12 seconds.
The alternative is rejected and remains only in the archived diagnostic script.
[Probe source](../../scripts/transfer_v3_probe.py).

The [same-follower plan probe](plan_probe.json) confirms substantial control-relevant
prediction error. During the first four seconds, predicted steering averages −0.036
rad versus −0.197 rad from the expert trajectory through the identical follower.
At 0.35 seconds ahead, the mean lateral error is only 0.096 m and longitudinal error
0.104 m, illustrating why low average metric error can still drive badly on a rover.
During seconds 8–12, predicted steering remains −0.022 rad while the target is +0.090
rad: recovery steering has the wrong sign. All sampled counterfactual expert plans
are feasible. [Recorded plans](plan_probe.npz), [source](../../scripts/transfer_v3_plan_probe.py).

## Bounded pilot

A new optional `w_steering` loss compares the actual pure-pursuit action derived from
predicted/target trajectories, normalized by the vehicle steering limit. The production
follower is unchanged. Training-side wheelbase/rear-axle/steering metadata and existing
sample speed are used to compute supervision; there are no additional deployed-model
inputs. Zero weight preserves the legacy objective. Resume rejects a changed weight.

Six local tests passed, including numerical parity with the deployed follower across
car/rover geometry, signs and saturation, finite gradients, and resume compatibility.
The predeclared 300-step pilot completed in 35.53 seconds on Modal. Scaling requires at least 20%
less tail drift without contacts on the fixed short validation probe; no claim of
improved driving is made from a training-loss change alone.


The [pilot gate](pilot_gate.json) passes: tail lateral deviation falls from 1.483 to
0.997 m (32.8%), with no contacts. Mean speed is 1.750 versus 1.761 m/s, so the
improvement is not explained by crawling. This is a short triage check, not a completed
lap or a clean causal ablation against an equal-step reference. The full matched
1,500-step fit completed; its only training-objective difference from expert-v2
is `w_steering=1`. [Pilot history](pilot_validation.jsonl), [timing](pilot_timing.json),
[driving trace](pilot_probe.json). The full regression suite passed **83 tests in
81.40 seconds on Modal**.


The full fit's fixed snapshots have validation ADE 2.076 m (step 500) and 2.238 m
(step 1,500). [History](full_validation.jsonl), [timing](full_timing.json),
[provenance](full_provenance.json). Full paired driving validation includes the
original baseline, v1 candidate, v2-selected checkpoint and both new snapshots.
The controller is the unchanged production pure-pursuit follower for every model.
The newly frozen v3 final suite remains untouched.


## Collection failure isolated and corrected

The [paired collection probe](collection_probe.json) falsified the hypothesis that
steering disagreement alone caused excessive expert intervention: disabling that
trigger still left **100% expert control**. The seeded expert racing line itself
lies 1.241–1.424 m from the centerline, exceeding the 1.25 m takeover threshold in
94.2% of recorded frames. The collector confused an intentional racing-line offset
with driving error. This explains why these nominal recovery examples resembled
expert demonstrations.

A versioned [reference-path intervention profile](../../configs/distributions/transfer_collection_reference_v3.yaml)
measures deviation from the optimized racing line instead. The [matched diagnostic](reference_collection_probe.json)
changes only that reference relative to the failed lateral-only variant: expert
control falls to **9.6%**, with **120/120 feasible labels and zero contacts**.
Mean pose displacement from the expert-controlled recording is 2.15 m. This is
evidence of useful policy state visitation, not evidence of improved trained driving.
Legacy collection settings remain supported and old datasets are unchanged.

The corrected collector uses privileged simulator racing-line offset, route progress
and expert supervision; these are collection inputs, not deployed model inputs.
[Source](../../src/fsop/sim/transfer_collection.py),
[six regression tests](../../tests/test_collection_intervention.py). The six new tests
plus two control-loss and four provenance tests pass locally. The earlier 83-test
Modal run predates this collector fix.

The [observation audit](reference_audit.json) confirms vehicle-origin labels, continuous
mount motion (maximum pose step 0.00538), and agreement between contact records and
the referee. The [rendered preview](reference_observations.jpg) was visually inspected:
low-camera cones are visible and trajectory overlays are consistent. No final-test
observations have been generated.


## Full control-loss validation: negative

[Paired results](validation.json), [fixed ranking](selection.json): baseline and v1
pass **3/4** configurations; v2-selected and both new control-loss snapshots pass
**2/4**. New step 500 has 38 contacts and two off-course events, versus v2-selected
37/two and original baseline 33/one. Both new snapshots time out on both rover
configurations. Step 500 increases mean progress to 0.636 from v2-selected 0.565,
but this does not meet the success criterion. The control-loss candidate is rejected
for promotion. No v3 final observations were opened.

The corrected-collection follow-up is predeclared in [the plan](plan.md). It changes
only the on-policy collection intervention settings relative to recovery-v2; the
original loss and fixed training schedule are retained. Its diagnostic already
passed the collection gate; the full new dataset must pass observation audits before
training. The 2026-11-14 16:51 UTC billing refresh reports v3 metered usage of **$3.75806030**;
see [metered costs](modal_metered_costs.json). This supersedes the earlier partial
$1.3477 snapshot, which is preserved separately. It is app-scoped usage, not an invoice.


## Broad collection quality gate: failed before training

The 24 corrected-reference, steering-unguarded episodes completed in 987.30 measured
worker-seconds: **2,266/2,413 feasible frames (93.91%)**, below the predeclared 95%
threshold. This dataset is preserved as `data/sim/transfer_reference_train_v3`; no
model was trained on it. Several faster Formula configurations and one rover reached
infeasible recovery states after steering-disagreement takeover was disabled.

The conditional three-case diagnostic restores the original 0.20 rad steering
safeguard while keeping the racing-line reference correction. Relative to v2 this
changes only the lateral reference. Its fixed cases/gates are recorded in the plan.
This is a collection diagnostic, not another model or hyperparameter search.


[Full collection audit](recovery_audit.json), [generation metrics](reference_generation.json),
[manifest](reference_manifest.json), [rendered observations](recovery_observations.jpg).
The 24-configuration montage was inspected: views vary coherently in height, translation,
rotation, quality and visible bodywork; there are no blank images. The rover takeover
fraction falls from 94.55% to 22.92%, but label feasibility fails. Geometry checks pass:
zero origin error, maximum camera-pose step below 0.007, and exact contact-log agreement.
The audit also aligns episode order with v2 and records pre/post index hashes.

The [guarded diagnostic](guarded_probe.json) passes all predeclared gates. Formula
`train_v2_00` returns to 84/84 feasible frames and zero contacts. Rover `train_v2_03`
has 120/120 feasible frames, zero contacts and 29.375% expert control; `train_v2_13`
has 120/120, zero contacts and 70% expert control. Mean rover takeover is 49.6875%,
compared with 100% on both original v2 recordings. The exact guarded rule is now
being collected on all 24 fixed training configurations in a distinct dataset;
training remains conditional on the full observation-quality audit.


Post-fix regression validation passed **89 tests in 81.45 seconds on Modal**
([JUnit results](collector_tests.xml)). Four additional local tests cover the strict
reference-reuse merger: collection-only changes are allowed, changed/new inference
code and disagreement in repeated expert outcomes are rejected. Existing paired
references may be reused only after complete configuration/seed/limit and source
hash checks; the unused collection module is the sole permitted source exception.
[Merger](../../scripts/transfer_v3_merge.py), [tests](../../tests/test_transfer_reference_reuse.py).


The full guarded dataset passes all [observation gates](guarded_audit.json):
**2,363/2,379 feasible frames (99.33%)**, zero origin error, continuous pose sequences
and consistent contact logs. Mean rover expert takeover is **47.95%**, down from
94.55% in v2. Collection took 819.68 measured worker-seconds. All 24 configurations
in the [rendered montage](guarded_observations.jpg) were visually inspected before
training. [Generation](guarded_generation.json), [manifest](guarded_manifest.json).

The original 114 replay/expert/validation episodes retain exact metadata bytes and
cached vision identities; only 24 on-policy episodes are replaced. Episode sampling
order matches v2. The guarded 300-step pilot has started with the unchanged v2
recovery recipe, original loss (`w_steering=0`), initialization, seed and learning
rate. No model improvement is claimed from the successful collection audit alone.


The guarded-data pilot [passes its driving gate](guarded_pilot_gate.json): tail
centerline deviation is **1.149 m**, versus reference **1.483 m**, with zero contacts
and mean speed **1.746 m/s**. All probe targets are feasible. The final four-second
window now has predicted/oracle steering −0.024/−0.014 rad, unlike the opposite-sign
recovery behavior in the original probe. This is still a short triage result, not a
completed course or proof of general transfer. [Probe](guarded_pilot_probe.json),
[training history](guarded_pilot_validation.jsonl), [timing](guarded_pilot_timing.json),
[provenance](guarded_pilot_provenance.json). The one predeclared full fit is now allowed.


## Reset-state speed ambiguity: measured

A [counterfactual training-configuration diagnostic](speed_ambiguity.json) holds actual
rendered pixels, geometry, desire, pose, augmentation RNG and empty temporal history
identical while changing measured speed. The v2-selected policy's predictions are
bitwise identical (maximum difference zero), but all four expert labels are feasible
and differ. At 0.35 s, predicted forward position is always **0.3007 m**; expert
positions at speeds 0, 0.5, 1.0 and 1.5 m/s are **0.0231, 0.1953, 0.3633 and 0.5276 m**.
Actual image hashes and the complete input values are recorded. This confirms a
structural ambiguity at reset: speed is available to the controller but absent from
the learned policy. [Source](../../scripts/transfer_v3_speed_ambiguity.py),
[plans](speed_ambiguity_plans.npz).

This finding does **not** establish that speed explains every closed-loop failure:
useful temporal visual motion can reveal speed later. It is not evidence that a new
input works before training and evaluating it. No speed input or new training arm
has been implemented from this diagnostic; current guarded full validation continues.


A [saved-prediction origin audit](predicted_origin_audit.json) also rules out origin
translation as the near-term steering-error explanation: predicted t=0 x is exactly
zero and maximum absolute y is below 8e-10 m in both original and guarded pilot
traces. No trajectory reanchoring change is warranted.


## Guarded full validation: improvement insufficient for promotion

The [paired report](guarded_paired_validation.json) passed all reference-reuse checks:
identical complete configurations, seeds, recovery settings, time limits and every
inference/simulation source hash; only the unused collection module differs.
Repeated expert outcomes agree. [Selection](guarded_selection.json).

| Model | Passes | Contacts | Off-course events | Mean progress |
|---|---:|---:|---:|---:|
| Original baseline | 3/4 | 33 | 1 | 0.750 |
| v1 candidate | 3/4 | 19 | 1 | 0.853 |
| v2 selected | 2/4 | 37 | 2 | 0.565 |
| v3 control loss, step 500 | 2/4 | 38 | 2 | 0.636 |
| v3 guarded data, step 500 | 2/4 | 35 | 2 | 0.767 |
| v3 guarded data, step 1500 | 2/4 | 26 | 2 | 0.660 |
| Expert | 4/4 | 0 | 0 | 1.000 |

The fixed ranking selects guarded step 1500 among new snapshots because contacts
break the tied pass/off-course counts. It still loses to both original baseline and
v1. Both guarded snapshots fail both rover courses. Step 500 improves rover progress
to 0.621/0.446 but still leaves the course. It is not a successful transfer model.
The full fit took **111.84 seconds**, with validation ADE 2.148 m at step 500 and
2.294 m at step 1500. [Training history](guarded_full_validation.jsonl),
[timing](guarded_full_timing.json), [provenance](guarded_full_provenance.json).

Failure metrics need care: the rover runs eventually travel far from the course
(RMS centerline deviation 62–149 m), averaging only 0.76–1.23 visible cones. Their
low aggregate online expert ADE is therefore not evidence of useful driving: it
mixes initial visible-track prediction with long periods after the course is lost.
One-second tracking errors of 0.12–0.67 m and low steering-rate RMS do not establish
recovery capability. The next useful diagnostic is the first departure and a paired
oracle-plan intervention, rather than another loss or threshold sweep. The measured
reset-state speed ambiguity remains a separate finding, not a proven explanation
for every departure.

**Do not replace the default model.** No new v3 export/default/final evaluation was
run because validation did not justify promotion. The previously validated v2 export
remains available as a research artifact. The frozen v3 final configurations remain
unrendered and untouched. The collector fix is useful and tested, but the shared-model
transfer goal remains active and unmet.


## First departure isolated by a paired intervention

The [60-second paired diagnostic](departure_report.json) has **zero prefix difference**
through 12 seconds. Policy-only commands −0.075 rad at 12 s while the feasible
expert plan through the identical follower commands +0.051 rad. Five cones remain
geometrically visible. It loses all cone visibility at 15.6 s, goes off course at
16.5 s and reaches 85.24 m reference-path error by 60 s. Replacing only the plan at
12 s completes a valid lap in **43.60 s**, with **zero contacts and off-course events**;
final reference-path error is 0.0066 m. Both arms use the same disclosed launch/stop
assistance. The intervention additionally uses privileged expert state/map/dynamics.

[Policy rendering](departure_policy.jpg), [intervention rendering](departure_oracle.jpg),
[source](../../scripts/transfer_v3_departure_probe.py). Both renderings were inspected.
This isolates this departure to prediction before visibility loss, not inability of
the follower to recover from the same state. Geometric visibility counts exclude
cone-on-cone occlusion; the images provide the complementary evidence.
The next bounded comparison is [late vision adaptation](../transfer_v4/plan.md), with
current-weight temporal history instead of stale initialization caches. The shared
transfer goal remains active; no default or final-test boundary changes.
