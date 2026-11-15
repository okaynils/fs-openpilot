# Transfer v3: diagnose low-speed prediction/follower interaction

V2 is preserved as a completed negative experiment. Its final suite is disclosed and
will not select this iteration. The active goal remains measured transfer improvement.

Hypotheses, ranked before probing:
1. A short time-based target becomes spatially too close on a rover and amplifies
   lateral prediction errors. Replacing only lateral target selection with a
   vehicle-scaled spatial lookahead should reduce drift with the checkpoint fixed.
2. Temporal prediction inconsistency dominates. Spatial lookahead will not resolve
   it, and steering/path disagreement will remain time-varying on matched observations.
3. Frozen visual features lack low-viewpoint geometry. Neither controller change
   resolves consistent path bias despite visible cones; perception diagnostics would
   be needed before any further training.

First feedback loop: 12-second paired driving of the v2-selected checkpoint on the
existing low-forward validation configuration (98011/5101), keeping speed control,
launch assistance, camera, seed and weights fixed. Record drift, contacts, commands,
and expert controls. A tail lateral deviation above 1 m reproduces the known failure.
Full closed-loop validation remains necessary if the short probe improves.

One spatial follower candidate: rear-axle radial target distance max(2*wheelbase,
0.8*current_speed), interpolated on the predicted first-four-second path. Existing
longitudinal control is unchanged. This is a single physics-motivated probe, not a
lookahead grid search. Keep it outside production until evidence supports it.

A new complete-configuration final metadata suite is frozen before probing. No final
images or driving outcomes may be opened until the controller/checkpoint choice is
frozen on validation and default retention is checked. Any further training requires
a new bounded evidence-driven plan, new run names and preserved provenance.

## Evidence-driven training extension

The spatial probe reproduced tail drift (1.482 m) and did not improve it (1.490 m).
It is rejected; no production follower change is planned. The same-follower plan
probe finds initial steering −0.036 versus expert-trajectory −0.197 rad, from about
0.096 m lateral and 0.104 m longitudinal error at 0.35 s. Later, predicted steering
has the wrong sign for recovery. Every probed expert target is feasible.

Refined hypothesis: a metric-coordinate NLL averaged over horizons and dimensions
underweights small but control-critical near-term errors on low-speed vehicles.
Test exactly one auxiliary Smooth-L1 loss on predicted versus target pure-pursuit
steering, normalized by each vehicle's steering limit, with weight 1 and beta 0.1.
The loss reproduces the existing follower's time lookahead, CoG/rear-axle transform,
minimum denominator and steering clamp. No deployed controller or input changes.
Wheelbase/rear-axle distance/steering limit are training-side supervision metadata;
measured speed already exists in each sample and in the deployment follower.

Use the v2 expert-only dataset and frozen adapter setup, same seed and learning rate.
First run a 300-step pilot (distinct run) and the same 12-second validation probe.
Only if tail drift improves by at least 20% without contacts, run one 1,500-step fit
from the original baseline and compare fixed 500/1500 snapshots on full validation.
No parameter sweep. Keep final v3 observations sealed until selection/default checks.

## Independent collector diagnostic

Initial hypothesis: the recorded 94.5% rover expert fraction is explained by immediate
steering-disagreement interventions. Preserve that default and expose the intervention configuration in each
recording. A single paired 24-second diagnostic uses existing training configuration
`train_v2_03`, track 880004/scene 7104 and the unchanged v1 collector checkpoint.
Compare the legacy rule against disabling only steering-disagreement takeover, retaining
1.25 m lateral-trigger and 1.5 s hold. Require expert control below 75%, at least 95%
feasible labels and fewer than three contacts before considering broader collection.
These diagnostic recordings are not added to any training or validation dataset.
No new fit/data generation is authorized by this diagnostic alone; first finish the
current validation and document any subsequent bounded comparison.

The paired collector diagnostic falsified the steering-trigger-only explanation:
both modes were 100% expert-controlled. Recorded centerline lateral offset was
1.241–1.424 m during valid expert racing, above the 1.25 m trigger in 94.2% of frames.
The trigger was measuring a deliberate racing-line offset as an error.

A regression test now reproduces that actual seeded mission. Preserve centerline
behavior as the default; add `reference_path` error = centerline lateral offset minus
the optimized reference path's signed offset at the same route progress. A distinct
versioned collection profile changes only that reference relative to the failed
lateral-only variant. Reuse the prior diagnostic as the control, run one new episode,
and retain the same expert-fraction/feasibility/contact gates. No production inference
or current validation behavior changes.

## Bounded reference-path recovery comparison

The corrected diagnostic passes all collection gates: 9.6% expert control, 120/120
feasible labels, no contacts. Once the current full driving validation is complete,
replace exactly the 24 v2 on-policy training episodes using the corrected reference
profile. Preserve their profiles, tracks, scenes, duration, collector checkpoint and
random-start setting. Copy the 114 replay/expert/validation episodes byte-for-byte;
feature-cache reuse is permitted only for these unchanged episodes, subject to the
existing metadata and vision-checkpoint hash checks. Record a new dataset manifest
and source archive. Never modify `transfer_train_v2`.

Audit and visually inspect the new observations before training. Require at least
95% feasible labels overall, lower rover expert-control fraction than the old 94.5%,
continuous camera motion, zero trajectory-origin error and matching contact logs.
If these fail, diagnose collection before spending training compute.

Use the exact `transfer_recovery_v2` training recipe (including w_steering=0), changing
only dataset and run name. This isolates the intervention-policy change from the
failed control-loss arm. First run 300 steps and require finite losses and feasible
short-probe behavior (no contacts, mean speed at least 1 m/s, tail centerline deviation
no worse than the v2-selected reference). The short probe is a triage test only:
centerline deviation is not racing-line tracking error and cannot establish success.
If it passes, run one fresh 1,500-step fit with fixed 500/1500 snapshots and full paired
validation. No weight or threshold sweep. A superior validation candidate must pass
default retention before the frozen v3 final suite is opened.

### Conditional safeguard diagnostic

The broad replacement collection is exposing contacts and infeasible labels on fast
Formula configurations, plus rover `train_v2_13`. If its completed aggregate audit
fails the 95% feasibility gate, do not train on it. Test one bounded correction:
restore the original 0.20 rad steering-disagreement safeguard, retaining only the
reference-path lateral fix. This changes one intervention field relative to the
original v2 rule. Probe exactly `train_v2_00`, `train_v2_03`, `train_v2_13` with the
same original seeds, duration and checkpoint. These were chosen from collection
failures and the original causal reproduction, not from validation outcomes.
Require >=95% feasible labels, <75% mean rover expert control, and <3 total contacts.
Preserve the unguarded data as a failed diagnostic; these three probe episodes have
diagnostic split and are never inserted into training. Do not scale if this gate
fails. No training or final-test budget is consumed by this diagnostic.


The guarded diagnostic passes: 324/324 feasible labels, zero contacts, mean rover
expert control 49.6875% (29.375% and 70%). Scale this exact guarded profile to the
same fixed 24 training configurations under `transfer_guarded_train_v3`. Keep all
other data, loss, seeds and schedule as specified above. Apply the same >=95% label
feasibility and geometry gates; inspect rendered examples before the same 300-step
pilot and conditional 1,500-step fit. Preserve the rejected unguarded dataset.
Sampling order is now aligned during assembly, with the audit recording its hash.


## Independent reset-state speed ambiguity diagnostic

Code inspection confirms that `v_ego` reaches the follower but not the calibrated
policy. Test a single existing training profile (`train_v2_03`, track 880004/scene
7104), keeping pose, camera, time, yaw rate, augmentation RNG and empty temporal
history identical while setting speed to 0, 0.5, 1.0 and 1.5 m/s. Use the unchanged
v2-selected checkpoint; hash actual rendered policy images and all geometry/desire
inputs, and compare predicted versus feasible expert trajectories. Prediction:
identical policy inputs produce identical predictions despite speed-dependent labels.
This diagnoses reset/empty-history ambiguity only; it does not prove that visual
motion cannot infer speed later or authorize another training arm. Current full
validation and the frozen final-test boundary remain unchanged.

## Next feedback loop after guarded validation failed

Both guarded snapshots pass 2/4 versus baseline 3/4, so the final suite stays sealed.
Do not infer useful recovery from low ADE after the car has left the visible course.
Use the existing low-forward validation configuration `val_v2_2`, track 98011/scene
5101, guarded step 500 and a bounded 60-second trace to locate the first departure.
Record actual images/projections around departure, nominal/reference-path lateral
errors, visible cones (physical sensor plus body mask), speed, steering, route progress,
label feasibility, predicted/oracle plans and tracking error. Preserve source/seed/hash
identity. No new training is authorized by the reset-speed diagnostic alone.

Ranked hypotheses to separate:
1. Policy geometry is wrong while enough cones remain visible. An oracle expert plan
   through the same follower after t=12 s should recover from the identical prefix.
2. Vehicle tracking rather than prediction causes departure. Oracle-plan intervention
   will still fail; direct expert control would be a subsequent, conditional probe.
3. Visibility is lost before useful recovery guidance is available. Images and visible
   cone counts should expose this transition; aggregate end-of-run ADE hides it.

Compare policy-only versus a single t=12 s oracle-plan intervention with the same
production follower, physics and disclosed launch/stop assistance. Verify the shared
prefix matches before interpreting the intervention. Keep this diagnostic separate
from deployment and final tests. Explicit measured-speed conditioning remains a
possible follow-up because identical reset inputs demonstrably require different
labels, but it must be tested against a capacity-matched constant-speed-input arm
if implemented; it is not yet a demonstrated cure for the observed departures.
