# Transfer v2: recovery and closed-loop selection

Starting snapshot: `9a8b496`, pushed on 2026-11-14. Prior experiment and checkpoints
remain immutable. The prior final configurations are now disclosed regression
cases, not independent final tests for this iteration.

The red feedback signal is the v1 paired driving suite: selected candidate 0/12
passes, baseline 7/12, expert 12/12. Simulation takes minutes per case; an offline
controller-error probe on fixed validation observations provides a faster diagnostic,
but cannot replace driving validation.

Ranked hypotheses and discriminating experiments:

1. ADE hides harmful near-term controls: compare steering/acceleration agreement,
   wrong-turn frequency and trajectory-origin offsets on identical validation frames.
2. Recovery-state coverage is inadequate: compare expert-only data against a matched
   dataset augmented with expert labels on model-visited states. Keep histories
   physically coherent; disclose collector interventions and exclude infeasible
   supervision without removing frames from temporal history.
3. Random configuration draws miss important bounds: deliberately cover installation
   extremes with distinct continuous vehicle/configuration combinations, rather than
   increasing the number of frames from the same installations.
4. Current-feature conditioning is insufficient: compare identical geometry inputs
   applied to current versus all non-padding history features. Compare frozen versus
   cautiously trainable policy weights while retaining the reference behavior test.
   Vision fine-tuning is conditional on evidence and requires valid feature histories;
   do not silently mix changed vision features with stale history caches.

Use a bounded pilot before full generation/training. Keep a maximum of three GPU
collection/evaluation workers. Save named validation snapshots and use a separate
closed-loop validation suite for selection. Select by valid completions first,
then off-course behavior and cone contacts; also report progress and speed so that
stopping or leaving the track cannot appear to be a collision improvement. Default
benchmark retention remains a promotion requirement (3/3 trackdrive, at most three
contacts, ADE no worse than 10% above the strict v1 baseline).

Freeze a new complete-configuration final suite before candidate training and keep
its data/evaluation untouched until selection. Final comparisons must include the
original baseline and the v1 candidate on the same tracks/seeds/configurations.
No final-test-driven checkpoint selection or model changes. Simulator results do
not establish real-vehicle readiness; hidden track or expert information is allowed
only in data collection/diagnostics and must be reported.

## Predeclared bounded comparison

The 300-step sequence-adapter pilot uses four training installations, four validation
installations and six legacy replay episodes. Its purpose is implementation/cost
validation, not model selection. Inspect its actual images and numerical behavior
before scaling. Pilot data are immutable (`transfer_v2_pilot`); the complete event-log
fix is applied to subsequent generation. Audit confirms the pilot's one contact was
already recorded, so its targets are not affected by that recording defect.

Full data: 24 stratified training installations × two collectors (expert, v1 candidate
with expert takeover), four expert validation installations, and the 86 preserved v1
recordings. Each new sequence lasts at most 24 seconds. Old validation episodes keep
their validation split. The v1 final configurations are not training data.

Three matched training arms initialize the original selected baseline, retain its
vision and policy weights, and train the same 13-input anchored adapter:

- `transfer_expert_v2`: current-feature conditioning, exclude model-visited episodes.
- `transfer_recovery_v2`: current-feature conditioning, include model-visited episodes.
- `transfer_sequence_v2`: sequence-wide conditioning, include model-visited episodes.

Use seed 184191, 1,500 steps, learning rate 0.0003, batch 64, 100-step warmup.
Evaluate fixed snapshots at steps 500 and 1,500 on all four closed-loop validation
configurations, paired with the original baseline and v1 candidate. No grid search.
Rank by referee passes descending, off-course events ascending, contacts ascending,
then progress descending. Report completion/clearance/stability as well; referee
passes permit contacts and are not a safety certificate. Select before new final tests.

If none improves validation driving, allow one further arm with sequence conditioning
and a cautiously trainable policy (policy learning-rate multiplier 0.05), retaining
frozen vision so all cached histories remain valid. Run only this one bounded
follow-up at the same fixed steps. Default retention is tested before promotion;
if it fails, do not replace the default. Final testing is for reporting, not another
round of checkpoint selection.

V2 driving time limits are fixed before the full fits: 100 simulated seconds for
Formula configurations and 180 for rovers, identically for every compared policy.
The rover bound accommodates the declared 1 m/s end of the speed envelope on an
80–120 m course plus launch/stop. Keeping the old 100-second limit would confound
slow-platform feasibility with policy failure. V1 regression runs retain 100 seconds.

Before opening final observations, the exported-runtime subset is fixed to final
configuration indices 0 (Formula) and 2 (rover), first track/scene seed. The generic
single-seed evaluator applies recovery to its second configuration; both runtime
and training-model runs use those identical conditions. Export parity additionally
uses 120 runtime ticks per recorded Formula/rover validation sequence, filling the
entire temporal history. These are checks of the frozen selected candidate, never
new selection criteria applied to final-test results.
