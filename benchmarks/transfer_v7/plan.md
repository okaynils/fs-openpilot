# Second-round correction collection

The v3 correction dataset used v1 for every on-policy episode. All subsequent v5
fits reused those observations. The v5 speed-250 candidate fails both rover
validation configurations. This motivates testing a genuine second round of
model-visited corrections before changing architecture or losses again.

Pilot: four existing training configurations (indices 0, 3, 4, 13), two collectors
(v1 and speed-250), identical paired track/scene seeds, 40 seconds from the mission
start. This covers two full-size placements and two low rover placements. Existing
24-second randomly started data remains unchanged. New tracks use seeds 882101+i;
no validation/final profiles or tracks enter collection. Settings and checkpoint
hashes are recorded in a new manifest. Maximum three existing collection workers.

Inspect both arms' rendered observations, label validity, intervention fraction,
contact counts and state trajectories before fitting. Quality gate: >=95% valid
labels in each arm; no more than two contacts per episode; current rover mean
expert-control fraction below 75%. Failure means investigate collection quality,
not train on invalid labels or silently relax the gate. Frame zero will be included
in any subsequent fit, and valid-label masks retain temporal history.

If quality passes, retain old expert/replay data and compare an equal-size old-
collector refresh against current-collector refresh, with identical speed-250
initialization, sample count, optimizer and 300-step budget. Only collector identity
changes. Use complete existing validation configurations for checkpoint selection;
short departure probes alone cannot authorize scaling. The broader final suite
remains sealed. Exact training configs will be recorded before launch. This pilot
does not claim sufficient coverage for the entire support range.

```sh
modal run scripts/modal_app.py::transfer_v7_refresh
```

## Gate failure follow-up

Before the last rover pair completed, train_v2_00 had 3/5 contacts and 62/71,
64/71 valid labels in previous/current arms. This already fails the predeclared
gate; no training is launched. Run expert-only controls for all four exact
configurations/tracks/scenes. If expert-only succeeds, the collection intervention
is insufficient; if it also fails, inspect track/profile feasibility and expert
labels before changing collection thresholds. This control changes only the
acting policy, and does not alter the failed dataset.

## Single-variable guard probe

Expert-only train_v2_00 returned 70/70 valid labels and zero contacts. The previous
collector permitted steering disagreement 0.162 rad at 8.69 m/s (current arm,
t=4.0), below its fixed 0.20-rad trigger; invalid labels appeared at 4.4 s and a
contact followed at 4.5 s. Add an optional trigger for
`v² / wheelbase * abs(tan(steer) - tan(expert_steer)) > 2 m/s²`.
This is a kinematic discrepancy proxy, not measured lateral acceleration or a tire
model. Retain all previous triggers/hysteresis. It is disabled by default and
uses privileged expert controls only during dataset collection.

Replay only the failed pair with this one change; preserve tracks/seeds/profiles,
checkpoints and run duration. Require >=95% valid labels and <=2 contacts in each
arm. This probe alone does not authorize training: the complete collection gate
and visual review must still pass. No threshold sweep is planned.

## Guarded full-pilot check

Both train_v2_00 guard replays passed at 70/70 valid labels and zero contacts,
but expert-control fractions were 86.3% and 97.5%. Preserve these episodes exactly
and generate only the remaining six paired episodes with the same guard. Check
the original full-pilot thresholds, particularly mean current-rover expert
control below 75%, before staging training. No additional guard tuning is planned.

For a subsequent paired fit, use the same valid frame indices available in both
arms of each configuration (intersection of valid masks over their shared time
range). Keep every image/history frame, masking only surplus labels. This ensures
equal supervised sample counts without pretending unequal episode lengths match.
Record original and derived mask hashes; keep source episodes immutable. Retain
identical existing replay/expert/correction data in both arms. This measures the
collector change under equal training budgets, not a comparison against zero
additional optimization.

Training configurations are now fixed as
`configs/train/transfer_refresh_{previous,current}_pilot_v7.yaml`: 300 steps,
batch 64, seed 184197, learning rate 3e-4, 100 warmup steps, frozen vision/policy,
current-frame geometry and measured-speed adapters, original losses, initial
frames included in both arms, initialization from v5 speed-250. Each arm retains
all prior v3 data and adds its four matched correction episodes. This is a small
pilot, not a claim of broad coverage. Select using the same four complete v2
validation configurations. Compare both arms, unchanged speed-250, baseline, v1
and expert under identical current code. Require improvement over unchanged
speed-250 and at least parity with baseline/v1 by the existing validation ranking
before any scale-up/default/final evaluation. The isolated collector comparison
also requires current-refresh to outrank previous-refresh; ties are inconclusive.

## Interrupted control arm

The current arm completed 300 steps and wrote timing/validation artifacts. The
previous arm stopped reporting after step 175, has no completion file or fixed
step-300 checkpoint, and Modal lists no live workers. No explicit failure cause
was available in system logs. The stale orchestrator was stopped before retry.
Preserve the interrupted run and checkpoint under its original name.

The Trainer restores optimizer/RNG state, but does not save the active shuffled
sampler permutation/cursor. A mid-epoch resume would therefore change the paired
sample stream. Use one distinct technical retry of the previous arm from the
same initialization, seed and configuration (`transfer_refresh_previous_pilot_v7_retry1`),
with only the run name changed. Do not repeat the completed current arm. This is
not a hyperparameter trial; the fixed evaluator now references the retry's step
300. A second interruption requires investigation, not automatic further retries.

## Supplemental local driving comparison during the spending-limit block

With checkpoint/export preserved and fresh rendered CPU parity established, run
the existing four validation configurations locally with explicit CPU Torch
selection. Compare baseline, v1, current v7 step300, its ONNX runtime, and expert
on identical tracks/seeds, recovery flags and 100/180-second limits. Preserve
source/weight hashes and write each complete case separately. No training or final
configuration access. Hardware/rendering differ from Modal, so historical Modal
scores are not the control; use within-run comparisons only. The missing
previous-refresh step300 still prevents the planned collector ablation, and local
results do not fulfill the requested GPU evaluation. No default promotion follows
from this supplemental run alone.
