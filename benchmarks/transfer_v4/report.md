# Transfer v4 — current-weight history and late vision adaptation

The bounded pilots failed their driving gate. No new candidate is promoted and the v3 final
suite remains sealed. [Predeclared plan](plan.md).

The [paired v3 departure experiment](../transfer_v3/departure_report.json) identifies
prediction error before visibility loss: an exactly matched 12-second prefix is
followed by failure under the learned plan, versus a clean completion when only the
plan is replaced with a privileged expert plan through the same controller.
The reset-state speed ambiguity is a separate limitation, not established as the
cause of this steady-speed departure.

The existing stage-2 image path uses hidden history cached from initialization even
while the vision weights change. The new opt-in `history_mode: fresh` encodes past
recorded frame pairs using current vision weights in FP32, detaches historical
features, retains padding and never reads a feature cache. Current-frame gradients
remain enabled. The prepared policy gathers current plus eight preceding features;
a structural graph guard and numerical test reject unsupported wider-history graphs.
No inference input or model export contract changes.

[History implementation](../../src/fsop/model/history.py),
[dataset sampling](../../src/fsop/data/sim_dataset.py),
[training integration](../../src/fsop/train/trainer.py),
[tests](../../tests/test_fresh_history.py). Fifteen focused local tests pass, including
exact actual-runtime queue parity, batching tolerance, current-weight behavior,
invalid-label temporal continuity, cache avoidance, graph compatibility and resume
rejection. All 104 Modal regression tests passed ([JUnit](junit.xml)). Both 100-step pilots
completed with finite losses and intended gradients: zero vision gradients in the frozen
arm, nonzero gradients across all 15,715,392 trainable vision parameters in the last-stage
arm. Fit times were 23.52/26.98 seconds and peak allocated GPU memory 0.525/0.781 GiB.
Bounded validation ADE changed from 1.752 m to 1.885/1.741 m respectively; these are
320-frame diagnostics, not full-dataset or closed-loop promotion results. Fixed short
driving probes failed: frozen/last-stage arms had tail mean absolute centerline
offset 7.997/1.915 m, contacts 1/0 and off-course events 1/1. Mean speeds
were 1.769/1.675 m/s, so neither result is a stationary-policy artifact. These
12-second probes are triage, not full completion tests. The conditional 1,000-step
fits have **not** been launched. The unchanged initialization had 1.189 m tail offset, zero contacts and zero
off-course events on the identical probe, confirming degradation after training.
Its mean speed was 1.398 m/s; the changed speed profiles are part of learned-policy
behavior, so the offset comparison alone does not isolate lateral prediction quality.


| Fixed 12-second validation probe | Tail mean absolute centerline offset (m) | Contacts | Off-course events |
|---|---:|---:|---:|
| [Unchanged v1 initialization](init_probe.json) | 1.189 | 0 | 0 |
| [Frozen vision, step 100](frozen_pilot_probe.json) | 7.997 | 1 | 1 |
| [Last-stage vision, step 100](last_pilot_probe.json) | 1.915 | 0 | 1 |

These matched runs use validation configuration `val_v2_2`, track seed 98011 and
scene seed 5101, with the existing disclosed mission launch/stop assistance. The
expert plan is computed only for diagnostic error reporting; neither trained arm
uses it to steer. The 12-second horizon deliberately cannot establish completion.
The planned full fits were withheld, so this does **not** show that vision adaptation
can never help. It shows this particular update recipe is unsuitable for scaling.
Offline ADE alone would have incorrectly favored the vision-tuned pilot.

The corrected current-weight training history is independently validated and remains
available. There is no evidence to promote either trained checkpoint or to claim
better cross-vehicle driving from this change. The v3 final suite remains sealed.

Source and dataset identities are recorded in [frozen provenance](frozen_pilot_provenance.json)
and [last-stage provenance](last_pilot_provenance.json). Checkpoints remain on Modal at
`runs/transfer_vision_{frozen,last}_pilot_v4/ckpt/step_000100.pt`; source archives were
also retrieved to `artifacts/experiments/` under each run name. No checkpoint was
overwritten, exported as a release, or installed as the default.

[Modal run ledger](modal_runs.json) and [timings](last_pilot_timing.json) retain compute
provenance. The initial [billing query](modal_metered_costs.json) returned no settled
hourly records. The later [17:07 UTC query](costs_reproducible_1707.json) reports
**$0.36200627** in app-scoped metered usage, not an invoice.
The two fits consumed 50.49 seconds combined, excluding staging, startup and driving
checks. All jobs in this phase completed; no full-fit jobs were launched.


## Precision diagnostic after the failed driving gate

[Paired results](precision_probe.json), [source](../../scripts/transfer_v4_precision_probe.py):
64 identical validation samples, seed 184192, current-weight FP32 history in both
arms. Switching current-frame/policy computation between FP32 and BF16 changed
steering by 0.000907/0.000695/0.000846 rad on average for initialization/frozen/last.
Maximum changes were 0.00629/0.00349/0.00668 rad. Both trained pilots worsened mean
steering error under both precisions: FP32 errors 0.01844/0.02017/0.02057 rad.
Their mean NLL nevertheless improved from -0.08485 to -0.09934/-0.16799.
This small recorded-input comparison does not prove closed-loop causality, but
provides no support for precision being the dominant failure mechanism. It supports
retaining driving-based selection rather than treating improved NLL as success.
The next bounded experiment is the [capacity-matched speed comparison](../transfer_v5/plan.md).
