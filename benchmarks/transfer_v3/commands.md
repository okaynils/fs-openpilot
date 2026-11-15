# Transfer v3 reproduction

Preserve existing run directories. Restore each run's archived source for historical
reproduction; use distinct names for fresh repeats. All commands below refer to the
recorded experiment, not permission to overwrite its artifacts.

```sh
# Metadata only; refuses a changed frozen design.
python scripts/transfer_v3_design.py

# Two validation-only diagnostics; no production controller changes.
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v3_probe.py'
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v3_plan_probe.py'

# Bounded pilot and fixed short driving gate.
modal run scripts/modal_app.py::transfer_v3_pilot_train
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v3_plan_probe.py runs/transfer_control_pilot_v3/ckpt/step_000300.pt runs/transfer_control_pilot_probe_v3'
modal run scripts/modal_app.py::sh --cmd 'python -m pytest tests -q'

# Only after the pilot gate passed: one full matched fit, then paired validation.
modal run scripts/modal_app.py::transfer_v3_control_train
modal run scripts/modal_app.py::transfer_suite \
  --suite configs/eval/transfer_v2_val.yaml --name transfer_control_validation_v3 \
  --checkpoints runs/cuda_racing_v6/ckpt/best.pt,runs/transfer_minimal_v1/ckpt/best.pt,runs/transfer_selected_v2/selected.pt,runs/transfer_control_v3/ckpt/step_000500.pt,runs/transfer_control_v3/ckpt/step_001500.pt
```

The unchanged v2 validation profiles remain selection data. Both v1 and v2 final
results are disclosed regression evidence; neither selects v3. The v3 final suite
has metadata only until validation selection and default-retention gates permit it.
Training uses the preserved `transfer_train_v2` dataset and frozen-vision caches.
`w_steering=0` reproduces the legacy objective; nonzero weight is part of resume
compatibility. No extra model or deployment inputs are introduced.


Collection diagnostics (training configuration only; never added to validation):

```sh
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v3_collection_probe.py'
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v3_reference_collection_probe.py'
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v3_preview_probe.py'
```


After the corrected collector diagnostic passed and the control-loss full validation
failed, the bounded replacement collection was launched:

```sh
modal run scripts/modal_app.py::transfer_v3_reference_data
```

This preserves 114 old episodes and replaces only 24 on-policy training episodes in
`data/sim/transfer_reference_train_v3`. Cached features are reused only when both
unchanged metadata and vision-checkpoint hashes match. The following commands are
prepared, **not executed because the observation gate failed**:

```sh
modal run scripts/modal_app.py::transfer_v3_reference_pilot
modal run scripts/modal_app.py::transfer_v3_reference_train
```


The dataset failed the 95% feasible-label gate (93.91%). Audit and conditional
safeguard diagnostic:

```sh
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v3_recovery_audit.py && python scripts/transfer_v3_guarded_collection_probe.py'
```


After the guarded diagnostic passed all three gates:

```sh
modal run scripts/modal_app.py::transfer_v3_reference_data --guarded
modal run scripts/modal_app.py::sh --cmd 'python -m pytest tests -q --junitxml=runs/transfer_tests_guarded_v3/junit.xml'
```

Guarded training configurations/entrypoints are prepared but not yet run. Their
full observation audit, short driving gate and fixed full-validation comparisons
remain required before opening any v3 final observations.


Full guarded observations passed all gates and were visually inspected:

```sh
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v3_recovery_audit.py data/sim/transfer_guarded_train_v3'
modal run scripts/modal_app.py::transfer_v3_guarded_pilot
```


The guarded pilot passed its fixed driving gate:

```sh
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v3_plan_probe.py runs/transfer_guarded_pilot_v3/ckpt/step_000300.pt runs/transfer_guarded_pilot_probe_v3'
modal run scripts/modal_app.py::transfer_v3_guarded_train
```


Full guarded fit completed; fixed-snapshot driving validation and a separate
training-configuration reset-state diagnostic:

```sh
modal run scripts/modal_app.py::transfer_suite --suite configs/eval/transfer_v2_val.yaml --name transfer_guarded_validation_v3 --checkpoints runs/transfer_guarded_v3/ckpt/step_000500.pt,runs/transfer_guarded_v3/ckpt/step_001500.pt
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v3_speed_ambiguity.py'
```


The guarded validation was merged with preserved references only after strict checks:

```sh
python scripts/transfer_v3_merge.py benchmarks/transfer_v3/validation.json benchmarks/transfer_v3/validation_cases benchmarks/transfer_v3/guarded_validation.json benchmarks/transfer_v3/guarded_validation_cases benchmarks/transfer_v3/guarded_paired_validation.json
python scripts/transfer_v2_select.py benchmarks/transfer_v3/guarded_paired_validation.json benchmarks/transfer_v3/guarded_selection.json
```

Both new snapshots fail promotion. No v3 final-test observations were opened.


Reproducible metered-cost query (use a Python environment with authenticated Modal SDK;
choose a new output path for every refresh):

```sh
python scripts/transfer_costs.py --ledger benchmarks/transfer_v3/modal_runs.json --start 2026-11-13 --end 2026-11-14T16:51:00 --output benchmarks/transfer_v3/costs_reproducible_1651.json
```

Executed with Modal SDK 1.6.2: $3.75806030 app-scoped metered usage. Hourly records
may exclude unsettled recent usage; this is not an invoice. V2/v3/v4/v5 app ledgers
have no overlapping app IDs.
