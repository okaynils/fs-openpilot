# Reproduction commands

Executed regression and export-contract check:

```sh
modal run scripts/modal_app.py::sh --cmd 'python -m pytest tests -q --junitxml=runs/transfer_tests_v5/junit.xml && python scripts/transfer_v5_contract_probe.py'
```

Paired 300-step pilots launched after the contract check passed:

```sh
modal run scripts/modal_app.py::transfer_v5_state_pilots
```

The pilot entrypoint first proves exact equality of all baseline/v1 vision tensors
and checks every existing cache identity without modifying any cache. It then
launches one constant-input control and one measured-speed fit. Both use distinct
run names and preserve v1 initialization, dataset identity and source provenance.
Do not rerun an existing run name; use a documented resume after a verified failure.


Executed pilot evaluation and export commands (each supplied to the Modal `sh` entrypoint):

```sh
python scripts/transfer_v3_plan_probe.py runs/transfer_state_constant_pilot_v5/ckpt/step_000300.pt runs/transfer_state_constant_probe_v5
python scripts/transfer_v3_plan_probe.py runs/transfer_state_speed_pilot_v5/ckpt/step_000300.pt runs/transfer_state_speed_probe_v5
python scripts/transfer_v5_state_probe.py
python scripts/transfer_v5_runtime_probe.py
```

Both short driving gates passed. Launched the predeclared full comparison:

```sh
modal run scripts/modal_app.py::transfer_v5_state_train
```

Only fixed snapshots 250/1000 from these fits are candidates for complete-configuration
validation. No v3 final-test observations have been requested.


Full fits completed; complete-configuration validation launched:

```sh
modal run scripts/modal_app.py::transfer_suite --suite configs/eval/transfer_v2_val.yaml --name transfer_state_validation_v5 --checkpoints runs/cuda_racing_v6/ckpt/best.pt,runs/transfer_minimal_v1/ckpt/best.pt,runs/transfer_state_constant_v5/ckpt/step_000250.pt,runs/transfer_state_constant_v5/ckpt/step_001000.pt,runs/transfer_state_speed_v5/ckpt/step_000250.pt,runs/transfer_state_speed_v5/ckpt/step_001000.pt
```

After all paired cases complete, select with:

```sh
python scripts/transfer_v2_select.py benchmarks/transfer_v5/validation.json benchmarks/transfer_v5/selection.json
```

Selection completed: speed step 250 ranks best among new snapshots, but loses
to baseline/v1. No default-retention or final-test evaluation was launched.


Metered usage (executed with authenticated Modal SDK 1.6.2):

```sh
python scripts/transfer_costs.py --ledger benchmarks/transfer_v5/modal_runs.json --start 2026-11-13 --end 2026-11-14T17:07:00 --output benchmarks/transfer_v5/costs_reproducible_1707.json
```

The $2.03764786 snapshot excludes the still-unsettled 17:00 interval. Preserve it
and choose a new output path for a later refresh.
