# Modal commands

Executed sequentially, preserving all prior datasets/checkpoints:

```sh
modal run scripts/modal_app.py::transfer_v7_refresh
modal run scripts/modal_app.py::transfer_v7_expert_control
modal run scripts/modal_app.py::transfer_v7_inspect
modal run scripts/modal_app.py::transfer_v7_guard_probe
modal run scripts/modal_app.py::transfer_v7_guarded_refresh
modal run scripts/modal_app.py::transfer_v7_stage
modal run scripts/modal_app.py::transfer_v7_train
```

The expert-control batch was cancelled after its first completed control; do not
rerun it under the same name. All collection/staging commands publish distinct
roots and refuse incompatible reuse. Training is resumable through the existing
Trainer checkpoint/optimizer/provenance contract; do not launch a duplicate job.

Prepared driving comparison (not yet launched):

```sh
modal run scripts/modal_app.py::transfer_suite --suite configs/eval/transfer_v2_val.yaml --name transfer_refresh_validation_v7 --checkpoints runs/cuda_racing_v6/ckpt/best.pt,runs/transfer_minimal_v1/ckpt/best.pt,runs/transfer_state_speed_v5/ckpt/step_000250.pt,runs/transfer_refresh_previous_pilot_v7_retry1/ckpt/step_000300.pt,runs/transfer_refresh_current_pilot_v7/ckpt/step_000300.pt
```

Apply the fixed gate after all four complete configurations finish:

```sh
python scripts/transfer_v7_select.py benchmarks/transfer_v7/validation.json benchmarks/transfer_v7/selection.json
```


Local compatibility checks executed after the Modal spending-limit rejection:

```sh
.venv/bin/python scripts/transfer_v7_local_export.py
.venv/bin/pytest -q
.venv/bin/pytest -q tests/test_reprojection.py tests/test_sim.py::test_venue_scene_sampling_and_render
```

The second test invocation needed local graphics access outside the sandbox.
Fresh rendered temporal parity is run separately with the same graphics access:

```sh
.venv/bin/python scripts/transfer_v7_rendered_parity.py
```


Supplemental local CPU driving validation (launched with local graphics access):

```sh
.venv/bin/python scripts/transfer_v7_local_driving.py
```

Exact executed source is archived in `local_driving/source.tar.gz`; a subsequent
line-wrap correction to the script changes no behavior.
