# Reproduction commands

Preserve prior directories and source archives. The following test run was executed:

```sh
modal run scripts/modal_app.py::sh --cmd 'python -m pytest tests -q --junitxml=runs/transfer_tests_v4/junit.xml'
```

The paired pilots were executed after 104 regression tests passed. The full-fit command
remains conditional on the short driving probes:

```sh
modal run scripts/modal_app.py::transfer_v4_vision_train
modal run scripts/modal_app.py::transfer_v4_vision_train --no-pilot
```

The first command runs one 100-step pilot for each arm. The second runs the fixed
1,000-step fits only after engineering gates pass. Both use distinct run names,
original v1 candidate initialization and the preserved guarded-v3 dataset. Historical
`cached` history remains available for reproduction; these arms explicitly require
`fresh` history and raw frames, without cached hidden-feature substitution.


Executed fixed driving checks (each command supplied to `modal run scripts/modal_app.py::sh --cmd`):

```sh
python scripts/transfer_v3_plan_probe.py runs/transfer_vision_frozen_pilot_v4/ckpt/step_000100.pt runs/transfer_vision_frozen_probe_v4
python scripts/transfer_v3_plan_probe.py runs/transfer_vision_last_pilot_v4/ckpt/step_000100.pt runs/transfer_vision_last_probe_v4
python scripts/transfer_v3_plan_probe.py runs/transfer_minimal_v1/ckpt/best.pt runs/transfer_vision_init_probe_v4
```

Both pilots failed the driving gate relative to unchanged initialization. The
`--no-pilot` full-fit command above was **not executed**. Do not rerun any existing
run name; preserve its checkpoints and source archive.
