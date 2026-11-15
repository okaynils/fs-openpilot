# Modal reproduction and artifact locations

These commands use the existing `fsop` volume. Run names are deliberately distinct;
do not re-run generation or fresh training over an existing run. Dataset manifests
include full configuration/seed lists and source hashes. Each generated dataset and
training run retains `source.tar.gz` on the volume. Reproduce from that snapshot;
changed source requires a new dataset/run name rather than silent reuse.

```sh
# Metadata only; refuses to change an existing frozen design.
python scripts/transfer_v2_experiment.py

# Bounded data pilot, then inspection.
modal run scripts/modal_app.py::transfer_v2_data
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v2_preview.py data/sim/transfer_v2_pilot'

# 300-step sequence pilot; successful run is transfer_pilot_v2_retry1.
modal run scripts/modal_app.py::transfer_v2_pilot_train
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v2_pilot_verify.py'

# Complete data, inspect its actual observations, then three sequential 1,500-step fits.
modal run scripts/modal_app.py::transfer_v2_data --name transfer_train_v2 --no-pilot
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v2_preview.py data/sim/transfer_train_v2'
modal run scripts/modal_app.py::transfer_v2_train

# Fixed closed-loop validation snapshots (one comma-separated argument, no spaces).
modal run scripts/modal_app.py::transfer_suite \
  --suite configs/eval/transfer_v2_val.yaml --name transfer_validation_v2 \
  --checkpoints runs/cuda_racing_v6/ckpt/best.pt,runs/transfer_minimal_v1/ckpt/best.pt,runs/transfer_expert_v2/ckpt/step_000500.pt,runs/transfer_expert_v2/ckpt/step_001500.pt,runs/transfer_recovery_v2/ckpt/step_000500.pt,runs/transfer_recovery_v2/ckpt/step_001500.pt,runs/transfer_sequence_v2/ckpt/step_000500.pt,runs/transfer_sequence_v2/ckpt/step_001500.pt
```

Training retains `best.pt` (offline ADE), `last.pt`, and the two named snapshots.
Driving selection is recorded separately; `best.pt` is not automatically the promoted
model. To resume an interrupted fit from its last saved checkpoint, use the matching
archived configuration/source and the same dataset manifest:

```sh
modal run --detach scripts/modal_app.py::train \
  --cfg configs/train/transfer_recovery_v2.yaml \
  --o 'resume=runs/transfer_recovery_v2/ckpt/last.pt'
```

A failure before the first saved checkpoint requires a distinct fresh run name.
The failed pilot `runs/transfer_pilot_v2` remains preserved. Its history-only cache
is also retained; successful stage-one runs use `transfer_baseline_strict_v2_full`.

The final suite must not be used to select checkpoints or tune parameters.
The commands below were fixed before opening its observations.

The predeclared fallback was triggered by worse driving validation in all three arms:

```sh
modal run scripts/modal_app.py::transfer_v2_policy_train
modal run scripts/modal_app.py::transfer_suite \
  --suite configs/eval/transfer_v2_val.yaml --name transfer_policy_validation_v2 \
  --checkpoints runs/transfer_policy_v2/ckpt/step_000500.pt,runs/transfer_policy_v2/ckpt/step_001500.pt

# After both validation reports exist: verify pairing, freeze selection, package/export.
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v2_finalize.py'
```

`transfer_v2_finalize.py` refuses to overwrite a merged report or package. Its merge
checks complete profiles, seeds, recovery offsets, time limits and every simulator,
policy and evaluator source hash before reusing reference results. Expert outcomes
must agree across the two validation executions. Only then is the best new candidate
copied into `runs/transfer_selected_v2/selected.pt`; the original checkpoint is retained.

Post-selection commands (run each only after its prerequisites have completed):

```sh
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v2_metrics.py default'
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v2_metrics.py context'
modal run scripts/modal_app.py::transfer_v2_final_data \
  --selection runs/transfer_validation_combined_v2/selection.json
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v2_metrics.py test'
modal run scripts/modal_app.py::transfer_suite \
  --suite configs/eval/transfer_v2_test.yaml --name transfer_final_v2 \
  --checkpoints runs/cuda_racing_v6/ckpt/best.pt,runs/transfer_minimal_v1/ckpt/best.pt,runs/transfer_selected_v2/selected.pt
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v2_metrics.py runtime_suite'
modal run scripts/modal_app.py::transfer_suite \
  --suite configs/eval/transfer_v2_runtime.yaml --name transfer_runtime_v2 \
  --checkpoints runs/transfer_selected_v2/selected.pt,runs/transfer_selected_v2/candidate.onnx
```

The runtime subset is fixed before test observations: first Formula and first rover,
first track/scene seed, with the single-seed evaluator's alternating recovery rule
(False, True). Both PyTorch and ONNX are evaluated on that identical subset; it is
an additional runtime check, not a replacement for the full independent test suite.

The executed post-selection metrics were bundled sequentially in `python scripts/transfer_v2_post.py`. The archived runtime suite matches the fixed `configs/eval/transfer_v2_runtime.yaml`.


The final same-sequence runtime diagnostic is reproducible with:

```sh
modal run scripts/modal_app.py::sh --cmd 'python scripts/transfer_v2_fresh_parity.py'
```

Use a distinct output path when repeating analysis to preserve the original results.
The post-selection CPU oracle is `python scripts/transfer_v2_final_oracle.py`; it
refuses to overwrite its existing case files. The comparison plot and final metric
summary are reproduced with `scripts/transfer_v2_plot.py` and
`scripts/transfer_v2_final_summary.py` after downloading their linked input reports.
