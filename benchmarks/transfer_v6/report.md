# Post-v5 diagnosis

No new training has been launched. The v5 candidates failed complete-configuration
validation; baseline/v1 remain stronger, and the v3 final suite remains sealed.

Three testable explanations guide the next diagnostic: adapter feature magnitudes
may violate the frozen policy input distribution; reset/low-speed coverage is sparse;
and metric trajectory loss may inadequately supervise steering-critical errors.
The first CPU-only probe measures raw/adapted feature norms on 64 fixed validation
samples (seed 184194) using actual checkpoint weights and model.plan. It bypasses
policy execution after capturing the inputs. No weights or data are modified.
Norm changes alone would not establish a driving failure cause.

```sh
modal run scripts/modal_app.py::transfer_feature_audit
```


The [audit](feature_norm_audit.json) confirms baseline cached norms 0.999951–1.000032.
V1 changes them up to 3.314; selected v5 speed-250 up to 3.844 (median ratio 1.565),
and speed-1000 up to 3.963. This is substantial distribution drift, but not yet a
causal explanation. The exact executed audit source is preserved under
`artifacts/experiments/transfer_feature_norm_audit/source.py`.

Predeclared intervention: v1 and selected v5 speed-250, each in unchanged and
current-feature-normalized arms on existing validation case `val_v2_2`, track 98011,
scene 5101, 60 seconds. The first 12 seconds must be identical (asserted state,
action and feature-norm prefix error below 1e-6). After frame 240, normalize only
the current adapted feature to unit length. Keep raw history, weights, geometry,
speed and follower identical. Expert labels are visualization only. Inspect valid
completion, off-course events, contacts and reference-path error, not only a norm
statistic. No additional fitting or final-test evaluation is authorized by a norm
change alone. Existing checkpoints/runtime remain unchanged.

```sh
modal run scripts/modal_app.py::transfer_norm_interventions
```

## Paired intervention outcome

Both jobs completed. The maximum state/action/feature-norm difference before
12 seconds was exactly **0.0** in both pairs. Neither intervention recovered
valid completion.

| Checkpoint | Arm | Contacts | Off-course events | First off-course (s) | Final reference error (m) |
|---|---|---:|---:|---:|---:|
| v1 | unchanged | 0 | 1 | 21.35 | 56.879 |
| v1 | normalize after 12 s | 0 | 1 | 17.95 | 56.587 |
| v5 speed-250 | unchanged | 0 | 1 | 18.25 | 44.557 |
| v5 speed-250 | normalize after 12 s | 1 | 1 | 17.05 | 72.489 |

Sources: [v1 paired trace](norm_v1.json), [speed-250 paired trace](norm_speed.json),
[intervention source](../../scripts/transfer_norm_intervention.py). Both intervention
renderings were inspected: [v1](norm_v1_intervened.jpg),
[speed-250](norm_speed_intervened.jpg). Cones remained visible at intervention
time; subsequent predictions took the vehicles off course. Exact executed source
archives are preserved under `artifacts/experiments/transfer_norm_v1_probe_v6/`
and `artifacts/experiments/transfer_norm_speed_probe_v6/`.

**Decision:** do not add inference-time normalization to runtime. It caused earlier
off-course events in both tests and an additional contact for speed-250. This
rejects this particular post-hoc fix on this departure; it does not establish that
norm-constrained retraining could never help. No checkpoint changed, no new fit
was launched, and the v3 final configurations remain sealed.

## Loss-gradient diagnostic (predeclared)

Ranked remaining explanations: (1) metric trajectory supervision inadequately
weights steering-critical errors; (2) auxiliary image/BEV gradients interfere with
partial vision adaptation; (3) sparse starts/stops limit learned speed sensitivity.
The speed coverage audit supports (3), but does not explain all steady departures.

Before further fitting, measure per-loss gradient norms and cosines on the same
32 fixed training samples (seed 184195) at v1 initialization and the v4 last-layer
pilot. Use the actual Trainer forward/loss paths, fresh history and FP32. No
optimizer update. If auxiliary interference is material, auxiliary gradients
should dominate and oppose trajectory gradients on these samples. Steering loss
is a diagnostic direction, not an added training term. Local gradient agreement
or conflict alone cannot establish a closed-loop cause. The Trainer reported zero missing/unexpected initialization tensors: the image
head also loads from v1. Sampling differs from the original v4 optimization
batches, so this does not replay its first update.

```sh
modal run scripts/modal_app.py::transfer_gradient_audit
```

### Gradient audit outcome

[Measured gradients](gradient_audit.json), [checkpoint/data provenance](gradient_provenance.json),
[audit source](../../scripts/transfer_gradient_audit.py). All eight FP32 batches
completed without an optimizer update. Auxiliary-to-plan gradient norm ratios
were 0.198–0.491 at initialization and 0.243–0.505 after the v4 pilot.
Image-loss gradients were zero for the selected trainable vision layers.
BEV/plan cosine similarities ranged from −0.092 to +0.172. These samples do not
support dominant opposing auxiliary gradients as the primary explanation.

Steering/plan alignment was variable (−0.100 to +0.736 initially); after the pilot
it was weakly positive (+0.183 to +0.207). This is consistent with imperfect
alignment of objectives, but does not establish that another loss-weight change
would improve driving. The earlier closed-loop steering-loss experiment already
failed its gate. No extra training sweep follows from this audit.

Local targeted regression verification: **17 passed** across initial-frame
sampling, fresh-history reconstruction and speed-conditioning tests. This is not
a new full-suite result. The exact executed diagnostic source archive includes
an inaccurate comment about a missing image head; logs show zero missing tensors.
The local comment was corrected without changing behavior.

All three diagnostic jobs completed. The [shutdown observation](shutdown_audit.json)
shows the latest job stopped; older jobs aged out of the recent-app listing.
[Metered costs](costs_partial_2045.json) currently total **$0.20639793**, excluding
unsettled usage for the latest gradient job. This is a partial estimate, not an
invoice. No model promotion or final-test access occurred.

A later [same-day billing refresh](costs_reproducible_2138.json) includes the
20:00 gradient-audit interval and totals **$0.24132181**. The earlier $0.20639793
partial snapshot is preserved. These remain app-scoped metered records, not an
invoice.
