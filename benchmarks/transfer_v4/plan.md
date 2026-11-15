# Transfer v4: late vision adaptation with current-weight temporal history

V3's guarded-data candidates still pass 2/4 validation cases versus baseline/v1 3/4.
The paired departure intervention has an exactly identical 12-second prefix; replacing
only the predicted plan with the expert plan through the same follower then completes
with zero contacts/off-course events. Policy-only loses visible cones at 15.6 s,
goes off course at 16.5 s and reaches 85 m reference-path error by 60 s. This supports
investigating prediction while the course is visible, before later visibility loss.
The measured reset-speed ambiguity is separate; no speed input is added in this arm.

Hypothesis: the frozen road-trained visual representation limits low-viewpoint cone
geometry adaptation. Compare two matched image-training arms initialized from the
v1 calibrated candidate (the strongest existing validation reference), using the
preserved guarded v3 dataset: frozen vision versus last-stage vision adaptation.
Both keep the pretrained policy frozen, train the same calibrated adapter and cone
heads, and use the same losses, sample seed, batch size and schedule. Only the set
of trainable vision parameters differs. No architecture/input/controller changes.

Correct the image-training history approximation first. The old stage-2 path reads
history cached with the initialization trunk even as that trunk changes. New opt-in
`fresh` history encodes recorded past frame pairs using current vision weights, with
no gradient through prior time steps (matching detached runtime queues). Preserve
zero padding and 0.2-second spacing. The prepared policy graph gathers exactly its
last nine feature slots: current plus eight previous. Verify this against the actual
graph and numerical output before omitting unused older slots. No new caches are
silently substituted; historical `cached` training remains supported and explicit.

Pilot: 100 steps per arm, batch 8, seed 184191, lr 3e-4, vision multiplier 0.1,
policy frozen, original plan/BEV/image losses and L2-SP 1e-3, warmup 100. Verify
current-weight history parity, nonzero finite gradients in the intended vision group,
finite losses, peak GPU memory below 34 GiB and fit time below 2 seconds/step before
scaling. Inspect fixed short driving probes for gross degradation, without selecting
from final data. A failed engineering gate must be diagnosed before a full fit.

Conditional full comparison: one fresh 1,000-step fit per arm, fixed snapshots at
250 and 1,000. Validation loss is a behavior diagnostic on the same bounded batch
subset for both arms; driving on complete held-out validation configurations selects
snapshots. Re-evaluate baseline/v1 references with the updated code, and retain the
original default only unless transfer/default gates justify promotion. No learning-rate
or layer sweep. The already-frozen v3 final suite remains unrendered and may be opened
only after a validation choice and default-retention checks.
