# V5: measured-speed conditioning, capacity-matched control

V3 established identical pixels/geometry/empty history with different feasible metric
trajectory targets at different speeds. V4 found BF16-to-FP32 steering differences
below 0.001 rad on average on 64 fixed validation frames; both failed pilots worsened
steering error in both precisions despite improving NLL. Precision is not supported
as the dominant explanation for those failures.

Add one zero-initialized current-feature residual adapter taking visual features,
existing calibrated geometry and measured forward speed in m/s (normalized by 15).
Compare actual speed with a constant zero input in the same architecture. Both use
identical v1 initialization, guarded-v3 dataset, sample order, losses and optimizer.
Freeze vision and policy. Existing visual caches remain valid: speed is already
recorded per frame and is not baked into the visual features. No new geometry,
privileged map state or sensor is supplied. Runtime already receives speed for its
controller; new versioned checkpoints explicitly require it for prediction too.

Bounded pilot: 300 steps per arm, batch 64, seed 184193, lr 3e-4, warmup 100,
original plan/BEV losses, no added steering loss. Check zero-step parity, actual speed
sensitivity versus constant-input invariance, finite training and compatibility.
Run identical short driving probes. Do not scale a degraded pilot. Only a useful
pilot may proceed to one fixed 1000-step comparison and complete-configuration
validation against baseline/v1, followed by default retention and the still-sealed
v3 final suite if warranted. No layer, learning-rate or loss-weight sweep.

The speed-input hypothesis concerns missing state information. It is not assumed
to explain all steady-speed failures, and adding an input alone is not success.


Both pilots passed the short gate (0 contacts/off-course events, tail offsets
1.033/1.128 m versus initialization 1.189 m). The authorized full comparison will
use fresh 1000-step fits and fixed snapshots at 250 and 1000, specified before
launch. The constant control outperformed speed on this short metric; speed did
not resolve the controlled reset ambiguity. Full validation remains necessary.
