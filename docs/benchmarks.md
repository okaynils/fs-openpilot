# Benchmark protocol

Benchmarks track progress over time. Each record is a JSON file in `benchmarks/results/`. The README
table and `benchmarks/README.md` are generated from these files with `fsop bench table`.

## Standard benchmark (`fsop bench run`)

1. **Open loop**, `fsop eval open-loop`, on the **test** split of `data/sim/v1`.
   - The test tracks are a deterministic, hash-based ~10% of tracks, held out by track rather than by
     episode.
   - The first 5 frames of every episode are skipped (history warm-up).
2. **Closed loop**, on `configs/eval/closed_loop_quick.yaml`:
   - 3 held-out trackdrive tracks at 1 lap each, the skidpad and acceleration layouts;
   - light DR, standing start, no steering noise;
   - the same mission layer for every policy (launch, stop-zone braking).

## Metrics

| metric | definition |
|---|---|
| lat err 1 s / 2 s | mean \|predicted − expert\| lateral offset of the plan at t = 1 s / 2 s, ego frame [m] |
| ADE 4 s | mean Euclidean position error over plan points up to 4 s [m] |
| heading 1 s | mean absolute heading error of the plan at 1 s [deg] |
| cone F1 | per-class Hungarian matching within 0.75 m of visible ground-truth cones, BEV, all ranges. Uses image-head + ground projection when available, otherwise the BEV head |
| trackdrive / accel / skidpad | closed-loop runs passed (all laps, stopped in the zone, not DQ'd) / runs; trackdrive also shows mean DOO (cones down or out) per run |

The "expert" is the counterfactual pure-pursuit rollout used for training labels. It is a reference
behaviour, not an optimum: lower plan error means "drives more like the expert", not "faster".

## Adding a benchmark

```bash
uv run fsop bench run --name <short_id> --ckpt runs/<run>/ckpt/best.pt \
    --model "<what changed>" --train-hardware "<e.g. RTX 4090>" --notes "<data, steps, tricks>"
```

`bench run` writes the record and regenerates the tables.
- **Results from elsewhere** (another machine, a pre-computed evaluation): use
  `fsop bench add --open-loop ol.json --closed-loop report.json …`.
- **Submitting:** open a PR with the JSON file and the regenerated tables. Fill in the git SHA,
  dataset and training config. Records are append-only: never edit old results; add a new one.

When the protocol changes, for example a new dataset version or suite, bump the version in the
record's `notes` and start a new comparable series rather than mixing the two in the same table.
