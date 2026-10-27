# Roadmap and lab notes

This is the living plan for fs-openpilot and its maintainer's playground. Status values: 🔜 next ·
🧪 in progress · ✅ done · 💡 idea. Results land in [benchmarks](../benchmarks/README.md) and changes in
the [changelog](../CHANGELOG.md).

## Now

| | item | why / notes |
|---|---|---|
| 🔜 | **Skidpad** | Every policy fails it. Record ~60 route-conditioned skidpad episodes (`fsop sim record --skidpad 60`), check that the desire pulses are actually used, and add skidpad to the open-loop test split. |
| 🧪 | **Fewer cone hits** | Recovery excursions + 7× data took DOO from 7.7 to 1.3 per run (see lab note 2026-10-27). Next: DAgger rounds on top (`fsop eval dagger`), and the 8-track `closed_loop_suite` with heavy DR to get a less noisy number. |
| 🔜 | **FSOCO in the loop** | Import FSOCO, train stage 2 with real images, and report image-head mAP on the FSOCO test split vs. a YOLO baseline (`fsop.baselines.yolo`). |
| ✅ | **Bigger run on a GPU** | `scripts/modal_app.py`: dataset v2 (7× v1) + full-trunk fine-tune on an A100, ~$14 of Modal credit end to end. |

## Next

| | item | notes |
|---|---|---|
| 💡 | Real-car log importer | Synchronized camera, wheel-speed and IMU logs → warped frames + odometry trajectory labels, as a training dataset. |
| 💡 | ROS 2 reference node | Wrap `fsop.runtime.FSDriver`; publish plan, cones and control. |
| 💡 | Vision lap counter | Detect the big orange gates, so the mission layer stops using sim progress. |
| 💡 | Cone map + racing line | Fuse cone detections with odometry (lightweight SLAM). Use the end-to-end plan on lap 1 and a map-based racing line on laps 2–10. |
| 💡 | Photoreal validation set | BlenderProc / Unreal renders of the same tracks to measure the sim-to-real gap. |
| 💡 | Jetson benchmark | Runtime latency on Orin NX/AGX, plus TensorRT fp16 drift. |
| 💡 | FSAE / FS-AI rule profiles | Make `fsop.track.rules` configurable per competition. |
| 💡 | Better BEV lifting | Per-camera features (un-fuse the trunk) or depth-aware lifting. |

## Playground

`playground/` is gitignored: scratch notebooks, one-off scripts and experiment configs live there. Once
an experiment is worth keeping, it graduates into `src/` with a test, a config in `configs/`, a lab
note below and, if it changes model quality, a benchmark record.

## Lab notes

Short, dated notes on what was tried and what was learned. Newest first.

- **2026-10-27: recovery excursions + first cloud run** (Modal: CPU containers for data, A100 for training)
  - **Diagnosis:** in closed loop `m1_stage2` drifted up to 1.2–4.8 m off-centre, but 90% of v1's frames
    were within 0.31 m and only 0.5% beyond 0.8 m (where the car starts touching cones). The model had
    barely seen the states it fails in.
  - **Sim change:** during noise bursts the expert now drives a parallel line up to ±1 m off-centre and
    returns afterwards; labels stay the centreline expert. Frames beyond 0.8 m: 0.5% → 5.9%.
  - **Dataset v2:** 480 trackdrive + 160 `tight` tracks, 2 random starts each, 40 skidpad, 20 accel:
    1438 episodes / 284k frames (7.2× v1), recorded in ~15 min on up to 100 Modal CPU containers.
  - **Training:** full-trunk stage 2 from `m1_stage2` (7k steps, bs 64, ~335 samples/s), then the
    history cache recomputed from that trunk and 4k more steps at a lower LR. The refresh pass helps:
    1 s lateral error 0.25 → 0.23 m, DOO 2.0 → 1.3 per run.
  - **Result (`a100_v2_refresh`):** lat err 1 s / 2 s 0.43 / 1.57 → 0.23 / 0.78 m, cone F1 0.40 → 0.71,
    trackdrive 3/3 with 7.7 → 1.3 DOO (privileged `geometric_gt`: 0.7). Skidpad still fails (56% done).
  - **Infrastructure lessons:**
    - Network volumes are slow per file (~50 ms of metadata per file), so frames are now sharded zarr
      (5 files per episode instead of ~400).
    - Spawn GPU jobs instead of waiting on them: a local client error stops even a `--detach`ed app.

- **2026-10-23: first real onboard video** (FSG onboard clip, 52 s, 878×522 screen recording; uncalibrated
  wide-angle camera, estimated ~1.2 m high and pitched ~36° down)
  - **Zero-shot** (reference model, level virtual camera): about 60% of the wide input lies above the
    real image (black). The plan is stuck in a hard left curve and only a few far cones are detected.
  - **Adapted:** profile with the virtual camera pitched 21° (wide slot covers about 2–26 m), boundary-line
    DR added, 192 sim episodes recorded, then `adapt_stage2` fine-tuning from the reference model (2.5k
    steps, about 25 min on the M1).
    - Sim validation lateral error at 1 s: 3.9 m → 1.0 m.
    - On the real clip the plan follows some turns and is no longer biased left, but cone detection is
      sparse (no real images in training yet).
  - **Takeaways:**
    - Real FS cameras are often steeply pitched, so profiles need a per-camera virtual pitch.
    - Both slots share one mount, so the road slot loses its long-range role. Per-slot mounts are worth
      adding.
    - FSOCO in stage 2 is the obvious next step for the cone head.
    - The cone colour layout in that clip didn't consistently follow blue-left/yellow-right.

- **2026-10-22: first end-to-end results** (M1, dataset v1: 160 tracks, 39k frames)
  - **Frozen trunk:** stage 1 plateaus. The road-trained features carry little cone geometry.
  - **Fine-tuning the last trunk stage** (stage 2) cuts the 1 s lateral plan error from 1.28 m
    (untouched) to 0.43 m.
  - **Cones:** the learned BEV head is weak (F1 0.13). The image head plus ground projection reaches
    F1 0.40, with 0.17 m position error.
  - **Launch:** standing still, the model cannot know whether to go; this belongs to the mission
    layer.
  - **ONNX Runtime with CoreML** drifts by about 3 m at the 10 s horizon (fp16), so the runtime
    defaults to CPU/CUDA.
