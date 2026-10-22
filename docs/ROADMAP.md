# Roadmap and lab notes

This is the living plan for fs-openpilot and its maintainer's playground. Status values: 🔜 next ·
🧪 in progress · ✅ done · 💡 idea. Results land in [benchmarks](../benchmarks/README.md) and changes in
the [changelog](../CHANGELOG.md).

## Now

| | item | why / notes |
|---|---|---|
| 🔜 | **Skidpad** | Every policy fails it. Record ~60 route-conditioned skidpad episodes (`fsop sim record --skidpad 60`), check that the desire pulses are actually used, and add skidpad to the open-loop test split. |
| 🔜 | **Fewer cone hits** | Trackdrive passes, but with about 8 DOO per run. Run DAgger rounds (`fsop eval dagger`), add more tracks and the `tight` track preset, and retrain. |
| 🔜 | **FSOCO in the loop** | Import FSOCO, train stage 2 with real images, and report image-head mAP on the FSOCO test split vs. a YOLO baseline (`fsop.baselines.yolo`). |
| 🔜 | **Bigger run on a GPU** | Run `cuda_stage2` with a full-trunk fine-tune on 5–10× more data. |

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
