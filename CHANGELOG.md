# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project uses [SemVer](https://semver.org/) once it reaches 1.0. Benchmark results are tracked
separately in [benchmarks/](benchmarks/README.md).

## [Unreleased]

### Added
- `fsop viz real`: run a model on a real onboard video. The output shows the plan and cones projected
  back through the physical lens, the model-frame inputs and an ego view.
- `fsop.geometry.warp.project_to_camera`: ego frame → physical camera pixels, distortion included.
- Boundary-line domain randomization: white or green chalk lines along the track edges, with gaps, in
  about 50% of light/heavy episodes.
- `configs/train/adapt_stage2.yaml`: adapt the reference model to a new team profile.
- Recovery excursions (`fsop sim record --offset-max 1.0`): during noise bursts the expert drives a line
  up to ±1 m beside the centreline, then returns; labels stay the centreline expert. Frames beyond 0.8 m
  off-centre (where cones get hit) go from 0.5% to 5.9% of the data.
- `fsop sim record --per-track N`: several random-start episodes per track.
- `scripts/modal_app.py`: run the pipeline on [Modal](https://modal.com). Data generation fans out over
  CPU containers (headless EGL rendering); caching, training and benchmarks run on a GPU.
- `fsop data cache-features --hidden-only --tag`: history-only caches for stage 2 (about 1 KB/frame).
- `configs/train/cuda_stage2_v2{,_refresh}.yaml`: GPU recipe for dataset v2, with a second pass on a
  history cache recomputed from the fine-tuned trunk.

### Changed
- Recorded frames are stored as one zarr shard per camera (per-frame chunks inside), so an episode is
  5 files instead of about 400. `fsop.sim.recorder.repack_frames` converts older datasets.
- The renderer falls back to headless EGL on Linux machines without a display.
- Train configs inherit recursively (`defaults:` chains).
- Feature caching prefetches episode frames on background threads.

### Fixed
- Fisheye warp produced scrambled remap tables: `cv2.fisheye.distortPoints` needs (1, N, 2) input.
  A test now covers it.
- `ModelFrameWarper` misread negative-stride input such as `bgr[..., ::-1]`.

## [0.2.0] - 2026-10-22

### Added
- Team profiles (`configs/profiles/`): vehicle parameters, camera mount and physical cameras. Use them
  with `--profile` on `sim record`, `sim drive`, `eval closed-loop`, `eval dagger`, `viz video` and
  `bench run`.
- `fsop.geometry.warp`: undistort and rotate real camera images into openpilot's model-frame slots.
  One camera can feed both slots. Includes `fsop camera show` and `fsop camera warp`.
- `fsop.runtime.FSDriver`: deployment runtime from images to plan, cones and steering/acceleration.
  Works with an ONNX file (CPU/CUDA) or a checkpoint.
- ONNX export now appends the image-head cone detections for each camera slot.
- Benchmarks: `fsop bench run/add/table`, versioned records in `benchmarks/results/`, and generated
  tables.
- `fsop viz video` and `fsop viz gif` (the README demo).
- Docs: architecture, team guide, benchmark protocol, roadmap; CONTRIBUTING; CI.

### Changed
- Camera domain randomization is now relative to the profile's nominal mount
  (`cam_height_jitter` replaces `cam_height` ranges). The defaults reproduce the previous distribution.

## [0.1.0] - 2026-10-22

### Added
- Initial pipeline:
  - rule-compliant track generation and validator;
  - Panda3D simulator with expert, noise injection and referee;
  - supercombo ONNX → PyTorch conversion with parity check;
  - cone heads;
  - stage 1/2 training on MPS/CUDA and FSOCO import;
  - open-/closed-loop evaluation, DAgger and ONNX export;
  - Rerun visualization.
