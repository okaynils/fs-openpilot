# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project uses [SemVer](https://semver.org/) once it reaches 1.0. Benchmark results are tracked
separately in [benchmarks/](benchmarks/README.md).

## [Unreleased]

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
