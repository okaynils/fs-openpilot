# Contributing to fs-openpilot

Thanks for your interest! Contributions are welcome from FS/FSAE teams and individuals:
- bug reports;
- sim and track improvements;
- new baselines, benchmark runs and team profiles;
- ports to other platforms.

The project is maintained as a living research project (see [docs/ROADMAP.md](docs/ROADMAP.md)). Small,
focused PRs with a clear motivation are the easiest to merge.

## Development setup

```bash
git clone https://github.com/okaynils/fs-openpilot && cd fs-openpilot
uv sync --extra sim --group dev
uv run fsop model fetch && uv run fsop model prepare      # needed for model tests and training
uv run pytest                                            # ~10 s; render/model tests skip if unavailable
uv run ruff check src tests scripts
```

Python 3.12, [uv](https://docs.astral.sh/uv/) for environments. The simulator needs OpenGL (macOS, or
Linux with a GPU or Xvfb). Set `FSOP_SKIP_RENDER=1` to skip rendering tests on headless machines.

## Code style

- **Indentation is 2 spaces** (as in openpilot); lines up to 120 characters; `ruff` must pass.
- Match the surrounding code: small modules, plain functions and dataclasses, no new frameworks.
  Configuration goes through dataclasses + OmegaConf (`fsop.config`) and YAML in `configs/`.
- Car- or team-specific values belong in a **profile** (`configs/profiles/`), never in code.
- **Frames.** fsop uses ego = x forward / y left / z up. openpilot's plan frame is y right / z down,
  so convert only through `fsop.geometry.op_frames`.
- **Ports.** Code ported from openpilot (MIT) keeps a docstring note with the source file.
- **Rule constants** go into `fsop.track.rules` with their source section. Mark anything not taken from
  the official rules `verified=False`.

## Tests

- Add or extend a test for every behaviour change. Geometry and conventions are guarded by tests, for
  example YUV bit-exactness, reprojection, warp, plan signs and the referee. Keep those passing.
- Tests must run in seconds without downloaded data. Use synthetic fixtures (see `tests/test_fsoco.py`)
  and `pytest.mark.skipif` for anything that needs the model files.

## Benchmarks

If a change affects model quality, include a benchmark record:

```bash
uv run fsop bench run --name <short_id> --ckpt <ckpt> --model "<what changed>" --train-hardware "<hw>" --notes "<…>"
```

Commit the new `benchmarks/results/*.json` together with the regenerated `benchmarks/README.md` and
README table. Never edit existing records; see [docs/benchmarks.md](docs/benchmarks.md).

## Data and weights

- **Do not commit datasets, checkpoints or FSOCO images.** `data/`, `runs/` and `artifacts/` are
  gitignored.
- To share trained weights, attach them to a GitHub Release, or link them from the benchmark record's
  `notes`.
- Respect the FSOCO terms of use and the licenses listed in the README.

## Pull requests

- Describe **what** changed and **why**, how you tested it, and any benchmark impact.
- Update [CHANGELOG.md](CHANGELOG.md) under *Unreleased*, and update the docs if commands or configs changed.
- One topic per PR; refactors separate from behaviour changes.

## Reporting issues

Please include:
- the command;
- the full error;
- your OS and hardware;
- `uv run python -c "import fsop.bench as b; print(b.hardware())"`.

For sim or model behaviour, a `.rrd` recording or a short video (`fsop viz video`) helps a lot.

## Safety

fs-openpilot is research software that controls (simulated) vehicles. Contributions must not weaken
the separation between this model and the car's safety systems (AS state machine, EBS, RES), and docs
must never present the model as validated for real-world driving without evidence.
