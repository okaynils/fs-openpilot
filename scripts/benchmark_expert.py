"""Matched, headless racing/centerline teacher comparison (no neural checkpoint).

Run: uv run python scripts/benchmark_expert.py --out runs/expert_racing.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from fsop.sim.env import FSEnv, SimCfg
from fsop.sim.expert import sample_expert
from fsop.track.fixed_layouts import acceleration, skidpad
from fsop.track.generate import generate_track


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, default=Path("runs/expert_racing.json"))
  parser.add_argument("--seeds", type=int, nargs="+", default=[2, 7, 19])
  parser.add_argument("--episode-seeds", type=int, nargs="+", default=[0, 1])
  args = parser.parse_args()
  results = []
  sources = [Path(__file__), Path("src/fsop/sim/raceline.py"), Path("src/fsop/sim/expert.py"),
             Path("src/fsop/sim/vehicle.py"), Path("src/fsop/sim/env.py"), Path("src/fsop/sim/randomization.py")]
  hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
  tracks = [generate_track(seed) for seed in args.seeds] + [skidpad(), acceleration()]
  for track in tracks:
    if track.closed:
      track.laps = 2
    for seed in args.episode_seeds:
      params = sample_expert(np.random.default_rng(seed), track.discipline)
      for mode in ("centerline", "racing"):
        cfg = SimCfg(render=False)
        cfg.noise.enabled = False
        env = FSEnv(track, cfg, seed=seed, expert_params=replace(params, path_mode=mode))
        while not env.done:
          env.step(*env.expert_action())
        result = {"track": track.id, "seed": seed, "mode": mode, "expert": asdict(env.expert_params),
                  "vehicle": asdict(env.vp), "summary": env.referee.summary()}
        line = env.mission.racing_line
        if line is not None:
          result["planner"] = {"optimized": line.optimized, "time": line.travel_time,
                               "baseline_time": line.baseline_time, "baseline_feasible": line.baseline_feasible,
                               "status": line.solver_message,
                               "converged": line.converged, "iterations": line.iterations}
        results.append(result)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"protocol": "matched expert, light DR, no noise, standing start, "
                                       "two trackdrive laps; prediction times are flying reference times",
                                       "source_sha256": hashes, "results": results}, indent=2))
        summary = result["summary"]
        print(track.id, seed, mode, summary["pass"], summary["time"], summary["doo"], flush=True)


if __name__ == "__main__":
  main()
