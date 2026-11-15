"""Paired correction labels share timestamps, without compressing history."""
import importlib.util
from pathlib import Path

import numpy as np


def test_matching_masks_preserve_invalid_and_surplus_observations():
  spec = importlib.util.spec_from_file_location('stage', Path(__file__).parents[1]/'scripts/transfer_v7_stage.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  left = np.array([True, False, True, True, True])
  right = np.array([True, True, False, True])
  a, b = module.paired_masks(left, right)
  assert a.tolist() == [True, False, False, True, False]
  assert b.tolist() == [True, False, False, True]
  assert len(a) == len(left) and len(b) == len(right)
  assert left.tolist() == [True, False, True, True, True]
  assert a.sum() == b.sum() == 2  # includes the stationary initial observation
