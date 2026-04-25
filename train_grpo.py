"""
Convenience launcher for GRPO training.

Usage:
  python train_grpo.py --episodes 200 --group-size 8
"""

import runpy


if __name__ == "__main__":
    runpy.run_module("training.grpo_training", run_name="__main__")

