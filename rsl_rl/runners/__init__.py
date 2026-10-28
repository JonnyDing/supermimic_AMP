"""Import compatibility for mjlab.rl, which eagerly imports OnPolicyRunner.

WMRL does not instantiate this class. No PPO training implementation is retained.
"""


class OnPolicyRunner:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("PPO runners have been removed; use scripts/train.py with WMRL")
