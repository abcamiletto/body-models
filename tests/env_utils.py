"""Environment knobs for scaling test cost in CI."""

from __future__ import annotations

import os


def int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def case_count(default: int) -> int:
    return int_env("BODY_MODELS_TEST_CASES", default)


def gradcheck_samples(default: int) -> int:
    return int_env("BODY_MODELS_GRADCHECK_SAMPLES", default)
