"""Backend-specific factories required by the SOMA model program."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from body_models.bodies.soma.correctives import CorrectiveNetwork
from body_models.runtime import Runtime


@dataclass(frozen=True)
class SomaLowerings:
    """Construct SOMA components that genuinely differ by array backend."""

    corrective_network: Callable[[Runtime, Any], CorrectiveNetwork]
    identity_source: Callable[[str, Any], Any]


__all__ = ["SomaLowerings"]
