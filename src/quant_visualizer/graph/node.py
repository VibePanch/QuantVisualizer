"""Framework-independent graph node metadata."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """A model layer represented in the visualizer graph."""

    id: str
    type: str
    input_shape: Any = None
    output_shape: Any = None
    parameters: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
