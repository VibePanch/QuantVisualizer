"""Framework-independent model graph data structures."""

from dataclasses import dataclass, field

from quant_visualizer.graph.node import Node


@dataclass
class ModelGraph:
    """A directed graph of model layers represented by node-id pairs."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def get_node(self, node_id: str) -> Node:
        """Return a node by id or raise KeyError."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"No graph node named {node_id!r}")
