"""Backend-agnostic styling and rendering for reaction network graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from pyvis.network import Network
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from rdkit.Chem import Draw

from .space import MoleculeNode


@dataclass
class GraphStyle:
    """Visual attributes for rendering a modification network"""

    support_color: str = "#8FBC8F"
    generated_color: str = "#4682b4"
    support_size: int = 20
    generated_size: int = 10
    background_color: str = "#ffffff"
    font_color: str = "black"

    @classmethod
    def default(cls) -> "GraphStyle":
        return cls()

    @classmethod
    def dark(cls) -> "GraphStyle":
        return cls(background_color="#000000", font_color="#fffacd")

    @classmethod
    def mellow(cls) -> "GraphStyle":
        return cls(
            support_color="#E76F51",
            generated_color="#457B9D",
            background_color="#F1FAEE",
            font_color="#2C3E50",
        )

    @classmethod
    def contrast(cls) -> "GraphStyle":
        return cls(
            support_color="#E71D73",
            generated_color="#6AAFE6",
            background_color="#2C3E50",
            font_color="#F8BBD0",
        )

    @classmethod
    def appeal(cls) -> "GraphStyle":
        return cls(
            support_color="#FF6347",
            generated_color="#4169E1",
            background_color="#F0F8FF",
            font_color="#2E3A8C",
        )

    @classmethod
    def blues(cls) -> "GraphStyle":
        return cls(
            support_color="#4682B4",
            generated_color="#87CEEB",
            background_color="#F8F8FF",
            font_color="#4682B4",
        )

    def is_support(self, molecule: MoleculeNode | None) -> bool:
        return bool(molecule and molecule.metadata.get("type") == "support")

    def node_color(self, molecule: MoleculeNode | None) -> str:
        return self.support_color if self.is_support(molecule) else self.generated_color

    def node_size(self, molecule: MoleculeNode | None) -> int:
        return self.support_size if self.is_support(molecule) else self.generated_size


# Layout name -> callable(graph) -> pos dict
_LAYOUTS: dict[str, Callable[[nx.DiGraph], dict]] = {
    "spring": lambda g: nx.spring_layout(g, seed=42),
    "tree": lambda g: graphviz_layout(g, prog="dot"),
}


class SpaceView:
    """A styled, renderable view of a reaction network graph.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        style: GraphStyle | None = None,
        layout: str | Callable[[nx.DiGraph], dict] = "spring",
    ):
        self.graph = graph
        self.style = style or GraphStyle.default()
        self.layout = layout

    def _molecule(self, node) -> MoleculeNode | None:
        return self.graph.nodes[node].get("molecule")

    def _compute_layout(self) -> dict:
        if callable(self.layout):
            return self.layout(self.graph)
        try:
            layout_fn = _LAYOUTS[self.layout]
        except KeyError:
            raise ValueError(f"Unknown layout '{self.layout}', expected one of {list(_LAYOUTS)} or a callable")
        return layout_fn(self.graph)

    def to_html(
        self,
        filename: str = "graph.html",
        height: str = "800px",
        width: str = "100%",
        notebook: bool = False,
    ) -> str:
        """Export the graph to an interactive pyvis HTML file. Returns the filename."""
        net = Network(
            height=height,
            width=width,
            notebook=notebook,
            bgcolor=self.style.background_color,
            font_color=self.style.font_color,
            directed=True,
        )
        net.from_nx(self.graph)

        for node in net.nodes:
            molecule = self._molecule(node["id"])
            node["color"] = self.style.node_color(molecule)
            node["size"] = self.style.node_size(molecule)

        net.save_graph(filename)
        return filename

    def to_matplotlib(self, figsize: tuple = (8, 6), with_labels: bool = True, show: bool = True):
        """Render the graph as a static matplotlib plot using the configured layout."""
        pos = self._compute_layout()
        node_colors = [self.style.node_color(self._molecule(node)) for node in self.graph.nodes]

        plt.figure(figsize=figsize)
        nx.draw(self.graph, pos, with_labels=with_labels, node_color=node_colors, arrows=True)
        if show:
            plt.show()

    def to_structures(self, figsize: tuple = (12, 8), node_size: float = 0.8, show: bool = True):
        """Render the graph with 2D molecule structure images in place of nodes."""
        pos = self._compute_layout()
        fig, ax = plt.subplots(figsize=figsize)

        nx.draw_networkx_edges(
            self.graph,
            pos,
            ax=ax,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
            connectionstyle="arc3,rad=0.05",
            alpha=0.1,
        )

        draw_opts = Draw.MolDrawOptions()
        draw_opts.backgroundColour = (1.0, 0.0, 0.5, 0.0)

        for node, (x, y) in pos.items():
            molecule = self._molecule(node)
            img = Draw.MolToImage(molecule.mol, size=(200, 130), kekulize=True, options=draw_opts)
            imagebox = OffsetImage(img, zoom=node_size)
            annotation = AnnotationBbox(
                imagebox,
                (x, y),
                frameon=False,
                pad=0.6,
                bboxprops=dict(edgecolor="black", facecolor="blue", boxstyle="round,pad=0.2", alpha=0.0),
            )
            ax.add_artist(annotation)

        ax.margins(0.2)
        ax.set_axis_off()
        plt.tight_layout()
        if show:
            plt.show()

    def show(self):
        """Convenience: render with the default matplotlib backend."""
        self.to_matplotlib()


__all__ = ["GraphStyle", "SpaceView"]
