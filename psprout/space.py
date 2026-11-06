from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any
import warnings

from rdkit import Chem
from rdkit.Chem import Descriptors

import networkx as nx

from .modifications.modifications import Motif, Modification, AdditionModification

if TYPE_CHECKING:
    from .visualization import GraphStyle, SpaceView

##
# setting up directed acyclic graph to keep track of modifications
#
# MoleculeNode -> ModificationEdge -> MoleculeNode
#
# identification of nodes via cononical SMILES

@dataclass
class MoleculeNode:
    """Store a molecule and its graph metadata."""

    smiles: InitVar[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(init=False)
    mol: Chem.Mol = field(init=False, repr=False)
    descriptors: dict[str, float | int] = field(init=False)

    def __post_init__(self, smiles: str) -> None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        self.mol = mol
        self.id = Chem.MolToSmiles(mol, canonical=True)
        self.descriptors = {
            "exact_mol_wt": Descriptors.ExactMolWt(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rings": Descriptors.RingCount(mol),
            "num_h_donors": Descriptors.NumHDonors(mol),
            "num_h_acceptors": Descriptors.NumHAcceptors(mol),
            "logp": Descriptors.MolLogP(mol),
        }

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the node."""
        return {
            "id": self.id,
            "smiles": self.id,
            "name": self.metadata.get("name"),
            "descriptors": self.descriptors,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "MoleculeNode":
        metadata = dict(data.get("metadata") or {})
        if data.get("name") is not None:
            metadata.setdefault("name", data["name"])

        node = cls(data["smiles"], metadata=metadata)
        if data.get("descriptors") is not None:
            node.descriptors = dict(data["descriptors"])
        return node

@dataclass
class ModificationEdge:
    """Store a modification represented by an edge in the graph."""

    name: str
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_atom_idx: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON compatible representation of the edge."""
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "parent_atom_idx": self.parent_atom_idx,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "ModificationEdge":
        return cls(
            name=data["name"],
            description=data.get("description"),
            metadata=dict(data.get("metadata") or {}),
            parent_atom_idx=data.get("parent_atom_idx"),
        )

class Space:
    """A chemical structure generator that can handle multiple origin molecules.
    """
    def __init__(
        self,
        support_molecules: Iterable[str],
    ):
        self.support_mols = [Chem.MolFromSmiles(smiles) for smiles in support_molecules]
        if any(mol is None for mol in self.support_mols):
            raise ValueError(f"Invalid support SMILES: {support_molecules}")

        self._graph = nx.DiGraph() # the DAG representing the reaction network
        
        for mol in self.support_mols:
            node = MoleculeNode(Chem.MolToSmiles(mol), metadata={"type": "support"})
            self._graph.add_node(node.id, molecule=node)

    @property
    def graph(self) -> nx.DiGraph:
        """Return the current reaction network graph."""
        return self._graph

    @property
    def g(self) -> nx.DiGraph:
        """Return the graph; use `graph` instead."""
        warnings.warn(
            "Space.g is deprecated; use Space.graph instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.graph

    @property
    def support_molecules(self) -> list[MoleculeNode]:
        """Return the support molecules used to initialize this space."""
        return [
            data["molecule"]
            for _, data in self.graph.nodes(data=True)
            if data.get("molecule") and data["molecule"].metadata.get("type") == "support"
        ]

    def get_support_molecules(self) -> list[MoleculeNode]:
        """Return support molecules; use `support_molecules` instead."""
        warnings.warn(
            "Space.get_support_molecules() is deprecated; use "
            "Space.support_molecules instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.support_molecules
        
    def get_graph(self) -> nx.DiGraph:
        """Return the graph; use `graph` instead."""
        warnings.warn(
            "Space.get_graph() is deprecated; use Space.graph instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.graph

    def view(
        self,
        style: GraphStyle | None = None,
        layout: str | Callable[[nx.DiGraph], dict] = "spring",
    ) -> SpaceView:
        """Return a styled, renderable `SpaceView` of the current graph.

        Args:
            style: a `GraphStyle` (e.g. `GraphStyle.dark()`), defaults to `GraphStyle.default()`
            layout: "spring", "tree", or a callable(graph) -> pos dict
        """
        from .visualization import GraphStyle, SpaceView

        return SpaceView(
            self.graph,
            style=style or GraphStyle.default(),
            layout=layout,
        )

    @property
    def smiles(self) -> list[str]:
        """Return all SMILES in the reaction network."""
        smiles_list = []
        for node in self.graph.nodes:
            molecule = self.graph.nodes[node].get("molecule")
            if molecule:
                smiles_list.append(molecule.id)
        return smiles_list

    def list_smiles(self) -> list[str]:
        """Return graph SMILES; use `smiles` instead."""
        warnings.warn(
            "Space.list_smiles() is deprecated; use Space.smiles instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.smiles

    def calculate_similarity(self) -> float:
        """Return average similarity; use its metric instead."""
        from .agents.metrics import AveragePairwiseSimilarityMetric

        warnings.warn(
            "Space.calculate_similarity() is deprecated; use "
            "AveragePairwiseSimilarityMetric.measure(space) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return AveragePairwiseSimilarityMetric().measure(self)

    def prune_to_support_paths(self) -> None:
        """Keep only support nodes and all nodes/edges that form paths connecting two support nodes.
        
        This method modifies the graph in-place, removing all nodes and edges that are not:
        1. Support molecules, or
        2. Part of a path connecting two support molecules
        """
        # Identify all support molecule nodes
        support_nodes = set()
        for node in self.graph.nodes:
            molecule = self.graph.nodes[node].get("molecule")
            if molecule and molecule.metadata.get("type") == "support":
                support_nodes.add(node)
        
        if len(support_nodes) < 2:
            # If there are fewer than 2 support nodes, keep only the support nodes
            nodes_to_keep = support_nodes
        else:
            # Modification edges only flow outward from supports, so two supports are
            # only "connected" if their subtrees merge on a shared descendant node.
            # Undirected connectivity captures that merge; directed edges are kept as-is.
            undirected = self.graph.to_undirected(as_view=True)
            nodes_to_keep = set(support_nodes)
            
            for source in support_nodes:
                for target in support_nodes:
                    if source != target:
                        for path in nx.all_simple_paths(undirected, source, target):
                            nodes_to_keep.update(path)
        
        nodes_to_remove = set(self.graph.nodes) - nodes_to_keep
        self.graph.remove_nodes_from(nodes_to_remove)

    def prune_to_shortest_support_paths(self) -> None:
        """Keep only support nodes and nodes/edges on a shortest path between two support nodes.
        
        Like `prune_to_support_paths`, but for each pair of support molecules only the
        shortest connecting path(s) are retained, rather than every simple path.
        """

        support_nodes = set()
        for node in self.graph.nodes:
            molecule = self.graph.nodes[node].get("molecule")
            if molecule and molecule.metadata.get("type") == "support":
                support_nodes.add(node)
        
        if len(support_nodes) < 2:
            nodes_to_keep = support_nodes
        else:
            undirected = self.graph.to_undirected(as_view=True)
            nodes_to_keep = set(support_nodes)
            
            for source in support_nodes:
                for target in support_nodes:
                    if source != target:
                        try:
                            for path in nx.all_shortest_paths(undirected, source, target):
                                nodes_to_keep.update(path)
                        except nx.NetworkXNoPath:
                            continue
        
        nodes_to_remove = set(self.graph.nodes) - nodes_to_keep
        self.graph.remove_nodes_from(nodes_to_remove)
        
    def run_agent(self, agent_type, modifications, max_iterations=2, **kwargs) -> list:
        """Run an agent and collect history; construct the agent directly instead.
        """
        import pandas as pd

        warnings.warn(
            "Space.run_agent() is deprecated; construct the agent and iterate "
            "over agent.run() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        agent = agent_type(self, modifications, **kwargs)
        history = []
        for state in agent.run(max_iterations=max_iterations):
            history.append(
                {
                    "iteration": state.iteration,
                    "frontier_size": len(state.frontier),
                    "visited_size": len(state.visited),
                    "node_count": self.graph.number_of_nodes(),
                    "edge_count": self.graph.number_of_edges(),
                }
            )
        return [pd.DataFrame(history), agent]

__all__ = ["Motif", "Space", "MoleculeNode", "ModificationEdge", "Modification", "AdditionModification"]
