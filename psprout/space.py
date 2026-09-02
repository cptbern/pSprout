from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors

import pandas as pd
import networkx as nx

from .modifications.modifications import Motif, Modification, AdditionModification

##
# setting up directed acyclic graph to keep track of modifications
#
# MoleculeNode -> ModificationEdge -> MoleculeNode
#
# identification of nodes via cononical SMILES

class MoleculeNode(dict):
    """A molecule in the modification DAG.
    Contains the canonical SMILES as unique identifier, RDKit molecule object, and metadata.
    Select descriptors are also computed and stored in the `descriptors` attribute.
    """
    id: str
    mol: Chem.Mol
    metadata: dict = field(default_factory=dict)

    def __init__(self, smiles: str, metadata: dict | None = None):
        self.mol = Chem.MolFromSmiles(smiles)
        self.id = Chem.MolToSmiles(self.mol, canonical=True)
        self.metadata = metadata or {}
        self.descriptors = {
            "exact_mol_wt": Descriptors.ExactMolWt(self.mol),
            "num_atoms": self.mol.GetNumAtoms(),
            "num_bonds": self.mol.GetNumBonds(),
            "num_rings": Descriptors.RingCount(self.mol),
            "num_h_donors": Descriptors.NumHDonors(self.mol),
            "num_h_acceptors": Descriptors.NumHAcceptors(self.mol),
            "logp": Descriptors.MolLogP(self.mol),
        }
        super().__init__(self.to_json_dict())

    def to_json_dict(self) -> dict:
        return {
            "id": self.id,
            "smiles": self.id,
            "name": self.metadata.get("name"),
            "descriptors": self.descriptors,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "MoleculeNode":
        meta = dict(data.get("metadata") or {})
        if data.get("name") is not None and "name" not in meta:
            meta["name"] = data["name"]
        node = cls(data["smiles"], metadata=meta)
        if data.get("descriptors") is not None:
            node.descriptors = data["descriptors"]
        return node

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key in {"id", "metadata", "descriptors"}:
            try:
                dict.__setitem__(self, key, value)
            except Exception:
                pass


class ModificationEdge(dict):
    """A modification in the modification graph."""
    name: str
    description: str | None = None
    metadata: dict = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        description: str | None = None,
        metadata: dict | None = None,
    ):
        self.name = name
        self.description = description
        self.metadata = metadata or {}
        self.parent_atom_idx = None
        super().__init__(self.to_json_dict())

    def to_json_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "parent_atom_idx": self.parent_atom_idx,
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "ModificationEdge":
        meta = dict(data.get("metadata") or {})
        return cls(
            data["name"],
            description=data.get("description"),
            metadata=meta,
        )

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key in {"name", "description", "metadata", "parent_atom_idx"}:
            try:
                dict.__setitem__(self, key, value)
            except Exception:
                pass

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

        self.g = nx.DiGraph() # the DAG representing the reaction network
        
        for mol in self.support_mols:
            node = MoleculeNode(Chem.MolToSmiles(mol), metadata={"type": "support"})
            self.g.add_node(node.id, molecule=node)

    @property
    def support_molecules(self) -> list[MoleculeNode]:
        """Return the support molecules used to initialize this space."""
        return [
            data["molecule"]
            for _, data in self.g.nodes(data=True)
            if data.get("molecule") and data["molecule"].metadata.get("type") == "support"
        ]

    def get_support_molecules(self) -> list[MoleculeNode]:
        """Return the list of support molecules."""
        return self.support_molecules
        
    def get_graph(self) -> nx.DiGraph:
        """Return the current reaction network graph."""
        return self.g

    def view(self, style=None, layout="spring"):
        """Return a styled, renderable `SpaceView` of the current graph.

        Args:
            style: a `GraphStyle` (e.g. `GraphStyle.dark()`), defaults to `GraphStyle.default()`
            layout: "spring", "tree", or a callable(graph) -> pos dict
        """
        from .visualization import GraphStyle, SpaceView

        return SpaceView(self.g, style=style or GraphStyle.default(), layout=layout)

    def list_smiles(self):
        """Convenience function: list all SMILES in the reaction network."""
        smiles_list = []
        for node in self.g.nodes:
            molecule = self.g.nodes[node].get("molecule")
            if molecule:
                smiles_list.append(molecule.id)
        return smiles_list
    
    
    def calculate_similarity(self) -> float:
        """Calculate the average similarity (currently only using Tanimoto) between all pairs of molecules in the graph."""
        molecules = [self.g.nodes[node]["molecule"].mol for node in self.g.nodes if "molecule" in self.g.nodes[node]]
        if len(molecules) < 2:
            return 1.0  # trivial case: less than two molecules.
            
        fps = [Chem.RDKFingerprint(mol) for mol in molecules]
        
        similarities = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.FingerprintSimilarity(fps[i], fps[j])
                similarities.append(sim)
        
        return sum(similarities) / len(similarities)

    def prune_to_support_paths(self) -> None:
        """Keep only support nodes and all nodes/edges that form paths connecting two support nodes.
        
        This method modifies the graph in-place, removing all nodes and edges that are not:
        1. Support molecules, or
        2. Part of a path connecting two support molecules
        """
        # Identify all support molecule nodes
        support_nodes = set()
        for node in self.g.nodes:
            molecule = self.g.nodes[node].get("molecule")
            if molecule and molecule.metadata.get("type") == "support":
                support_nodes.add(node)
        
        if len(support_nodes) < 2:
            # If there are fewer than 2 support nodes, keep only the support nodes
            nodes_to_keep = support_nodes
        else:
            # Modification edges only flow outward from supports, so two supports are
            # only "connected" if their subtrees merge on a shared descendant node.
            # Undirected connectivity captures that merge; directed edges are kept as-is.
            undirected = self.g.to_undirected(as_view=True)
            nodes_to_keep = set(support_nodes)
            
            for source in support_nodes:
                for target in support_nodes:
                    if source != target:
                        for path in nx.all_simple_paths(undirected, source, target):
                            nodes_to_keep.update(path)
        
        nodes_to_remove = set(self.g.nodes) - nodes_to_keep
        self.g.remove_nodes_from(nodes_to_remove)

    def prune_to_shortest_support_paths(self) -> None:
        """Keep only support nodes and nodes/edges on a shortest path between two support nodes.
        
        Like `prune_to_support_paths`, but for each pair of support molecules only the
        shortest connecting path(s) are retained, rather than every simple path.
        """

        support_nodes = set()
        for node in self.g.nodes:
            molecule = self.g.nodes[node].get("molecule")
            if molecule and molecule.metadata.get("type") == "support":
                support_nodes.add(node)
        
        if len(support_nodes) < 2:
            nodes_to_keep = support_nodes
        else:
            undirected = self.g.to_undirected(as_view=True)
            nodes_to_keep = set(support_nodes)
            
            for source in support_nodes:
                for target in support_nodes:
                    if source != target:
                        try:
                            for path in nx.all_shortest_paths(undirected, source, target):
                                nodes_to_keep.update(path)
                        except nx.NetworkXNoPath:
                            continue
        
        nodes_to_remove = set(self.g.nodes) - nodes_to_keep
        self.g.remove_nodes_from(nodes_to_remove)
        
    def run_agent(self, agent_type, modifications, max_iterations=2, **kwargs) -> list:
        """Convenience function to send an agent into Space. Returns the space, the agent, and a DataFrame containing the history of the agent's execution. Helpful for quickly running an agent and analyzing its performance and impact.

        Args:
            agent_type (_type_): the Agent class to run
            modifications (_type_): the toolkit (list of modifications) to use
            max_iterations (int, optional): number of iterations. Defaults to 2.

        Returns:
            list: A list containing the space, the agent, and a DataFrame containing the history of the agent's execution.
        """
        space = self
        agent = agent_type(space, modifications, **kwargs)
        history = []
        for state in agent.run(max_iterations=max_iterations):
            history.append(
                {
                    "iteration": state.iteration,
                    "frontier_size": len(state.frontier),
                    "visited_size": len(state.visited),
                    "node_count": self.get_graph().number_of_nodes(),
                    "edge_count": self.get_graph().number_of_edges(),
                }
            )
        return [pd.DataFrame(history), agent]

__all__ = ["Motif", "Space", "MoleculeNode", "ModificationEdge", "Modification", "AdditionModification"]
