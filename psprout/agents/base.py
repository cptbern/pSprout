"""Base abstractions for graph-traversal agents over a `Space`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
import random

from rdkit import Chem

from ..modifications.modifications import Modification
from ..space import MoleculeNode, Space


@dataclass(frozen=True)
class Proposal:
    """A candidate expansion: applying `modification` to the molecule at
    `parent_id` produces `product_mol`, identified by its canonical `product_smiles`."""
    parent_id: str
    modification: Modification
    product_mol: Chem.Mol
    product_smiles: str


@dataclass
class AgentState:
    """Snapshot of an agent's traversal progress, yielded once per iteration."""
    iteration: int = 0
    frontier: set[str] = field(default_factory=set)
    visited: set[str] = field(default_factory=set)
    rng: random.Random = field(default_factory=random.Random)


class Agent(ABC):
    """Base class for graph-traversal policies over a `Space`.

    Subclasses implement `select()` to decide, given all proposals generated
    from the current frontier, which ones actually get realized in the graph
    and carried forward as the next frontier.
    """

    def __init__(
        self,
        space: Space,
        modifications: Iterable[Modification],
        objective: "Objective | None" = None,
        retain_all_edges: bool = False,
        verbose: bool = False,
        seed: int | None = None,
    ):
        self.space = space
        self.modifications = list(modifications)
        self.objective = objective
        self.retain_all_edges = retain_all_edges
        self.verbose = verbose
        self._rng = random.Random(seed)

    # -- candidate generation (shared by all agents) -------------------

    def propose(self, node_id: str) -> list[Proposal]:
        """Apply every modification to the molecule at `node_id` and return
        the resulting proposals. Does not touch the graph."""
        mol = self.space.g.nodes[node_id]["molecule"].mol
        proposals = []
        for modification in self.modifications:
            products = modification.apply(mol)
            if self.verbose:
                print(f"Generated {len(products)} products from modification: {modification.name}")
            for product_mol in products:
                product_smiles = Chem.MolToSmiles(product_mol, canonical=True)
                proposals.append(Proposal(node_id, modification, product_mol, product_smiles))
        return proposals

    def score(self, proposal: Proposal) -> float:
        """Score a proposal's product molecule via `self.objective`, without
        realizing it in the graph. Requires `objective` to be set."""
        if self.objective is None:
            raise ValueError(f"{type(self).__name__} requires an objective to score proposals.")
        return self.objective.score(self.space, proposal.product_mol)

    # -- policy hook: subclasses must implement -------------------------

    @abstractmethod
    def select(self, proposals: list[Proposal], state: AgentState) -> list[Proposal]:
        """Given all proposals generated from the current frontier, choose
        which ones to realize. This is the only method most subclasses need to override."""
        raise NotImplementedError

    # -- stopping condition: subclasses may override --------------------

    def is_exhausted(self, state: AgentState) -> bool:
        """Default: exhausted once the frontier dies out."""
        return not state.frontier

    # -- graph mutation (shared, mirrors the former Space.expand() body) --

    def realize(self, proposals: list[Proposal]) -> set[str]:
        """Add selected proposals to `self.space.g`, deduping by canonical
        SMILES. Returns the set of node IDs that should join the next frontier."""
        next_ids: set[str] = set()
        for proposal in proposals:
            if proposal.product_smiles not in self.space.g.nodes:
                node = MoleculeNode(proposal.product_smiles)
                self.space.g.add_node(node.id, molecule=node)
                self.space.g.add_edge(proposal.parent_id, node.id, modification=proposal.modification.name)
                next_ids.add(node.id)
            elif self.retain_all_edges:
                self.space.g.add_edge(proposal.parent_id, proposal.product_smiles, modification=proposal.modification.name)
        return next_ids

    # -- one iteration ---------------------------------------------------

    def advance(self, state: AgentState) -> AgentState:
        all_proposals: list[Proposal] = []
        for node_id in state.frontier:
            all_proposals.extend(self.propose(node_id))

        selected = self.select(all_proposals, state)
        next_frontier = self.realize(selected)

        return AgentState(
            iteration=state.iteration + 1,
            frontier=next_frontier,
            visited=state.visited | next_frontier,
            rng=state.rng,
        )

    # -- driver ------------------------------------------------------------

    def run(self, max_iterations: int | None = None):
        """Yield an `AgentState` after each iteration, until `is_exhausted()`
        fires or `max_iterations` is reached."""
        support_ids = {node.id for node in self.space.get_support_molecules()}
        state = AgentState(iteration=0, frontier=support_ids, visited=set(support_ids), rng=self._rng)

        count = 0
        while not self.is_exhausted(state) and (max_iterations is None or count < max_iterations):
            if self.verbose:
                print(f"[{type(self).__name__}] iteration {state.iteration}: frontier={len(state.frontier)}")
            state = self.advance(state)
            count += 1
            yield state
