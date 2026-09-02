"""An agent that greedily grows paths toward connecting support molecules."""

from __future__ import annotations

import networkx as nx

from .base import Agent, AgentState, Proposal
from .objective import Objective
from .metrics import SimilarityToSupportMetric


class PathBuildingAgent(Agent):
    """Greedy best-first search that grows the graph toward whichever support
    molecule each proposal is most similar to, stopping as soon as any two
    support molecules become connected.

    If no `objective` is given, defaults to maximizing `SimilarityToSupportMetric`
    (i.e. "pull toward the nearest support molecule"), which naturally favors
    proposals that drift from their origin support toward a different one.

    This is a heuristic greedy search, not a true shortest-path search: it
    keeps the top `beam_width` proposals *overall* per iteration (not per
    frontier node), so only the most promising branches are pursued.
    """

    def __init__(self, *args, beam_width: int = 3, **kwargs):
        objective = kwargs.pop(
            "objective",
            Objective(terms={"similarity_to_support": (SimilarityToSupportMetric(), 1.0)}),
        )
        super().__init__(*args, objective=objective, **kwargs)
        self.beam_width = beam_width

    def select(self, proposals: list[Proposal], state: AgentState) -> list[Proposal]:
        scored = [(self.score(proposal), proposal) for proposal in proposals]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [proposal for _, proposal in scored[: self.beam_width]]

    def is_exhausted(self, state: AgentState) -> bool:
        return not state.frontier or self._supports_connected()

    def _supports_connected(self) -> bool:
        support_ids = [node.id for node in self.space.get_support_molecules()]
        if len(support_ids) < 2:
            return False
        undirected = self.space.g.to_undirected(as_view=True)
        for i in range(len(support_ids)):
            for j in range(i + 1, len(support_ids)):
                if nx.has_path(undirected, support_ids[i], support_ids[j]):
                    return True
        return False


