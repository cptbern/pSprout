"""Greedy best-first traversal over an agent frontier.

OBjective -> score frontier -> expand highest-scoring node (retaining all others 
"""

from __future__ import annotations

from .base import Agent, AgentState, Proposal


class GreedyAgent(Agent):
    """Expand the highest-scoring open frontier molecule at each step.

    Unlike beam search, unexpanded frontier nodes are retained. The objective
    scores molecules already in the frontier; all proposals from the selected
    molecule are passed through the normal filter and selection lifecycle.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.objective is None:
            raise ValueError("GreedyAgent requires an objective to rank frontier molecules.")

    def select_frontier(self, state: AgentState) -> set[str]:
        """Choose one highest-scoring molecule from the open frontier."""
        if not state.frontier:
            return set()

        best_node = min(
            state.frontier,
            key=lambda node_id: (
                -self.objective.score(self.space, self.space.graph.nodes[node_id]["molecule"].mol),
                node_id,
            ),
        )
        return {best_node}

    def select(
        self,
        proposals: list[Proposal],
        state: AgentState,
    ) -> list[Proposal]:
        """Realize all valid proposals from the selected frontier molecule."""
        return proposals

    def advance(self, state: AgentState) -> AgentState:
        """Advance one best-first step while retaining open frontier nodes."""
        frontier = self.select_frontier(state)
        proposals = self.generate_proposals(frontier)
        proposals = self.filter_proposals(proposals, state)
        selected = self.select(proposals, state)
        realized = self.realize(selected)
        next_frontier = (state.frontier - frontier) | realized

        next_state = AgentState(
            iteration=state.iteration + 1,
            frontier=next_frontier,
            visited=state.visited | realized,
            rng=state.rng,
        )
        self.report(next_state, selected)
        return next_state


GreedyBestFirstAgent = GreedyAgent
