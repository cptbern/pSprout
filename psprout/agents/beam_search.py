"""A beam-search agent: keeps only the top-scoring proposals each iteration."""

from __future__ import annotations

from .base import Agent, AgentState, Proposal


class BeamSearchAgent(Agent):
    """Keeps only the `beam_width` highest-`objective`-scoring proposals each
    iteration, bounding graph growth instead of keeping every proposal
    (`ExhaustiveExpansionAgent`) or a random sample (`RandomWalkAgent`).

    Requires an `objective` to rank proposals.
    """

    def __init__(self, *args, beam_width: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam_width = beam_width

    def select(self, proposals: list[Proposal], state: AgentState) -> list[Proposal]:
        scored = [(self.score(proposal), proposal) for proposal in proposals]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [proposal for _, proposal in scored[: self.beam_width]]
