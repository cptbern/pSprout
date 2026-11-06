"""The exhaustive expansion agent: keeps every idea, mirroring the former Space.expand()."""

from __future__ import annotations

from .base import Agent, AgentState, Proposal


class ExhaustiveExpansionAgent(Agent):
    """Keeps all proposals every iteration, growing the graph breadth-first
    It's like the original `Space.expand()` implementation.
    Inherits the traversal behavior from 'Agent': select() receives the proposals
    along with the state (but ignores it) and returns all proposals unchanged since
    it does no ranking, selecting, etc.
    """

    def select(self, proposals: list[Proposal], 
               state: AgentState, # ignored
               ) -> list[Proposal]:
        return proposals
