"""The exhaustive expansion agent: keeps every idea, mirroring the former Space.expand()."""

from __future__ import annotations

from .base import Agent, AgentState, Proposal


class ExhaustiveExpansionAgent(Agent):
    """Keeps all proposals every iteration, growing the graph breadth-first
    It's like the original `Space.expand()` implementation."""

    def select(self, proposals: list[Proposal], state: AgentState) -> list[Proposal]:
        return proposals
