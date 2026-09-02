"""A random-walk agent: samples a small number of proposals per frontier node."""

from __future__ import annotations

import math
from collections import defaultdict

from .base import Agent, AgentState, Proposal


class RandomWalkAgent(Agent):
    """Samples `k` proposals per frontier node each iteration, instead of
    keeping all of them.

    If `objective` is set and `temperature` is not None, sampling is weighted
    by a softmax over proposal scores (higher temperature = more uniform,
    lower temperature = more greedy). Otherwise sampling is uniform.
    """

    def __init__(
        self,
        *args,
        k: int = 1,
        temperature: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.temperature = temperature

    def select(self, proposals: list[Proposal], state: AgentState) -> list[Proposal]:
        by_parent: dict[str, list[Proposal]] = defaultdict(list)
        for proposal in proposals:
            by_parent[proposal.parent_id].append(proposal)

        chosen: list[Proposal] = []
        for group in by_parent.values():
            chosen.extend(self._sample(group, state))
        return chosen

    def _sample(self, group: list[Proposal], state: AgentState) -> list[Proposal]:
        k = min(self.k, len(group))
        if self.objective is not None and self.temperature is not None:
            weights = [math.exp(self.score(proposal) / self.temperature) for proposal in group]
            total = sum(weights)
            probabilities = [w / total for w in weights]
            return state.rng.choices(group, weights=probabilities, k=k) if k else []
        return state.rng.sample(group, k)
