"""A random-walk agent: samples a small number of proposals per frontier node.

"Not all who wander are lost", J.R.R. Tolkien
"""

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
        if k <= 0:
            raise ValueError("RandomWalkAgent requires k > 0.")
        if temperature is not None and temperature <= 0:
            raise ValueError("RandomWalkAgent temperature must be positive when set.")
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
        if not k:
            return []

        if self.objective is not None and self.temperature is not None:
            remaining = list(range(len(group)))
            weights = [math.exp(self.score(proposal) / self.temperature) for proposal in group]
            selected: list[Proposal] = []

            while len(selected) < k and remaining:
                total = sum(weights[index] for index in remaining)
                threshold = state.rng.random() * total
                cumulative = 0.0
                for index in remaining:
                    cumulative += weights[index]
                    if threshold <= cumulative:
                        selected.append(group[index])
                        remaining.remove(index)
                        break
                else:
                    last = remaining.pop()
                    selected.append(group[last])
            return selected

        return state.rng.sample(group, k)
