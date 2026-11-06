"""Simulated-annealing traversal for bounded, uninterrupted runs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Callable, Iterable
import math

from .base import Agent, AgentState, Proposal


@dataclass(frozen=True)
class ExponentialTemperatureSchedule:
    """Temperature schedule with a lower bound.
    
    args:
    - decay: (0, 1]
    """

    #TODO: are these appropriate?
    initial: float = 1.0
    decay: float = 0.95
    minimum: float = 1e-6

    def __post_init__(self) -> None:
        if not 0 < self.decay <= 1:
            raise ValueError("temp decay must be in (0, 1].")

    def __call__(self, iteration: int) -> float:
        return max(self.minimum, self.initial * self.decay**iteration)


class SimulatedAnnealingAgent(Agent):
    """Explore one stochastic trajectory per active frontier node.

    A proposal that improves the objective is always accepted. A worse
    proposal is accepted with probability ``exp(delta / temperature)``. Runs
    are intentionally bounded and are not resumable in this first version.
    """

    def __init__(
        self,
        *args,
        temperature_schedule: Callable[[int], float] | None = None,
        initial_temperature: float = 1.0,
        cooling: float = 0.95,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.objective is None:
            raise ValueError("SimulatedAnnealingAgent requires an objective.")
        self.temperature_schedule = temperature_schedule or ExponentialTemperatureSchedule(
            initial=initial_temperature,
            decay=cooling,
        )

    def generate_proposals(
        self,
        frontier: Iterable[str],
    ) -> list[Proposal]:
        """Generate proposals in stable node order for seeded runs."""
        proposals: list[Proposal] = []
        for node_id in sorted(frontier):
            proposals.extend(self.propose(node_id))
        return proposals

    def select(
        self,
        proposals: list[Proposal],
        state: AgentState,
    ) -> list[Proposal]:
        """Choose one random candidate per active trajectory."""
        by_parent: dict[str, list[Proposal]] = defaultdict(list)
        for proposal in proposals:
            by_parent[proposal.parent_id].append(proposal)
        return [
            state.rng.choice(by_parent[parent_id])
            for parent_id in sorted(by_parent)
        ]

    def advance(self, state: AgentState) -> AgentState:
        proposals = self.generate_proposals(state.frontier)
        proposals = self.filter_proposals(proposals, state)
        candidates = self.select(proposals, state)
        candidates_by_parent = {proposal.parent_id: proposal for proposal in candidates}
        temperature = self.temperature_schedule(state.iteration)
        accepted: list[Proposal] = []
        next_frontier: set[str] = set()

        for parent_id in sorted(state.frontier):
            proposal = candidates_by_parent.get(parent_id)
            if proposal is None:
                continue

            current_mol = self.space.graph.nodes[parent_id]["molecule"].mol
            current_score = self.objective.score(self.space, current_mol)
            proposal_score = self.score(proposal)
            delta = proposal_score - current_score
            acceptance_probability = 1.0 if delta >= 0 else math.exp(delta / temperature)

            if delta >= 0 or state.rng.random() < acceptance_probability:
                accepted.append(proposal)
                next_frontier.add(proposal.product_smiles)
            else:
                next_frontier.add(parent_id)

        self.realize(accepted)
        next_state = AgentState(
            iteration=state.iteration + 1,
            frontier=next_frontier,
            visited=state.visited | next_frontier,
            rng=state.rng,
        )
        self.report(next_state, accepted)
        return next_state

    def run(self, max_iterations: int | None = None):
        """Run a bounded trajectory; resumability is intentionally unsupported."""
        if max_iterations is None:
            raise ValueError("SimulatedAnnealingAgent requires max_iterations.")
        return super().run(max_iterations=max_iterations)
