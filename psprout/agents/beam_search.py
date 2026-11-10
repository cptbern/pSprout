"""A beam-search agent: keeps only the top-scoring proposals each iteration."""

from __future__ import annotations

from .base import Agent, AgentState, Proposal


class BeamSearchAgent(Agent):
    """Keeps only the `beam_width` highest-`objective`-scoring proposals each
    iteration, bounding graph growth instead of keeping every proposal
    (`ExhaustiveExpansionAgent`) or a random sample (`RandomWalkAgent`).
    
    Objectives are passed via the base 'Agent' constructor:
    
    ```
    objective = Objective(
        terms={
            "similarity_to_support": (SimilarityToSupportMetric(), 1.0),
        }
    )

    agent = BeamSearchAgent(
        space=space,
        modifications=modifications,
        objective=objective,
        beam_width=5,
    )
    ```

    Requires an `objective` to rank proposals.
    """

    def __init__(self, *args, beam_width: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        if beam_width <= 0:
            raise ValueError("BeamSearchAgent requires beam_width > 0.")
        self.beam_width = beam_width

    def select(self, proposals: list[Proposal], state: AgentState) -> list[Proposal]:
        scored = [(self.score(proposal), proposal) for proposal in proposals]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [proposal for _, proposal in scored[: self.beam_width]]
