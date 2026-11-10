"""Non-mutating proposal inspection agent.
Need this for when we want to non-destructively examine what an expansion would do.

Possible uses:
- debug mod libs
- inspect filters to examine what makes it through
- good for agent.reports
...
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Agent, AgentState, Proposal


@dataclass(frozen=True)
class InspectionProposal:
    """A lightweight inspection record for one filtered proposal."""
    parent_id: str
    modification: str
    product_smiles: str
    score: float | None = None


@dataclass(frozen=True)
class InspectionReport:
    """proposals inspected during one agent iteration."""
    iteration: int
    proposals: tuple[InspectionProposal, ...]


class InspectAgent(Agent):
    """inspect proposals without adding anything to the graph.

    Proposals pass through the normal frontier, generation, and filtering
    lifecycle. The filtered proposals are recorded in reports and none
    are selected for realization. It's non-destructive.
    """
    def __init__(self, *args, score_proposals: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_proposals = score_proposals
        self.reports: list[InspectionReport] = []

    @property
    def last_report(self) -> InspectionReport | None:
        """Return the most recent report, or None before the first step."""
        return self.reports[-1] if self.reports else None

    def select(
        self,
        proposals: list[Proposal],
        state: AgentState,
    ) -> list[Proposal]:
        records = tuple(
            InspectionProposal(
                parent_id=proposal.parent_id,
                modification=proposal.modification.name,
                product_smiles=proposal.product_smiles,
                score=self.score(proposal) if self.score_proposals else None,
            )
            for proposal in proposals
        )
        self.reports.append(InspectionReport(state.iteration, records))
        return []
