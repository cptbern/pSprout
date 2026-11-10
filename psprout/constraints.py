"""Reusable proposal constraints for agent workflows.

Separate hard contraints from search policy, e.g.:
- reject MW > 500 / respect a specfic descriptor range
- reject certain SMARTS
- reject already viited mols

NOT ACTIVE BY DEFAULT (agent defaults to filters=())
to use, for instance:
```
agent = BeamSearchAgent(
    space,
    modifications,
    filters=[
        DescriptorRangeFilter("exact_mol_wt", maximum=500),
    ],
)
```

#TODO: look into computational cost. it's probably O(P*F) for P proposals and F filters?
- SMARTS: expensive
- 

Maybe: filter as early as chemically possible, 
order cheap filters before SMARTS filters,
cache descriptors, deduplicate before running the expensive stuff
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rdkit import Chem
from rdkit.Chem import Descriptors

from .agents.base import Proposal
from .space import Space


@dataclass(frozen=True)
class FilterDecision:
    """result of applying a proposal filter."""
    accepted: bool
    reason: str | None = None


class ProposalFilter(Protocol):
    """Protocol for reusable proposal-level constraints."""
    def evaluate(self, space: Space, proposal: Proposal) -> FilterDecision:
        """Accept or reject a proposal before policy selection."""
        ...


#TODO: a bit specific ...
_DESCRIPTOR_FUNCTIONS = {
    "exact_mol_wt": Descriptors.ExactMolWt,
    "num_atoms": lambda mol: mol.GetNumAtoms(),
    "num_bonds": lambda mol: mol.GetNumBonds(),
    "num_rings": Descriptors.RingCount,
    "num_h_donors": Descriptors.NumHDonors,
    "num_h_acceptors": Descriptors.NumHAcceptors,
    "logp": Descriptors.MolLogP,
}


@dataclass(frozen=True)
class DescriptorRangeFilter:
    """Accept proposals whose descriptor value lies within the given bounds."""

    descriptor: str
    minimum: float | None = None
    maximum: float | None = None

    #def __post_init__(self) -> None:
        # if self.descriptor not in _DESCRIPTOR_FUNCTIONS:
        #     raise ValueError(
        #         f"Unknown descriptor {self.descriptor!r}; expected one of "
        #         f"{list(_DESCRIPTOR_FUNCTIONS)}"
        #     )
        # if self.minimum is None and self.maximum is None:
        #     raise ValueError("DescriptorRangeFilter requires a minimum or maximum.")
        # if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
        #     raise ValueError("DescriptorRangeFilter minimum cannot exceed maximum.")

    def evaluate(self, space: Space, proposal: Proposal) -> FilterDecision:
        value = _DESCRIPTOR_FUNCTIONS[self.descriptor](proposal.product_mol)
        if self.minimum is not None and value < self.minimum:
            return FilterDecision(False, f"{self.descriptor} is below minimum {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            return FilterDecision(False, f"{self.descriptor} is above maximum {self.maximum}")
        return FilterDecision(True)


class ExcludedSmartsFilter:
    """Reject proposals whose product contains a SMARTS substructure."""

    def __init__(self, smarts: str):
        self.smarts = smarts
        self.pattern = Chem.MolFromSmarts(smarts)
        if self.pattern is None:
            raise ValueError(f"Invalid SMARTS pattern: {smarts}")

    def evaluate(self, space: Space, proposal: Proposal) -> FilterDecision:
        if proposal.product_mol.HasSubstructMatch(self.pattern):
            return FilterDecision(False, f"product matches excluded SMARTS {self.smarts!r}")
        return FilterDecision(True)
