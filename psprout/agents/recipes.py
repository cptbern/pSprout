"""Reusable objective recipes for agent policies.
It's a factory system: they return ordinary Objective()s so they're re-usable in Agents.

Usage, e.g:

agent = BeamSearchAgent(
    space,
    modifications,
    objective=diversity_objective(),
)
"""

from __future__ import annotations

from .metrics import (
    DescriptorTargetMetric,
    LipinskiRuleOfFiveMetric,
    NoveltyMetric,
    SimilarityToSupportMetric,
)
from .objective import Objective


def diversity_objective(weight: float = 1.0) -> Objective:
    """Return an objective that rewards distance from existing graph molecules."""
    return Objective(
        terms={
            "novelty": (NoveltyMetric(), weight),
        }
    )


def support_similarity_objective(weight: float = 1.0) -> Objective:
    """Return an objective that rewards similarity to any support molecule."""
    return Objective(
        terms={
            "support_similarity": (SimilarityToSupportMetric(), weight),
        }
    )


def descriptor_target_objective(
    descriptor: str,
    target: float,
    weight: float = 1.0,
) -> Objective:
    """Return an objective that rewards a descriptor value near a target."""
    return Objective(
        terms={
            descriptor: (DescriptorTargetMetric(descriptor, target), weight),
        }
    )


def lipinski_objective(weight: float = 1.0) -> Objective:
    """Return an objective that rewards compliance with Lipinski's rules."""
    return Objective(
        terms={
            "lipinski_rule_of_five": (LipinskiRuleOfFiveMetric(), weight),
        }
    )


def drug_like_objective(weight: float = 1.0) -> Objective:
    """Return a soft objective for a typical small-molecule descriptor profile."""
    return Objective(
        terms={
            "drug_like_mol_wt": (
                DescriptorTargetMetric("exact_mol_wt", 350.0),
                0.002 * weight,
            ),
            "drug_like_logp": (
                DescriptorTargetMetric("logp", 2.5),
                0.5 * weight,
            ),
            "drug_like_hbd": (
                DescriptorTargetMetric("num_h_donors", 1.0),
                0.25 * weight,
            ),
            "drug_like_hba": (
                DescriptorTargetMetric("num_h_acceptors", 4.0),
                0.25 * weight,
            ),
        }
    )

# currently, `OBjective` combines metrics as weighted sums:
# obj = Objective( terms={ "metric1": (Metric1(), 1.0),
#                           "metric2": (Metric2(), 0.5)})
#
# so metric1*1 + metric2 *0.5 = total
#
# below is a convenience function to combine Objectives directly. might not need it.

def combine_objectives(*objectives: Objective) -> Objective:
    """Combine named weighted terms from several objectives."""
    terms = {}
    for objective in objectives:
        for name, term in objective.terms.items():
            if name in terms:
                raise ValueError(f"Duplicate objective term: {name!r}")
            terms[name] = term
    return Objective(terms)
