"""Core public API for pSprout."""

from __future__ import annotations

from importlib import import_module
import warnings

from .modifications import Modification, ModificationLibrary
from .space import Space

__all__ = ["Space", "Modification", "ModificationLibrary"]

_DEPRECATED_EXPORTS = {
    "Motif": ("psprout.modifications", "Motif"),
    "AdditionModification": ("psprout.modifications", "AdditionModification"),
    "ReactionSMARTSModification": (
        "psprout.modifications",
        "ReactionSMARTSModification",
    ),
    "ModificationEdge": ("psprout.space", "ModificationEdge"),
    "MoleculeNode": ("psprout.space", "MoleculeNode"),
    "GraphStyle": ("psprout.visualization", "GraphStyle"),
    "SpaceView": ("psprout.visualization", "SpaceView"),
    "Agent": ("psprout.agents", "Agent"),
    "AgentState": ("psprout.agents", "AgentState"),
    "Proposal": ("psprout.agents", "Proposal"),
    "Metric": ("psprout.agents", "Metric"),
    "Objective": ("psprout.agents", "Objective"),
    "AveragePairwiseSimilarityMetric": (
        "psprout.agents",
        "AveragePairwiseSimilarityMetric",
    ),
    "DescriptorTargetMetric": ("psprout.agents", "DescriptorTargetMetric"),
    "NoveltyMetric": ("psprout.agents", "NoveltyMetric"),
    "SimilarityToSupportMetric": (
        "psprout.agents",
        "SimilarityToSupportMetric",
    ),
    "ExhaustiveExpansionAgent": (
        "psprout.agents",
        "ExhaustiveExpansionAgent",
    ),
    "RandomWalkAgent": ("psprout.agents", "RandomWalkAgent"),
    "BeamSearchAgent": ("psprout.agents", "BeamSearchAgent"),
    "PathBuildingAgent": ("psprout.agents", "PathBuildingAgent"),
}


def __getattr__(name: str):
    """Provide deprecated top-level imports during the API transition."""
    try:
        module_name, attribute_name = _DEPRECATED_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    warnings.warn(
        f"Importing {name} from psprout is deprecated; import it from "
        f"{module_name} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

