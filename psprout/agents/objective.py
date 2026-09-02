"""Scoring abstractions for agents: 
-`Metric` measures a molecule
-`Objective` combines metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rdkit import Chem

from ..space import Space


class Metric(ABC):
    """A named, reusable measurement of a molecule in the context of a `Space`.

    Operates directly on a `Chem.Mol` (rather than the node_id) so that
    candidate proposals can be scored before they are realized in the graph.
    """

    @abstractmethod
    def measure(self, space: Space, mol: Chem.Mol) -> float:
        raise NotImplementedError

    def __call__(self, space: Space, mol: Chem.Mol) -> float:
        return self.measure(space, mol)


class Objective:
    """A weighted combination of `Metric`s producing a single scalar score.

    Positive weights reward higher metric values, negative weights penalize them.
    """

    def __init__(self, terms: dict[str, tuple[Metric, float]]):
        self.terms = terms

    def score(self, space: Space, mol: Chem.Mol) -> float:
        return sum(weight * metric.measure(space, mol) for metric, weight in self.terms.values())

