"""Concrete `Metric` implementations."""

from __future__ import annotations

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors

from ..space import Space
from .objective import Metric

# Mirrors the descriptor set computed on `MoleculeNode`, but computed on demand
# so candidate molecules - which are not yet realized proposals - can be scored the same way.
_DESCRIPTOR_FUNCS = {
    "exact_mol_wt": Descriptors.ExactMolWt,
    "num_atoms": lambda mol: mol.GetNumAtoms(),
    "num_bonds": lambda mol: mol.GetNumBonds(),
    "num_rings": Descriptors.RingCount,
    "num_h_donors": Descriptors.NumHDonors,
    "num_h_acceptors": Descriptors.NumHAcceptors,
    "logp": Descriptors.MolLogP,
}


class SimilarityToSupportMetric(Metric):
    """Max Tanimoto similarity of a molecule to any support molecule."""

    def measure(self, space: Space, mol: Chem.Mol) -> float:
        fp = Chem.RDKFingerprint(mol)
        support_fps = [Chem.RDKFingerprint(support_mol) for support_mol in space.support_mols]
        if not support_fps:
            return 0.0
        return max(DataStructs.FingerprintSimilarity(fp, support_fp) for support_fp in support_fps)


class AveragePairwiseSimilarityMetric(Metric):
    """Average Tanimoto similarity across all molecule pairs currently in the graph.

    A global metric: `mol` is accepted for interface consistency but ignored.
    """

    def measure(self, space: Space, mol: Chem.Mol | None = None) -> float:
        molecules = [
            data["molecule"].mol
            for _, data in space.graph.nodes(data=True)
            if "molecule" in data
        ]
        if len(molecules) < 2:
            return 1.0

        fingerprints = [Chem.RDKFingerprint(molecule) for molecule in molecules]
        similarities = [
            DataStructs.FingerprintSimilarity(first, second)
            for index, first in enumerate(fingerprints)
            for second in fingerprints[index + 1 :]
        ]
        return sum(similarities) / len(similarities)


class DescriptorTargetMetric(Metric):
    """Negative absolute distance of a descriptor value from a target (closer to 0 is better)."""

    def __init__(self, descriptor: str, target: float):
        self.descriptor = descriptor
        self.target = target

    def measure(self, space: Space, mol: Chem.Mol) -> float:
        value = _DESCRIPTOR_FUNCS[self.descriptor](mol)
        return -abs(value - self.target)


class LipinskiRuleOfFiveMetric(Metric):
    """Return the negative number of Lipinski Rule-of-Five violations."""
    def measure(self, space: Space, # ignored
                mol: Chem.Mol) -> float:
        violations = sum(
            (
                Descriptors.MolWt(mol) > 500,
                Descriptors.MolLogP(mol) > 5,
                Descriptors.NumHDonors(mol) > 5,
                Descriptors.NumHAcceptors(mol) > 10,
            )
        )
        return -float(violations)


class NoveltyMetric(Metric):
    """Distance from the nearest already-realized molecule, using the maximum similarity."""
    def measure(self, space: Space, mol: Chem.Mol) -> float:
        fp = Chem.RDKFingerprint(mol)
        other_fps = [
            Chem.RDKFingerprint(space.graph.nodes[other]["molecule"].mol)
            for other in space.graph.nodes
            if "molecule" in space.graph.nodes[other]
        ]
        if not other_fps:
            return 1.0  # trivial case: no other molecules to compare against.
        max_similarity = max(
            DataStructs.FingerprintSimilarity(fp, other_fp) for other_fp in other_fps
        )
        return 1.0 - max_similarity

