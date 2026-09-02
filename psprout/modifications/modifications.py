"""Modification classes for pSprout."""

from __future__ import annotations
from operator import mod

from rdkit import Chem
from rdkit.Chem import rdChemReactions
from dataclasses import dataclass
from enum import StrEnum
import yaml

from psprout import modifications


@dataclass(frozen=True)
class Motif:
    """A chemical motif with an explicit attachment point ('*').

    Examples:
        [*:1]C(=O)O -> carboxy group
        [*:1]C -> methyl group
    """

    name: str
    smiles: str


class ModificationType(StrEnum):
    ADDITION = "addition"
    PURGE = "purge"
    SUBSTITUTION = "substitution"
    TRANSFORMATION = "transformation"
    

class Modification:
    """A chemical modification that can be applied to a molecule."""
    def __init__(
        self,
        name: str,
        type: ModificationType,
        description: str | None = None,
        metadata: dict | None = None,
    ):
        self.name = name
        self.type = type
        self.description = description
        self.metadata = metadata or {}
        
    def apply( self, mol: Chem.Mol ) -> [Chem.Mol]:
        pass
    
    @staticmethod
    def _sanitize_mol(mol: Chem.Mol) -> Chem.Mol | None:
        """Sanitize a molecule and return a new molecule. None otherwise."""
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Exception as e:
            print(f"Sanitization failed: {e}")
            return None

class ReactionSMARTSModification(Modification):
    """A modification that applies a reaction SMARTS to a molecule."""
    def __init__(
        self,
        symbol: str,
        name: str,
        smarts: str,
        description: str | None = None,
        metadata: dict | None = None,
    ):
        super().__init__(name, ModificationType.TRANSFORMATION, description, metadata)
        self.symbol = symbol
        self.smarts = smarts
        self.reaction = rdChemReactions.ReactionFromSmarts(smarts)

    def apply(self, mol: Chem.Mol) -> list[Chem.Mol]:
        """Apply the reaction SMARTS to a molecule and return unique products.
        
        - Adds explicit hydrogens to the input molecule for SMARTS matching
        - Removes explicit hydrogens from products for cleaner output
        - Deduplicates products by canonical SMILES to remove duplicates
        """
        # Add explicit hydrogens
        mol_with_h = Chem.AddHs(mol)
        products = self.reaction.RunReactants((mol_with_h,))
        
        # Track unique mols by canonical SMILES
        unique_products = {}
        
        for product_set in products:
            for product in product_set:
                sanitized_product = Modification._sanitize_mol(product)
                if sanitized_product is None:
                    continue
                
                # de-duplicate by canonical SMILES
                product_no_h = Chem.RemoveHs(sanitized_product)
                smiles = Chem.MolToSmiles(product_no_h, canonical=True)
                if smiles not in unique_products:
                    unique_products[smiles] = product_no_h
        
        return list(unique_products.values())
    
    @staticmethod
    def load_modifications_from_yaml(
        yaml_file: str, into: dict | None = None
    ) -> list[ReactionSMARTSModification]:
        """Parse a yaml file to obtain list of ReactionSMARTSModifications

        Args:
            yaml_file (str): the path to the yaml file containing the modifications
            into (dict | None): if provided, each modification is also written into this
                mapping under its `symbol` (e.g. pass `globals()` to make symbols like
                `AddHydroxyl` directly available in the caller's namespace)

        Returns:
            list[ReactionSMARTSModification]: a list of ReactionSMARTSModification objects
        """
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        modifications = []
        for mod in data['modifications']:
            # print(f"Loading modification: \"{mod['name']}\", SMARTS: {mod['reactionSMARTS']}")
            modification = ReactionSMARTSModification(
                symbol=mod['symbol'],
                name=mod['name'],
                smarts=mod['reactionSMARTS'],
                description=mod['description'],
                metadata=mod.get('metadata', {})
            )
            modifications.append(modification)

        if into is not None:
            into.update({modification.symbol: modification for modification in modifications})

        return modifications


class AdditionModification(Modification):
    """A modification that adds a motif to a molecule and produces a list of metabolites.
    """
    def __init__(
        self,
        name: str,
        motif: Motif | str,
        description: str | None = None,
        metadata: dict | None = None,
    ):
        super().__init__(name, ModificationType.ADDITION, description, metadata)
        if isinstance(motif, str):
            motif = Motif(name=name, smiles=motif)
        self.motif = motif

    @staticmethod
    def _remove_dummy_atom(mol: Chem.Mol) -> Chem.Mol:
        """Remove dummy atoms from a molecule after attachment."""
        dummy = next(atom for atom in mol.getAtoms() if atom.GetAtomicNum() == 0)

        rw = Chem.RWMol(mol)
        rw.RemoveAtom(dummy.GetIdx())

        result = rw.GetMol()
        Chem.SanitizeMol(result)

        return result
    
    @staticmethod
    def _add_fragment(
        origin: Chem.Mol,
        atom_idx: int,
        motif: str,
        ) -> Chem.Mol | None:
        """Attach a motif to the origin molecule at the specified atom index."""
        motif = Chem.MolFromSmiles(motif)

        dummy = next(atom for atom in motif.GetAtoms() if atom.GetAtomicNum() == 0)
        neighbors = list(dummy.GetNeighbors())

        if len(neighbors) != 1:
            raise ValueError("Fragment must have exactly one attachment point (dummy atom).")

        motif_attach_idx = neighbors[0].GetIdx()
        fused_mol = Chem.CombineMols(origin, motif)
        rw = Chem.RWMol(fused_mol)

        origin_size = origin.GetNumAtoms()
        motif_attach_idx += origin_size
        dummy_idx = dummy.GetIdx() + origin_size

        rw.RemoveAtom(dummy_idx)
        if motif_attach_idx > dummy_idx:
            motif_attach_idx -= 1

        try:
            rw.AddBond(
                atom_idx,
                motif_attach_idx,
                order=Chem.rdchem.BondType.SINGLE,
            )
        except Exception as e:
            print(f"Failed to add a single bond: {e}")
            return None

        result = rw.GetMol()
        
        return Modification._sanitize_mol(result)


    def _find_attachment_points(self, mol: Chem.Mol) -> list[int]:
        """Find all valid attachment points in the molecule."""
        points = []

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                continue

            if atom.GetTotalNumHs() > 0:
                points.append(atom.GetIdx())

        return points
    
    def apply(self, mol: Chem.Mol) -> list[Chem.Mol]:
        products = []
        seen = set()

        for atom_idx in self._find_attachment_points(mol):
            new_mol = self._add_fragment(mol, atom_idx, self.motif.smiles)

            if new_mol is None:
                continue

            smiles = Chem.MolToSmiles(new_mol, canonical=True)

            if smiles not in seen:
                seen.add(smiles)
                products.append(new_mol)

        return products
