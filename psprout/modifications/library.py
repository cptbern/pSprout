"""Named, loadable collections of modifications."""

from __future__ import annotations

from pathlib import Path

from .modifications import Modification, ModificationType, ReactionSMARTSModification

_PACKAGE_DIR = Path(__file__).parent

# Bundled libraries, keyed by short name -> filename within this package.
_BUNDLED_LIBRARIES = {
    "simple": "modifications_simple.yaml",
    "medchem": "modifications_medchem.yaml",
}


class ModificationLibrary:
    """A named, iterable collection of modifications loaded from YAML.

    Supports dict-like lookup (`library["AddHydroxyl"]`), attribute access
    (`library.AddHydroxyl`), iteration, filtering by type, and combining
    libraries with `|`.
    """

    def __init__(self, modifications: dict[str, Modification]):
        self._modifications = dict(modifications)

    @classmethod
    def load(cls, name: str) -> "ModificationLibrary":
        """Load one of the libraries bundled with pSprout (e.g. "simple", "library")."""
        try:
            filename = _BUNDLED_LIBRARIES[name]
        except KeyError:
            raise ValueError(
                f"Unknown bundled library '{name}', expected one of {list(_BUNDLED_LIBRARIES)}"
            )
        return cls.load_yaml(_PACKAGE_DIR / filename)

    @classmethod
    def load_yaml(cls, yaml_file: str | Path) -> "ModificationLibrary":
        """Load modifications from any YAML."""
        modifications = ReactionSMARTSModification.load_modifications_from_yaml(str(yaml_file))
        return cls({modification.symbol: modification for modification in modifications})

    def of_type(self, modification_type: ModificationType) -> "ModificationLibrary":
        """Return a new library containing only modifications of the given type."""
        return ModificationLibrary(
            {symbol: mod for symbol, mod in self._modifications.items() if mod.type == modification_type}
        )

    def as_dict(self) -> dict[str, Modification]:
        """Return the underlying {symbol: modification} mapping."""
        return dict(self._modifications)

    def __getitem__(self, key: str | tuple[str, ...]):
        if isinstance(key, tuple):
            return tuple(self._modifications[k] for k in key)
        return self._modifications[key]

    def __getattr__(self, name: str) -> Modification:
        try:
            return self._modifications[name]
        except KeyError:
            raise AttributeError(name)

    def __iter__(self):
        return iter(self._modifications.values())

    def __len__(self) -> int:
        return len(self._modifications)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._modifications

    def __or__(self, other: "ModificationLibrary") -> "ModificationLibrary":
        return ModificationLibrary({**self._modifications, **other._modifications})

    def __repr__(self) -> str:
        return f"ModificationLibrary({list(self._modifications)!r})"


__all__ = ["ModificationLibrary"]
