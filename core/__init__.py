from . import attacks, defenses, models

__all__ = [
    "BadNets",
    "Blended",
    "LabelConsistent",
    "REFINE",
    "REFINE_CG",
    "REFINE_SSL",
    "REFINE_PDB",
    "REFINE_PDB_SSL",
    "REFINE_ADAPTIVE",
    "models",
]

_ATTACK_EXPORTS = {"BadNets", "Blended", "LabelConsistent"}
_DEFENSE_EXPORTS = {"REFINE", "REFINE_CG", "REFINE_SSL", "REFINE_PDB", "REFINE_PDB_SSL", "REFINE_ADAPTIVE"}


def __getattr__(name):
    if name in _ATTACK_EXPORTS:
        value = getattr(attacks, name)
        globals()[name] = value
        return value
    if name in _DEFENSE_EXPORTS:
        value = getattr(defenses, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'core' has no attribute '{name}'")
