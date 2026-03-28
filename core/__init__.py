from importlib import import_module

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
    "pipeline",
]

_ATTACK_EXPORTS = {"BadNets", "Blended", "LabelConsistent"}
_DEFENSE_EXPORTS = {"REFINE", "REFINE_CG", "REFINE_SSL", "REFINE_PDB", "REFINE_PDB_SSL", "REFINE_ADAPTIVE"}


def __getattr__(name):
    if name == "models":
        value = import_module(".models", package=__name__)
        globals()[name] = value
        return value
    if name == "pipeline":
        value = import_module(".pipeline", package=__name__)
        globals()[name] = value
        return value
    if name in _ATTACK_EXPORTS:
        attacks = import_module(".attacks", package=__name__)
        value = getattr(attacks, name)
        globals()[name] = value
        return value
    if name in _DEFENSE_EXPORTS:
        defenses = import_module(".defenses", package=__name__)
        value = getattr(defenses, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'core' has no attribute '{name}'")
