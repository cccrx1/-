from importlib import import_module

__all__ = ["BadNets", "Blended", "LabelConsistent"]

_LAZY_IMPORTS = {
    "BadNets": ".BadNets",
    "Blended": ".Blended",
    "LabelConsistent": ".LabelConsistent",
}


def __getattr__(name):
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'core.attacks' has no attribute '{name}'")
    module = import_module(module_path, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
