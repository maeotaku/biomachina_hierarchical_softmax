import sys
from collections import defaultdict

_module_to_models = defaultdict(set)  # dict of sets to check membership of model in module
_model_to_module = {}  # mapping of model names to module names
_model_entrypoints = {}  # mapping of model names to entrypoint fns


def register(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add model to __all__ in module
    model_name = fn.__name__
    mod.__all__ = [model_name]

    # add entries to registry dict/sets
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    return fn


def model_entrypoint(model_name):
    """Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]


def is_model(model_name):
    """ Check if a model name exists
    """
    return model_name in _model_entrypoints
