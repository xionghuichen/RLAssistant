import functools
from typing import Callable
import yaml

import functools
import warnings
def deprecated_alias(**aliases):
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)
        return wrapper
    return deco

def rename_kwargs(func_name, kwargs, aliases):
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError('{} received both {} and {}'.format(
                    func_name, alias, new))
            warnings.warn('{} is deprecated; use {}'.format(alias, new),
                          DeprecationWarning,
                          3)
            kwargs[new] = kwargs.pop(alias)


def load_yaml(path):
    fs = open(path, encoding="UTF-8")
    try:
        private_config = yaml.load(fs)
    except TypeError:
        private_config = yaml.safe_load(fs)
    return private_config