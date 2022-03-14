"""Register pytorch classes and functions with gin."""
import gin
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import inspect

optimizers = [obj for name, obj in inspect.getmembers(optim)
              if inspect.isclass(obj)]
for o in optimizers:
    gin.config.external_configurable(o, module='optim')

modules = [obj for name, obj in inspect.getmembers(nn) if inspect.isclass(obj)]
for m in modules:
    gin.config.external_configurable(m, module='nn')

funcs = [f for name, f in inspect.getmembers(F) if inspect.isfunction(f)]
for f in funcs:
    try:
        gin.config.external_configurable(f, module='F')
    except Exception:
        pass

funcs = [f for name, f in inspect.getmembers(torchvision.models)
         if inspect.isfunction(f)]
for f in funcs:
    try:
        gin.config.external_configurable(f, module='models')
    except Exception:
        pass

funcs = [f for name, f in inspect.getmembers(torchvision.models.segmentation)
         if inspect.isfunction(f)]
for f in funcs:
    try:
        gin.config.external_configurable(f, module='models.segmentation')
    except Exception:
        pass

funcs = [f for name, f in inspect.getmembers(torchvision.models.detection)
         if inspect.isfunction(f)]
for f in funcs:
    try:
        gin.config.external_configurable(f, module='models.detection')
    except Exception:
        pass

transforms = [obj for name, obj in inspect.getmembers(torchvision.transforms)
              if inspect.isclass(obj)]
for t in transforms:
    try:
        gin.config.external_configurable(t, module='transforms')
    except Exception:
        pass

datasets = [obj for name, obj in inspect.getmembers(torchvision.datasets)
            if inspect.isclass(obj)]
for d in datasets:
    try:
        gin.config.external_configurable(d, module='datasets')
    except Exception:
        pass


def load_config(gin_files, gin_bindings=[]):
    """Load gin configuration files.

    Args:
    gin_files: path or list of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.

    """
    if isinstance(gin_files, str):
        gin_files = [gin_files]
    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=gin_bindings,
                                        skip_unknown=False)
