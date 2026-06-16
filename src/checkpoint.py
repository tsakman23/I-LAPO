"""Lightweight checkpointing for the LAOM / I-LAPO pipeline.

Each training stage produces a self-describing checkpoint: the model
``state_dict`` together with the class name and the exact constructor kwargs
used to build it. This lets downstream consumers (e.g. SAC fine-tuning, or
re-running stages 2/3 without repeating the ~14h stage-1 LAM training) rebuild
the module from disk without re-deriving dataset dimensions.

Construction sites attach their kwargs to the module as ``model._build_kwargs``
so saving stays DRY with the code that actually builds the model.
"""

import json
import os
from dataclasses import asdict, is_dataclass

import torch

from src.nn import (
    ActionDecoder,
    Actor,
    IResNetDecoder,
    LAOMWithLabels,
    LAOMWithLabelsInvertible,
    StandardizedActionDecoder,
)

# Class name -> class, for rebuilding from a checkpoint.
_REGISTRY = {
    cls.__name__: cls
    for cls in (
        Actor,
        ActionDecoder,
        IResNetDecoder,
        LAOMWithLabels,
        LAOMWithLabelsInvertible,
        StandardizedActionDecoder,
    )
}


def save_model(path, model, build_kwargs=None, extra=None):
    """Save a model's state_dict plus enough metadata to rebuild it.

    Args:
        path: destination ``.pt`` file.
        model: an ``nn.Module``. If ``build_kwargs`` is None we read it from
            ``model._build_kwargs`` (attached at construction time).
        build_kwargs: constructor kwargs used to build ``model``.
        extra: optional dict of additional metadata to store.
    """
    if build_kwargs is None:
        build_kwargs = getattr(model, "_build_kwargs", None)
    if build_kwargs is None:
        raise ValueError(
            f"No build_kwargs for {type(model).__name__}; pass build_kwargs "
            "explicitly or attach model._build_kwargs at construction."
        )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_class": type(model).__name__,
        "build_kwargs": build_kwargs,
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_model(path, map_location=None, eval_mode=True):
    """Rebuild a model from a checkpoint written by ``save_model``.

    Returns ``(model, payload)`` where ``payload`` carries the metadata.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model_class = payload["model_class"]
    if model_class not in _REGISTRY:
        raise KeyError(
            f"Unknown model_class {model_class!r}; known: {sorted(_REGISTRY)}"
        )
    model = _REGISTRY[model_class](**payload["build_kwargs"])
    model.load_state_dict(payload["state_dict"])
    if map_location is not None:
        model = model.to(map_location)
    if eval_mode:
        model.eval()
    return model, payload


def save_run_config(checkpoint_dir, run_name, config):
    """Dump the full run config alongside the checkpoints for provenance."""
    run_dir = os.path.join(checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg = asdict(config) if is_dataclass(config) else config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    return run_dir


def stage_path(checkpoint_dir, run_name, filename):
    return os.path.join(checkpoint_dir, run_name, filename)
