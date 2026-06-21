"""Lightweight checkpointing for the LAOM / I-LAPO pipeline.

Each training stage produces a self-describing checkpoint: the model
`state_dict` together with the class name and the exact constructor kwargs
used to build it. This lets downstream consumers (e.g. SAC fine-tuning, or
re-running stages 2/3 without repeating the ~14h stage-1 LAM training) rebuild
the module from disk without re-deriving dataset dimensions.

Construction sites attach their kwargs to the module as `model._build_kwargs`
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
        path: destination `.pt` file.
        model: an `nn.Module`. If `build_kwargs` is None we read it from
            `model._build_kwargs` (attached at construction time).
        build_kwargs: constructor kwargs used to build `model`.
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
    """Rebuild a model from a checkpoint written by `save_model`.

    Returns `(model, payload)` where `payload` carries the metadata.
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


def save_run_config(run_dir, config):
    """Dump the full run config alongside the checkpoints for provenance."""
    os.makedirs(run_dir, exist_ok=True)
    cfg = asdict(config) if is_dataclass(config) else config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    return run_dir


def stage_path(run_dir, filename):
    """Path of a single stage checkpoint inside a run directory."""
    return os.path.join(run_dir, filename)


def dataset_tag(data_path):
    """Derive a dataset tag like `cheetah-dcs` / `walker-vanilla` from a data
    path. Used as the top level of the structured checkpoint tree so runs group
    by environment + benchmark suite.
    """
    p = (data_path or "").lower()
    env = next((e for e in ("cheetah", "hopper", "walker") if e in p), "unknown-env")
    if "dcs" in p:
        suite = "dcs"
    elif "vanilla" in p:
        suite = "vanilla"
    else:
        raise ValueError(f"Unknown benchmark suite in {data_path!r}; expected 'dcs' or 'vanilla' (DMControl).")
    return f"{env}-{suite}"


def run_checkpoint_dir(checkpoint_dir, dataset, model, seed, run_id, fd_coef=None):
    """Build the structured run directory where a run's stage checkpoints live:

        {checkpoint_dir}/{dataset}/{model}/seed-{seed}/[fd-coef-{fd_coef}/]{run_id}

    `fd_coef` is included only when not None (i-ResNet / I-LAPO runs); plain
    LAOM runs omit that level. `run_id` is typically the W&B run id so the leaf
    directory is unique per run.
    """
    parts = [checkpoint_dir, dataset, model, f"seed-{seed}"]
    if fd_coef is not None:
        parts.append(f"fd-coef-{fd_coef}")
    parts.append(str(run_id))
    return os.path.join(*parts)


# Canonical stage selections the pipeline supports. Anything else is a typo and
# should fail fast rather than silently skip training.
_ALLOWED_STAGES = {"1", "2", "3", "12", "23", "123"}


def parse_stages(stages):
    """Parse the `stages` config flag into a set of stage numbers.

    Accepts `1`, `2`, `3`, `12`, `23`, `123` (and the aliases
    `full` / `all` for the complete pipeline). Returns e.g. `{2, 3}`.
    """
    s = str(stages).strip().lower()
    if s in ("full", "all"):
        s = "123"
    if s not in _ALLOWED_STAGES:
        raise ValueError(
            f"Invalid stages={stages!r}. Allowed: 1, 2, 3, 12, 23, 123 "
            "(or 'full' / 'all' for the complete pipeline)."
        )
    return {int(c) for c in s}
