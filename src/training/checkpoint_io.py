"""Fail-closed helpers for PyTorch checkpoint persistence.

Only tensor state and primitive metadata are supported.  In particular, this
module never falls back to Python's unrestricted pickle loader and never
allowlists checkpoint-provided Python globals.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping


def normalize_checkpoint_metadata(value: Any) -> Any:
    """Convert research metrics to weights-only-safe primitive containers.

    Training metrics commonly originate as NumPy arrays or scalar values.  A
    direct ``torch.save`` of those objects records NumPy reconstruction
    globals, which strict weights-only loading correctly refuses.  Normalising
    metadata at write time retains the values without expanding the loader's
    trusted-global set.
    """

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - NumPy is a declared dependency
        np = None
    if np is not None:
        if isinstance(value, np.generic):
            return normalize_checkpoint_metadata(value.item())
        if isinstance(value, np.ndarray):
            return normalize_checkpoint_metadata(value.tolist())

    # NumPy 2.x floating scalars can also satisfy ``isinstance(x, float)``;
    # therefore NumPy normalization must precede the primitive fast path.
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, (str, bytes, bool, int, float)):
                raise TypeError(
                    "Checkpoint metadata keys must be primitive values; "
                    f"received {type(key).__name__}"
                )
            normalized[key] = normalize_checkpoint_metadata(item)
        return normalized
    if isinstance(value, list):
        return [normalize_checkpoint_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_checkpoint_metadata(item) for item in value)

    raise TypeError(
        "Checkpoint metadata must contain only primitive values; "
        f"received {type(value).__name__}"
    )


def load_weights_only_checkpoint(
    path: Path | str, *, map_location: Any = "cpu"
) -> Mapping[str, Any]:
    """Load a checkpoint mapping without permitting arbitrary pickle code."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - declared dependency
        raise RuntimeError("PyTorch is required to load a checkpoint") from error

    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "Secure checkpoint loading requires a PyTorch version that supports "
            "weights_only=True"
        ) from error

    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return checkpoint


def tensor_state_dict_semantic_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash dense tensor values independently of checkpoint serialization.

    The raw bytes of a ``.pth`` file are intentionally retained as a separate
    transport-integrity digest.  This digest represents the ordered tensor
    names, dtypes, shapes, and values, so copying a model into a new campaign
    or reserializing the same state cannot manufacture a new scientific model
    identity.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - declared dependency
        raise RuntimeError("PyTorch is required to hash a model state") from error

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Model state must be a non-empty tensor mapping")

    digest = hashlib.sha256()
    digest.update(b"dense-tensor-state-dict-semantic-sha256-v1\0")

    def update_field(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)

    keys = list(state_dict)
    if any(not isinstance(key, str) or not key for key in keys):
        raise TypeError("Model-state keys must be non-empty strings")
    if len(set(keys)) != len(keys):  # defensive for unusual Mapping objects
        raise ValueError("Model-state keys must be unique")

    for key in sorted(keys):
        tensor = state_dict[key]
        if not torch.is_tensor(tensor):
            raise TypeError(f"Model-state value {key!r} is not a tensor")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise TypeError(
                f"Model-state tensor {key!r} must be dense and non-quantized"
            )
        if tensor.device.type == "meta":
            raise TypeError(f"Model-state tensor {key!r} has no materialized values")

        materialized = tensor.detach().cpu().contiguous()
        raw = (
            materialized.reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes(order="C")
        )
        update_field(key.encode("utf-8"))
        update_field(str(materialized.dtype).encode("ascii"))
        update_field(
            ",".join(str(int(size)) for size in materialized.shape).encode("ascii")
        )
        update_field(raw)

    return digest.hexdigest()
