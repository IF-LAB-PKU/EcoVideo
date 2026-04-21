from __future__ import annotations

from typing import Dict, Protocol, Tuple, Type

import torch


class VideoGenerator(Protocol):
    def generate(self, prompt: str) -> Tuple[torch.Tensor, float]:
        ...


_GENERATORS: Dict[str, Type[VideoGenerator]] = {}


def register_generator(name: str):
    def deco(cls: Type[VideoGenerator]):
        _GENERATORS[name] = cls
        return cls

    return deco


def create_generator(name: str, **kwargs) -> VideoGenerator:
    if name not in _GENERATORS:
        raise ValueError(f"Unknown generator: {name}. Available: {sorted(_GENERATORS.keys())}")
    return _GENERATORS[name](**kwargs)
