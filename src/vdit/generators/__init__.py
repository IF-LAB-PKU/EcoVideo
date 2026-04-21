from __future__ import annotations

import warnings

from vdit.generators.base import VideoGenerator, create_generator, register_generator

# Side-effect imports: ensure decorators run and generators get registered.
# Import errors are surfaced as warnings so users can see which optional backend
# dependency is missing instead of getting a confusing "Unknown generator" later.
try:
    from vdit.generators import wan_t2v as _wan_t2v  # noqa: F401
except ImportError as e:  # pragma: no cover
    warnings.warn(f"Failed to import Wan generator: {e}", RuntimeWarning)

try:
    from vdit.generators import cogvideo_t2v as _cogvideo_t2v  # noqa: F401
except ImportError as e:  # pragma: no cover
    warnings.warn(f"Failed to import CogVideo generator: {e}", RuntimeWarning)

__all__ = [
    "VideoGenerator",
    "create_generator",
    "register_generator",
]
