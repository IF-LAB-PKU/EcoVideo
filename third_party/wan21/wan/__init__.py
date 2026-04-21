# Minimal Wan2.1 package surface for EcoVideo T2V inference.
# Non-T2V modules are intentionally not imported here to avoid optional
# dependencies being required for a text-to-video-only release.
from . import configs
from .text2video import WanT2V

__all__ = ["configs", "WanT2V"]
