# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Minimal Wan2.2 package surface for EcoVideo T2V inference.
# Do not import I2V/S2V/Animate here: those paths require optional packages
# such as decord/librosa/peft/sam2 and are not part of this release.
from . import configs
from .text2video import WanT2V

__all__ = ["configs", "WanT2V"]
