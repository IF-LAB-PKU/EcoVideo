# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Minimal T2V config surface used by EcoVideo.
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_t2v_A14B import t2v_A14B

WAN_CONFIGS = {
    't2v-A14B': t2v_A14B,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
}

SUPPORTED_SIZES = {
    't2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
}
