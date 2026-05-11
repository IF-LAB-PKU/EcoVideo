# Third-party Code

This directory contains vendored third-party components used by EcoVideo.

## Components

| Component | Source | License | Description |
|---|---|---|---|
| `wan21/wan` | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | Apache 2.0 | Wan2.1 T2V path with EcoVideo entropy-based keyframe selection |
| `wan22/wan` | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | Apache 2.0 | Wan2.2 T2V path with EcoVideo entropy-based keyframe selection |
| `raft` | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) | BSD 3-Clause | Optional optical-flow estimator for interval scoring |

## Modifications

- **Wan2.1 / Wan2.2**: Added attention-entropy collection hooks, keyframe selection logic, and non-key context support for T2V. Removed non-T2V modules (I2V, S2V, animate, VACE, etc.).
- **RAFT**: Removed training and dataset code; kept inference core only.

## Note

The EcoVideo release focuses on T2V inference only. Third-party code remains subject to its original upstream license.
