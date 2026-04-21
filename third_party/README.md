# Third-party Code

This directory contains third-party components used by EcoVideo.

## What is kept in the public release

- `wan21/wan`: minimal Wan2.1 T2V path plus EcoVideo modifications.
- `wan22/wan`: minimal Wan2.2 T2V path plus EcoVideo modifications.
- `raft`: optional RAFT optical-flow estimator for interval scoring.

## What is intentionally not imported by default

The EcoVideo release focuses on T2V. Wan I2V, S2V, Animate, VACE and prompt-extension paths are not imported by `wan/__init__.py`, because those paths require optional packages such as `decord`, `librosa`, `peft`, `sam2`, `modelscope`, `dashscope`, and `xfuser`.

## Required action before public release

For each vendored project, preserve the upstream license and copyright notices,
add the upstream URL, and describe modifications. If an upstream license does not
allow redistribution, remove the vendored code and ask users to install/download
it separately.
