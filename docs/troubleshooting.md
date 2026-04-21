# Troubleshooting

## `ModuleNotFoundError: easydict` / `ftfy` / `regex`

Install the base dependencies:

```bash
pip install -r requirements/base.txt
```

## `Unknown generator: wan`

Run from the repository root and set:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

Then run:

```bash
PYTHONPATH=src python tools/check_env.py
```

## Wan2.2 asks for `sam2`, `librosa`, `peft`, or `decord`

The public T2V release should not import S2V/Animate/I2V modules by default. Check that `third_party/wan22/wan/__init__.py` only imports `configs` and `WanT2V`.

## `xformers` install fails

Do not install xformers blindly. Install it only when a compatible wheel exists for your PyTorch and CUDA version. If unavailable, set `use_xformers: false` in the EDEN config and use the standard attention path.
