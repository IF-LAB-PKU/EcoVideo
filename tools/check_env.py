from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

REQUIRED_IMPORTS = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "einops",
    "torchdiffeq",
    "yaml",
    "numpy",
    "scipy",
    "PIL",
    "cv2",
    "av",
    "tqdm",
    "easydict",
    "ftfy",
    "regex",
]


def check_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception as exc:
        print(f"[MISSING] {name}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Check EcoVideo runtime environment and checkpoint paths.")
    parser.add_argument("--wan21_ckpt", default=os.environ.get("WAN21_CKPT"))
    parser.add_argument("--wan22_ckpt", default=os.environ.get("WAN22_CKPT"))
    parser.add_argument("--cogvideo_ckpt", default=os.environ.get("COGVIDEO_CKPT"))
    parser.add_argument("--eden_ckpt", default=os.environ.get("EDEN_CKPT"))
    parser.add_argument("--raft_ckpt", default=os.environ.get("RAFT_CKPT"))
    args = parser.parse_args()

    print("== Python imports ==")
    ok = True
    for name in REQUIRED_IMPORTS:
        ok = check_import(name) and ok

    try:
        import torch
        print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device_count={torch.cuda.device_count()}, first_gpu={torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    print("\n== Checkpoint paths ==")
    for label, value, is_file in [
        ("WAN21_CKPT", args.wan21_ckpt, False),
        ("WAN22_CKPT", args.wan22_ckpt, False),
        ("COGVIDEO_CKPT", args.cogvideo_ckpt, False),
        ("EDEN_CKPT", args.eden_ckpt, True),
        ("RAFT_CKPT", args.raft_ckpt, True),
    ]:
        if not value:
            print(f"[SKIP] {label} not set")
            continue
        p = Path(value).expanduser()
        exists = p.is_file() if is_file else p.is_dir()
        print(f"[{'OK' if exists else 'BAD'}] {label}={p}")
        ok = exists and ok

    print("\n== EcoVideo imports ==")
    try:
        import vdit  # noqa: F401
        from vdit.generators.wan_pkg_loader import load_wan_package
        root = Path.cwd()
        w21 = load_wan_package(root, "2.1")
        w22 = load_wan_package(root, "2.2")
        print(f"[OK] loaded {w21.__name__} and {w22.__name__}")
    except Exception as exc:
        ok = False
        print(f"[BAD] EcoVideo import check failed: {exc}")

    if not ok:
        raise SystemExit(1)
    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
