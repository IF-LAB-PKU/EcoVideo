from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local EDEN config from the template.")
    parser.add_argument("--template", default="configs/eden_infer.yaml")
    parser.add_argument("--eden_ckpt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    template = Path(args.template)
    output = Path(args.output)
    if not template.is_file():
        raise FileNotFoundError(f"Template not found: {template}")
    if not Path(args.eden_ckpt).is_file():
        raise FileNotFoundError(f"EDEN checkpoint not found: {args.eden_ckpt}")

    cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
    cfg["pretrained_eden_path"] = str(Path(args.eden_ckpt).expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
