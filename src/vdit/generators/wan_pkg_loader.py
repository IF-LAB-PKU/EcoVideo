from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def _find_project_root(start: Path) -> Path:
    """Find repository root that contains third_party/wan21 and third_party/wan22."""
    candidates = []
    if os.environ.get("ECOVIDEO_ROOT"):
        candidates.append(Path(os.environ["ECOVIDEO_ROOT"]).expanduser().resolve())
    candidates.append(Path.cwd().resolve())
    candidates.extend(start.resolve().parents)

    for cand in candidates:
        if (cand / "third_party" / "wan21" / "wan").is_dir() and (
            cand / "third_party" / "wan22" / "wan"
        ).is_dir():
            return cand
    raise FileNotFoundError(
        "Cannot find EcoVideo repository root containing third_party/wan21/wan and "
        "third_party/wan22/wan. Run commands from the repository root or set ECOVIDEO_ROOT."
    )


def load_wan_package(root_dir: Path | None, version: str) -> ModuleType:
    """
    Load third_party/wanXX/wan as a uniquely named Python package:

      - version="2.1" -> package name "wan_v21"
      - version="2.2" -> package name "wan_v22"

    Using unique names avoids sys.path/sys.modules conflicts between Wan versions.
    """
    root = _find_project_root(root_dir or Path(__file__).resolve())
    if version == "2.1":
        pkg_dir = root / "third_party" / "wan21" / "wan"
        pkg_name = "wan_v21"
    elif version == "2.2":
        pkg_dir = root / "third_party" / "wan22" / "wan"
        pkg_name = "wan_v22"
    else:
        raise ValueError(f"Unknown wan version: {version}")

    init_py = pkg_dir / "__init__.py"
    if not init_py.is_file():
        raise FileNotFoundError(f"Cannot find {init_py}")

    if pkg_name in sys.modules:
        return sys.modules[pkg_name]  # type: ignore[return-value]

    spec = importlib.util.spec_from_file_location(
        pkg_name,
        init_py,
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create spec for {pkg_name} from {init_py}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod
