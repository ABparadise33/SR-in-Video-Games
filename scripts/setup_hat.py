#!/usr/bin/env python3
"""Clone and optionally install the official HAT repository."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path


HAT_REPO = "https://github.com/XPixelGroup/HAT.git"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hat-root", type=Path, default=Path("external/HAT"))
    parser.add_argument("--install", action="store_true", help="Run pip install and setup.py develop.")
    parser.add_argument(
        "--fix-basicsr-torchvision",
        action="store_true",
        help="Patch the older BasicSR torchvision import used by some environments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hat_root = args.hat_root

    if not hat_root.exists():
        hat_root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", HAT_REPO, str(hat_root)])
    else:
        print(f"Using existing HAT repository: {hat_root}")

    if args.install:
        run(["pip", "install", "-r", "requirements.txt"], cwd=hat_root)
        run(["python", "setup.py", "develop"], cwd=hat_root)

    if args.fix_basicsr_torchvision:
        spec = importlib.util.find_spec("basicsr")
        if spec is None or spec.origin is None:
            raise SystemExit("BasicSR is not importable. Run with --install first.")
        degradations = Path(spec.origin).parent / "data" / "degradations.py"
        text = degradations.read_text(encoding="utf-8")
        old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
        new = "from torchvision.transforms.functional import rgb_to_grayscale"
        if old in text:
            degradations.write_text(text.replace(old, new), encoding="utf-8")
            print(f"Patched {degradations}")
        else:
            print(f"No torchvision functional_tensor import found in {degradations}")


if __name__ == "__main__":
    main()
