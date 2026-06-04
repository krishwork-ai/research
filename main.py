"""
main.py — entrypoint for `uv run main.py <command>`.

Loads .env if present, then delegates to cli.main().
"""

from __future__ import annotations

import sys
from pathlib import Path


def _load_dotenv() -> None:
    """
    Minimal .env loader — no python-dotenv required.
    Reads KEY=value lines from .env in the project root and sets
    them in os.environ if not already present.
    """
    import os

    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    _load_dotenv()

    from research.cli import main

    sys.exit(main())
