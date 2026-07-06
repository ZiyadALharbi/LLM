"""Compatibility wrapper for the old pretraining/gpt.py entry point."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from complete_version.run_demo import main
from complete_version import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
