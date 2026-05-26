"""Shared path constants for orbit-paper research scripts.

Resolves repo-rooted paths from this file's location so scripts work
regardless of the CWD they're invoked from.
"""

from pathlib import Path

# This file lives at scripts/research/orbit_paper/_paths.py; up 3 dirs is repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = REPO_ROOT / "paper"
SOLUTIONS_DIR = PAPER_DIR / "solutions"
DATA_DIR = PAPER_DIR / "data"
