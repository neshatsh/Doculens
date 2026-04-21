"""Shared pytest configuration — adds project root to sys.path."""
import sys
from pathlib import Path

# Ensure `src` is importable without installation
sys.path.insert(0, str(Path(__file__).parent))
