import sys
from pathlib import Path

# Make scripts/ importable as a package from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
