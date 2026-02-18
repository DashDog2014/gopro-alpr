import sys
from pathlib import Path

# Add repo root to import path so "import src..." works from /scripts
sys.path.append(str(Path(__file__).resolve().parents[1]))