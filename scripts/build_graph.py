from __future__ import annotations

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.graph_builder import main


if __name__ == "__main__":
    main()

