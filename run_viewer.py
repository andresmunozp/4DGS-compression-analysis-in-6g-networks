#!/usr/bin/env python
"""Quick-start script for the 4DGaussians Metrics Viewer.

Equivalent to:  python -m viewers.metrics_viewer.app [args]

Run from the project root:
    python run_viewer.py
    python run_viewer.py --port 9090 --no-debug
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from metrics_viewer.app import main

if __name__ == "__main__":
    main()
