from __future__ import annotations

import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
demo_root_str = str(DEMO_ROOT)
if demo_root_str not in sys.path:
    sys.path.insert(0, demo_root_str)