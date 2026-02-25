import sys
from pathlib import Path

import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
