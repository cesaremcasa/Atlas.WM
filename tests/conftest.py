"""Shared pytest setup.

Makes ``scripts/`` importable regardless of the invocation cwd — tests
import the pipeline scripts (``train``, ``split_data``, ``export_onnx``,
``oracle_friction_agent``) directly, and a relative ``sys.path`` entry only
worked when pytest ran from the repo root (review finding M-1).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
