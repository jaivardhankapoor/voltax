"""Smoke-run every example script in fast mode (``VOLTAX_FAST=1``)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("[0-9]*.py"))


@pytest.mark.extended
@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script):
    env = {**os.environ, "VOLTAX_FAST": "1", "MPLBACKEND": "Agg"}
    result = subprocess.run([sys.executable, str(script)], env=env,
                            capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-3000:]
