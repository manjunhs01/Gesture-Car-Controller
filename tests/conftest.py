import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real user data folder (logs, pid file)."""
    monkeypatch.setenv("GESTURE_CAR_HOME", str(tmp_path / "home"))
