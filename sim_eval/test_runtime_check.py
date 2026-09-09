"""Catch incompatible Isaac versions before the robot loader starts."""
from types import SimpleNamespace

import runtime_check


def test_version_mismatch_is_actionable_without_importing_isaac(tmp_path, monkeypatch):
    (tmp_path / 'simulator.py').write_text('m.KIT_FILES = {(5, 1, 0): "omnigibson_5_1_0.kit"}\n')
    monkeypatch.setattr(runtime_check.importlib.util, 'find_spec',
                        lambda name: SimpleNamespace(origin=str(tmp_path / '__init__.py')))
    monkeypatch.setattr(runtime_check.importlib.metadata, 'version', lambda name: '4.5.0.0')
    message = runtime_check.isaac_version_problem()
    assert '4.5.0.0' in message and '5.1.0' in message
    monkeypatch.setattr(runtime_check.importlib.metadata, 'version', lambda name: '5.1.0.0')
    assert runtime_check.isaac_version_problem() is None
