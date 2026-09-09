from types import SimpleNamespace
import pytest

from teleop_joystick import SimSession
from joystick_service import BehaviorDriver


def test_stopped_timeline_is_rejected_before_reading_articulation():
    session = SimSession(SimpleNamespace(og=SimpleNamespace(sim=SimpleNamespace(
        is_playing=lambda: False))), None, SimpleNamespace())
    with pytest.raises(RuntimeError, match="timeline"):
        session.snapshot()


def test_fatal_rpc_is_not_retried_during_cleanup(monkeypatch):
    import joystick_service as service
    driver = BehaviorDriver.__new__(BehaviorDriver)
    driver.conn = object()
    monkeypatch.setattr(service, 'send_msg', lambda *args: None)
    monkeypatch.setattr(service, 'recv_msg', lambda *args: {'error': 'articulation failed'})
    with pytest.raises(RuntimeError, match='articulation failed'):
        driver.rpc('step')
    monkeypatch.setattr(service, 'send_msg', lambda *args: pytest.fail('retried failed RPC'))
    with pytest.raises(ConnectionError):
        driver.rpc('stop')


def test_human_long_horizon_result_does_not_evaluate_automatic_success():
    def forbidden_success(env):
        pytest.fail('Unannotated human demo evaluated as a scripted task')
    task = SimpleNamespace(UNANNOTATED=True, success=forbidden_success)
    session = SimSession(None, task, SimpleNamespace())
    assert session.handle({'cmd': 'episode_result'}) == {
        'task_success': False, 'success_evaluated': False}
