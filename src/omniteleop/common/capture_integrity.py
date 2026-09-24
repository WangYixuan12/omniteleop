"""Recording invariants. No robot commands, filtering, or guessed clock corrections."""
from __future__ import annotations

import time
import numpy as np


def state_snapshot(component, width):
    """Position and stamp from ONE cached message, never two racing getters."""
    state = component.get_state()
    pos = np.asarray(state['pos'], dtype=np.float64)
    stamp = state['timestamp_ns']
    if pos.shape != (width,) or not np.isfinite(pos).all():
        raise ValueError('Invalid/nonfinite joint positions')
    if not isinstance(stamp, (int, np.integer)) or stamp <= 0:
        raise ValueError('Missing integer source timestamp')
    return pos.copy(), int(stamp), state


class ProgressGuard:
    """Reject backward source clocks and a cached stream that stops progressing.

    This is a local liveness check, NOT an acquisition-age estimate across clocks.
    """
    def __init__(self, timeout_s=.15):
        self.timeout_s = timeout_s
        self.last = {}

    def check(self, key, stamp, now=None):
        now = time.monotonic() if now is None else now
        old, changed = self.last.get(key, (stamp, now))
        if stamp < old:
            raise ValueError(f'{key}: source clock moved backward')
        if stamp != old:
            changed = now
        if now - changed > self.timeout_s:
            raise ValueError(f'{key}: source stream stopped progressing')
        self.last[key] = (stamp, changed)


def freeze_ranges(pos, names, tolerance):
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != len(names) or len(pos) < 2:
        raise ValueError('Freeze check requires at least two samples x named joints')
    if not np.isfinite(pos).all() or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Nonfinite state or invalid freeze tolerance')
    span = np.ptp(pos, axis=0)
    return span, sorted([(n, float(p)) for n, p in zip(names, span) if p > tolerance],
                        key=lambda item: -item[1])


def validate_stereo_stamps(stamps, previous=None):
    """ZED eyes/depth in one recorded set must share one positive source stamp."""
    if not stamps or any(not isinstance(s, (int, np.integer)) or s <= 0 for s in stamps):
        return False
    return len(set(stamps)) == 1 and (previous is None or stamps[0] > previous)
