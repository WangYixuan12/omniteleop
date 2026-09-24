"""Reject SDK composite framesets that reuse IR while delivering a newer RGB frame."""
import numpy as np


class FramesetGuard:
    """Inspect metadata before image copies; rejected inputs never advance the reference.

    Repeats/skew are recoverable: drop the composite and wait for another. A clock or
    frame-number regression is fatal. Consumers retain the original frame timestamps
    and tick indices; no interpolation or substitution of another camera is performed.
    """
    STREAMS = ('left_ir', 'right_ir', 'color', 'depth')

    def __init__(self, streams, *, max_rgb_ir_ms=20., max_ir_pair_ms=2.):
        self.selected = [self.STREAMS.index(s) for s in streams]
        self.rgb_ir = float(max_rgb_ir_ms)
        self.ir_pair = float(max_ir_pair_ms)
        self.previous = None
        self.rejected = 0

    def inspect(self, numbers, stamps_ms):
        numbers = np.asarray(numbers, dtype=np.int64)
        stamps = np.asarray(stamps_ms, dtype=np.float64)
        if numbers.shape != (4,) or stamps.shape != (4,) or not np.isfinite(stamps).all() or np.any(stamps <= 0):
            raise ValueError('Invalid RealSense frame provenance')
        if self.previous is not None:
            old_numbers, old_stamps = self.previous
            selected = self.selected
            if np.any(numbers[selected] < old_numbers[selected]) or np.any(stamps[selected] < old_stamps[selected]):
                raise ValueError('RealSense stored stream moved backward or restarted')
            if np.any(numbers[selected] == old_numbers[selected]) or np.any(stamps[selected] == old_stamps[selected]):
                self.rejected += 1
                return 'stored stream reused in newer composite frameset'
        if abs(stamps[0]-stamps[1]) > self.ir_pair:
            self.rejected += 1
            return 'IR stereo capture skew exceeds 2 ms'
        if abs(stamps[2]-stamps[0]) > self.rgb_ir:
            self.rejected += 1
            return 'RGB/IR capture skew exceeds 20 ms'
        self.previous = (numbers.copy(), stamps.copy())
        return None
