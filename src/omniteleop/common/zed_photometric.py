"""Observe ZED exposure state from its owning grab thread, without setters.

EXPOSURE/GAIN are SDK slider values, not milliseconds or physical gain. Settings
are sampled once per second and are not exact per-exposure measurements.
"""
import time


class PhotometricTelemetry:
    SETTINGS = ('AEC_AGC', 'EXPOSURE', 'GAIN', 'WHITEBALANCE_AUTO',
                'WHITEBALANCE_TEMPERATURE', 'EXPOSURE_TIME', 'ANALOG_GAIN', 'DIGITAL_GAIN')

    def __init__(self, camera, sdk, interval_s=1.):
        if interval_s <= 0:
            raise ValueError('interval_s must be positive')
        self.camera, self.sdk, self.interval_s = camera, sdk, interval_s
        self.last_poll = -float('inf')
        self.snapshot = {'status': 'not_sampled', 'capture_timestamp_ns': None}

    def update(self, capture_timestamp_ns):
        now = time.monotonic()
        if now-self.last_poll < self.interval_s:
            return
        self.last_poll = now
        values, errors = {}, {}
        begin = time.perf_counter()
        for name in self.SETTINGS:
            setting = getattr(self.sdk.VIDEO_SETTINGS, name, None)
            if setting is None:
                errors[name] = 'unsupported_enum'
                continue
            try:
                status, value = self.camera.get_camera_settings(setting)
                if status == self.sdk.ERROR_CODE.SUCCESS:
                    values[name] = int(value)
                else:
                    errors[name] = str(status)
            except Exception as exc:
                errors[name] = type(exc).__name__ + ': ' + str(exc)
        # Replace atomically so the info callback never observes a partial update.
        self.snapshot = dict(status='sampled', capture_timestamp_ns=int(capture_timestamp_ns),
                             wall_timestamp_ns=time.time_ns(), poll_duration_ms=(time.perf_counter()-begin)*1000,
                             values=values, unavailable=errors,
                             semantics='SDK settings sampled after image publication, not per-exposure metadata; EXPOSURE/GAIN are SDK slider units')
