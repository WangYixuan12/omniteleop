"""Per-frame RealSense provenance; optional SDK metadata always has a validity bit."""
import json
import time

import numpy as np

FIELDS = ('sensor_timestamp', 'frame_timestamp', 'actual_exposure', 'gain_level',
          'auto_exposure', 'white_balance', 'backend_timestamp', 'time_of_arrival')


def frame_metadata(frames, rs):
    if not frames or not all(frames):
        raise ValueError('Incomplete RealSense frameset')
    stamps = np.asarray([f.get_timestamp() for f in frames], dtype=np.float64)
    if not np.isfinite(stamps).all() or (stamps <= 0).any():
        raise ValueError('Invalid RealSense timestamp')
    domains = [str(f.get_frame_timestamp_domain()) for f in frames]
    if any(d != str(rs.timestamp_domain.global_time) for d in domains):
        raise ValueError(f'RealSense global time not established: {domains}')
    values = np.zeros((len(frames), len(FIELDS)), dtype=np.int64)
    valid = np.zeros_like(values, dtype=np.bool_)
    for i, f in enumerate(frames):
        for j, name in enumerate(FIELDS):
            enum = getattr(rs.frame_metadata_value, name, None)
            if enum is not None and f.supports_frame_metadata(enum):
                values[i, j] = f.get_frame_metadata(enum)
                valid[i, j] = True
    return dict(stream_timestamp_ms=stamps, frame_metadata=values,
                frame_metadata_valid=valid, host_sample_ns=np.int64(time.time_ns()))


def sensor_settings(dev, rs):
    """Startup snapshot in SDK option units; not mislabeled per-frame exposure."""
    settings = {}
    for sensor in dev.query_sensors():
        name = sensor.get_info(rs.camera_info.name)
        opts = {}
        for key in ('exposure', 'gain', 'white_balance', 'enable_auto_exposure',
                    'enable_auto_white_balance', 'emitter_enabled', 'laser_power'):
            option = getattr(rs.option, key)
            if sensor.supports(option):
                opts[key] = dict(value=float(sensor.get_option(option)),
                                 description=sensor.get_option_description(option))
        settings[name] = opts
    for key in ('serial_number', 'firmware_version', 'name'):
        enum = getattr(rs.camera_info, key)
        if dev.supports(enum):
            settings[key] = dev.get_info(enum)
    return json.dumps(settings, sort_keys=True, allow_nan=False)


def specs(n_streams):
    return (('stream_timestamp_ms', (n_streams,), np.float64),
            ('frame_metadata', (n_streams, len(FIELDS)), np.int64),
            ('frame_metadata_valid', (n_streams, len(FIELDS)), np.bool_),
            ('host_sample_ns', (), np.int64))


def describe(group, prefix, streams):
    group.attrs[prefix + '_stream_order'] = json.dumps(streams)
    group.attrs[prefix + '_metadata_fields'] = json.dumps(FIELDS)
    group.attrs[prefix + '_timestamp_domain'] = 'global_time; SDK frame timestamp, not asserted exposure midpoint'
    group.attrs[prefix + '_metadata_note'] = 'SDK raw metadata units; false validity means unavailable, never zero exposure'


def set_option_checked(sensor, option, value):
    """Set once, then allow up to 0.5 s for delayed UVC option readback.

    Real D455 white-balance writes can initially return the previous setting.
    This bounded startup check still fails if the requested value never appears.
    """
    value = float(value)
    limits = sensor.get_option_range(option)
    if not np.isfinite(value) or not limits.min <= value <= limits.max:
        raise ValueError(f"Invalid {option} value {value}; range {limits.min}..{limits.max}")
    if limits.step > 0:
        value = limits.min + round((value - limits.min) / limits.step) * limits.step
    sensor.set_option(option, value)
    deadline = time.monotonic() + .5
    got = float(sensor.get_option(option))
    while (not np.isfinite(got) or abs(got - value) > max(1e-6, limits.step / 2)) and time.monotonic() < deadline:
        time.sleep(.02)
        got = float(sensor.get_option(option))
    if not np.isfinite(got) or abs(got - value) > max(1e-6, limits.step / 2):
        raise ValueError(f"{option} readback {got} differs from requested {value}")
    return got

# Option values are SDK readbacks, NOT inferred exposure metadata for a frame.
OPTION_FIELDS = ('exposure', 'gain', 'white_balance', 'enable_auto_exposure',
                 'enable_auto_white_balance', 'emitter_enabled', 'laser_power')
COLOR_FIELDS = ('red_mean', 'green_mean', 'blue_mean', 'luma_mean', 'luma_p05',
                'luma_p95', 'any_channel_ge_250_fraction', 'all_channels_le_5_fraction')


def option_snapshot(sensors, rs):
    start = time.time_ns()
    values = np.zeros((len(sensors), len(OPTION_FIELDS)), dtype=np.float64)
    valid = np.zeros_like(values, dtype=np.bool_)
    for i, sensor in enumerate(sensors):
        for j, name in enumerate(OPTION_FIELDS):
            option = getattr(rs.option, name, None)
            try:
                if option is not None and sensor.supports(option):
                    value = float(sensor.get_option(option))
                    if np.isfinite(value):
                        values[i, j], valid[i, j] = value, True
            except RuntimeError:
                pass  # A failed optional readback is missing, never a zero setting.
    return dict(sensor_option_values=values, sensor_option_valid=valid,
                sensor_option_start_ns=np.int64(start),
                sensor_option_end_ns=np.int64(time.time_ns()))


def color_statistics(rgb, roi=(0., 0., 1., 1.)):
    roi = np.asarray(roi, dtype=float)
    if (roi.shape != (4,) or not np.isfinite(roi).all() or np.any(roi < 0)
            or np.any(roi > 1) or roi[0] >= roi[2] or roi[1] >= roi[3]):
        raise ValueError('Quality ROI must be normalized x0,y0,x1,y1 with positive area')
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected RGB uint8 image')
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = np.floor(roi * [w, h, w, h]).astype(int)
    region = rgb[y0:y1, x0:x1]
    if region.size == 0:
        raise ValueError('Quality ROI contains no pixels at this resolution')
    # Decimation bounds the capture-process overhead, about 25k sampled pixels.
    step = max(1, int(np.ceil(np.sqrt(region.shape[0] * region.shape[1] / 25000))))
    px = region[::step, ::step].reshape(-1, 3)
    luma = px @ np.array([.2126, .7152, .0722])
    return np.asarray([*px.mean(axis=0), luma.mean(), *np.percentile(luma, [5, 95]),
                       np.any(px >= 250, axis=1).mean(), np.all(px <= 5, axis=1).mean()])


def telemetry_specs():
    return (('sensor_option_values', (2, len(OPTION_FIELDS)), np.float64),
            ('sensor_option_valid', (2, len(OPTION_FIELDS)), np.bool_),
            ('sensor_option_start_ns', (), np.int64),
            ('sensor_option_end_ns', (), np.int64),
            ('color_quality', (len(COLOR_FIELDS),), np.float64))


def apply_serial_profile(dev, rs, profile):
    """Explicit paired exposure/gain settings in SDK option units, keyed by serial."""
    allowed = {'color_exposure_option', 'color_gain', 'white_balance',
               'ir_exposure_option', 'ir_gain', 'laser_power'}
    if not isinstance(profile, dict) or set(profile) - allowed:
        raise ValueError(f'Invalid camera profile: expected keys {sorted(allowed)}')
    for exposure, gain in (('color_exposure_option', 'color_gain'), ('ir_exposure_option', 'ir_gain')):
        if (exposure in profile) != (gain in profile):
            raise ValueError(f'{exposure} and {gain} must be supplied together')
    for sensor, prefix in ((dev.first_color_sensor(), 'color'), (dev.first_depth_sensor(), 'ir')):
        exposure, gain = prefix + '_exposure_option', prefix + '_gain'
        if exposure in profile:
            set_option_checked(sensor, rs.option.enable_auto_exposure, 0)
            set_option_checked(sensor, rs.option.exposure, profile[exposure])
            set_option_checked(sensor, rs.option.gain, profile[gain])
    if 'white_balance' in profile:
        sensor = dev.first_color_sensor()
        set_option_checked(sensor, rs.option.enable_auto_white_balance, 0)
        set_option_checked(sensor, rs.option.white_balance, profile['white_balance'])
    if 'laser_power' in profile:
        set_option_checked(dev.first_depth_sensor(), rs.option.laser_power, profile['laser_power'])
