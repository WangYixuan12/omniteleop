"""Native joint and observed joint-command sidecars, independent of image row rate.

Passive subscriptions only. No hardware commands, interpolation or inferred latency.
Missing/failed sidecars keep .partial names and must block dataset admission.
"""
from __future__ import annotations

from collections import deque
import copy
import json
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np

STATE_TOPICS = {'left_arm': 'state/arm/left', 'right_arm': 'state/arm/right',
                'head': 'state/head', 'torso': 'state/torso'}


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        item = value.item()
        return json_value(item) if isinstance(item, bytes) else item
    if isinstance(value, bytes):
        return value.decode('utf8')
    raise TypeError(type(value).__name__)


def episode_gap_evidence(stamps, start_wall_ns, stop_wall_ns, clock):
    """Separate episode coverage from unused pre/post-roll, in the source clock.

    Expand the boundary by half the probe RTT: uncertainty must not hide a gap.
    A late tail gap is retained as evidence, but cannot invalidate earlier images.
    """
    offset, rtt = clock['offset_ns'], clock['rtt_ns']
    if (not isinstance(offset, (int, np.integer))
            or not isinstance(rtt, (int, np.integer)) or not 0 <= rtt <= 10_000_000):
        raise ValueError('Invalid native clock offset/RTT')
    start = int(start_wall_ns) + int(offset) - (int(rtt) + 1) // 2
    stop = int(stop_wall_ns) + int(offset) + (int(rtt) + 1) // 2
    if stop < start:
        raise ValueError('Native episode stop precedes start')
    result = {}
    for group, values in stamps.items():
        ts = np.asarray(values, dtype=np.int64)
        if len(ts) < 2 or np.any(np.diff(ts) <= 0):
            raise ValueError(f'Invalid native source sequence: {group}')
        gaps = np.diff(ts)
        needed = (ts[:-1] < stop) & (ts[1:] > start)
        result[group] = dict(
            source_window_ns=[start, stop],
            bracketed=bool(ts[0] <= start and ts[-1] >= stop),
            max_episode_source_gap_ns=int(gaps[needed].max()) if needed.any() else 0,
            max_all_source_gap_ns=int(gaps.max()),
            outside_gap_over_50ms_count=int(np.sum((gaps > 50_000_000) & ~needed)))
    return result


def record_event(owner, kind, payload):
    """Append recording evidence without issuing hardware commands or doing file I/O."""
    recorder = getattr(owner, '_native_state_recorder', None)
    if recorder is not None:
        recorder.event(kind, payload)


class NativeStateRecorder:
    SCHEMA = 'native_state_v2'
    def __init__(self, node, save_dir, joint_names, command_topics, *, capacity=8192):
        from dexcomm.codecs import JointStateCodec, JointCmdCodec
        self.save_dir = Path(save_dir)
        self.names = joint_names
        if set(joint_names) != set(STATE_TOPICS) or any(not v for v in joint_names.values()):
            raise ValueError('Native recording requires four named joint streams')
        self.capacity = capacity
        if capacity < 4096:
            raise ValueError('Native queue must accommodate the full pre-roll buffer')
        self.lock = threading.Lock()
        self.history = deque(maxlen=4096)
        self.latest = {}
        self.active = None
        self.jobs = []
        self.subs = {}
        for group, topic in STATE_TOPICS.items():
            self.subs['state/' + group] = node.create_subscriber(
                topic, decoder=JointStateCodec.decode,
                callback=self._callback('state', group), buffer_size=4096)
        for group, topic in command_topics.items():
            self.subs['command/' + group] = node.create_subscriber(
                topic, decoder=JointCmdCodec.decode,
                callback=self._callback('command', group), buffer_size=4096)
        self.command_topics = command_topics
        self._clock_stop = threading.Event()
        self._clock_thread = None

    def _append(self, row):
        with self.lock:
            self.history.append(row)
            if row['type'] == 'state':
                self.latest[row['group']] = row
            for job in self.jobs:
                if job['closed'] or time.monotonic() >= job['stop_at']:
                    continue
                try:
                    job['queue'].put_nowait(row)
                except queue.Full:
                    job['drops'] += 1

    def event(self, kind, payload):
        self._append(dict(type='event', group=kind, callback_wall_ns=time.time_ns(),
                          callback_monotonic_ns=time.monotonic_ns(),
                          payload=copy.deepcopy(payload)))

    def _callback(self, kind, group):
        def callback(message):
            self._append(dict(type=kind, group=group, callback_wall_ns=time.time_ns(),
                              callback_monotonic_ns=time.monotonic_ns(),
                              payload=copy.deepcopy(getattr(message, 'data', message))))
        return callback

    def start_clock_sampling(self, queries, *, period_s=30.0):
        """Read-only clock probes, outside the servo/image threads; retain failures too.

        Calibration events do not rewrite the episode's initial clock calibration or
        change live control/age gates. Offline consumers use the entire probe history.
        """
        if self._clock_thread is not None:
            raise RuntimeError('Clock sampler already started')
        if period_s <= 0:
            raise ValueError('Clock sampling period must be positive')
        def run():
            while not self._clock_stop.wait(period_s):
                for domain, query in queries.items():
                    with self.lock:
                        active = self.active is not None
                    if self._clock_stop.is_set() or not active:
                        break
                    try:
                        calibration = query()
                        self.event('clock', dict(domain=domain, calibration=calibration))
                    except Exception as exc:
                        self.event('clock_error', dict(domain=domain, error=repr(exc)))
                        print(f'[native_state] clock probe failed ({domain}): {exc}', flush=True)
        self._clock_thread = threading.Thread(target=run, name='recording_clock_probes', daemon=True)
        self._clock_thread.start()

    def start(self, episode_id, metadata):
        with self.lock:
            if self.active is not None:
                raise RuntimeError('Native recorder already active')
            now = time.monotonic_ns()
            if any(now - self.latest.get(g, {}).get('callback_monotonic_ns', 0) > 150_000_000
                   for g in self.names):
                raise RuntimeError('Native state missing/stale before episode; wait for all four streams')
            path = self.save_dir / f'episode_{episode_id}_native_{uuid.uuid4().hex}.jsonl'
            job = dict(path=path, queue=queue.Queue(self.capacity), stop_at=float('inf'),
                       start_ns=time.time_ns(), stop_ns=None, drops=0, closed=False,
                       aborted=False, error=None)
            for row in self.history:
                if now - row['callback_monotonic_ns'] <= 1_000_000_000:
                    job['queue'].put_nowait(row)
            header = dict(type='metadata', schema=self.SCHEMA, episode_id=episode_id,
                          start_wall_ns=job['start_ns'], joint_names=self.names,
                          state_topics=STATE_TOPICS, command_topics=self.command_topics,
                          clock_calibration=copy.deepcopy(metadata), pre_roll_s=1, post_roll_s=.5,
                          timestamp_semantics='source publication stamp; acquisition semantics unverified; callback is not receive time',
                          action_coverage='joint wire topics plus follower dispatch intervals, base mode/twist, FC16 sends and FC03 requests/replies; API dispatch is not actuator execution',
                          base_pose_semantics='ARKit selected by recording_context; t_ns in existing publisher is post-USB host time, NOT phone acquisition time')
            job['thread'] = threading.Thread(target=self._write, args=(job, header), daemon=True)
            self.jobs = [j for j in self.jobs if not j['closed'] or j['thread'].is_alive()]
            self.jobs.append(job)
            self.active = job
            job['thread'].start()
            return path.name

    def stop(self, *, aborted=False):
        with self.lock:
            if self.active is not None:
                self.active.update(stop_at=time.monotonic() + .5, stop_ns=time.time_ns(),
                                   aborted=bool(aborted))
                self.active = None

    def _write(self, job, header):
        partial = Path(str(job['path']) + '.partial')
        counts = {g: 0 for g in self.names}
        source_stamps = {g: [] for g in self.names}
        previous, first, last, max_gap = {}, {}, {}, {g: 0 for g in self.names}
        errors = set()
        event_counts = {}
        try:
            partial.parent.mkdir(parents=True, exist_ok=True)
            with partial.open('x', buffering=65536) as f:
                f.write(json.dumps(header, default=json_value, allow_nan=False) + '\n')
                flushed = time.monotonic()
                while True:
                    with self.lock:
                        done = time.monotonic() >= job['stop_at'] and job['queue'].empty()
                        if done:
                            job['closed'] = True
                    if done:
                        break
                    try:
                        row = job['queue'].get(timeout=.05)
                    except queue.Empty:
                        continue
                    if row['type'] == 'event':
                        event_counts[row['group']] = event_counts.get(row['group'], 0) + 1
                    if row['type'] == 'state':
                        group, data = row['group'], row['payload']
                        pos = np.asarray(data['pos'], dtype=float)
                        stamp = data['timestamp_ns']
                        if pos.shape != (len(self.names[group]),) or not np.isfinite(pos).all():
                            errors.add(f'invalid state: {group}')
                            continue
                        if not isinstance(stamp, (int, np.integer)) or stamp <= 0:
                            errors.add(f'invalid source time: {group}')
                            continue
                        if group in previous:
                            gap = int(stamp) - previous[group]
                            if gap <= 0:
                                errors.add(f'nonmonotonic source time: {group}')
                            max_gap[group] = max(max_gap[group], gap)
                        previous[group] = int(stamp)
                        source_stamps[group].append(int(stamp))
                        counts[group] += 1
                        first.setdefault(group, row['callback_wall_ns'])
                        last[group] = row['callback_wall_ns']
                    f.write(json.dumps(row, default=json_value, allow_nan=False) + '\n')
                    if time.monotonic() - flushed >= 1:
                        f.flush()
                        flushed = time.monotonic()
                gap_evidence = {}
                try:
                    gap_evidence = episode_gap_evidence(
                        source_stamps, job['start_ns'], job['stop_ns'],
                        header['clock_calibration']['ntp'])
                except (ValueError, KeyError, TypeError) as exc:
                    errors.add(f'episode source coverage unavailable: {exc}')
                for group in self.names:
                    if counts[group] < 2 or first.get(group, 2**63) > job['start_ns'] or last.get(group, 0) < job['stop_ns']:
                        errors.add(f'unbracketed episode: {group}')
                    if group in gap_evidence:
                        if not gap_evidence[group]['bracketed']:
                            errors.add(f'unbracketed source episode: {group}')
                        if gap_evidence[group]['max_episode_source_gap_ns'] > 50_000_000:
                            errors.add(f'state gap >50 ms inside episode: {group}')
                subscriber_stats = {}
                for topic, sub in self.subs.items():
                    if hasattr(sub, 'get_stats'):
                        subscriber_stats[topic] = sub.get_stats()
                        if subscriber_stats[topic].get('callback_dropped_count', 0):
                            errors.add(f'callback loss since recorder startup: {topic}')
                complete = not (errors or job['drops'] or job['aborted'])
                summary = dict(type='summary', complete=complete, stop_wall_ns=job['stop_ns'],
                               counts=counts, event_counts=event_counts, max_source_gap_ns=max_gap, queue_drops=job['drops'],
                               episode_gap_evidence=gap_evidence,
                               errors=sorted(errors), aborted=job['aborted'],
                               subscriber_stats=subscriber_stats,
                               limitation='No source sequence IDs: transport completeness is not proven by timestamp cadence')
                f.write(json.dumps(summary, default=json_value, allow_nan=False) + '\n')
            job['summary'] = summary
            if complete:
                partial.rename(job['path'])
                print(f'[native_state] complete: {job["path"]}', flush=True)
            else:
                print(f'[native_state] INCOMPLETE: {partial}; '
                      f'errors={summary["errors"]}; queue_drops={job["drops"]}; '
                      f'aborted={job["aborted"]}', flush=True)
        except Exception as exc:
            job['error'] = repr(exc)
            Path(str(partial) + '.error').write_text(repr(exc))
            print(f'[native_state] WRITE FAILED: {partial}: {exc!r}', flush=True)
        finally:
            with self.lock:
                job['closed'] = True

    def close(self):
        self._clock_stop.set()
        if self._clock_thread is not None:
            self._clock_thread.join(timeout=8)
            if self._clock_thread.is_alive():
                print('[native_state] clock probe still exiting; no further probes scheduled', flush=True)
        self.stop(aborted=True)
        for job in self.jobs:
            job['thread'].join(timeout=5)
            if job['thread'].is_alive():
                raise RuntimeError('Native writer did not stop; sidecar incomplete')
        for sub in self.subs.values():
            sub.shutdown()
