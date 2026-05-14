import threading
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional

from dexcomm.codecs import EEPassThroughCmdCodec
from dexcontrol.core.arm import Arm

SLAVE_ID = 0x09

# Robotiq Hand-E status registers: 0x07D0..0x07D2 (3 registers, 6 bytes).
STATUS_REG_ADDR_HI = 0x07
STATUS_REG_ADDR_LO = 0xD0
STATUS_REG_COUNT = 3

HANDE_RESET_COMMAND = bytes.fromhex("09 10 03 E8 00 03 06 00 00 00 00 00 00 73 30")
HANDE_ACTIVATE_COMMAND = bytes.fromhex("09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1")
FC16_WRITE_MULTIPLE_REGISTERS = 0x10


def modbus_crc(data: bytes) -> bytes:
    """Modbus RTU CRC16, returned in little-endian byte order."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc.to_bytes(2, byteorder="little")


def build_hande_command(pos: float, speed: float = 1.0, force: float = 1.0) -> bytes:
    """Build a Robotiq Hand-E 'go to position' Modbus RTU command.

    Args:
        pos:   normalized position in [0.0, 1.0]
               0.0 = fully open, 1.0 = fully closed
        speed: normalized speed in [0.0, 1.0]
        force: normalized force in [0.0, 1.0]
    """
    pos = max(0.0, min(1.0, pos))
    speed = max(0.0, min(1.0, speed))
    force = max(0.0, min(1.0, force))

    rPR = round(pos * 255)  # position request
    rSP = round(speed * 255)  # speed
    rFR = round(force * 255)  # force

    # Function 0x10 = write multiple registers
    # Start address = 0x03E8
    # Write 3 registers = 6 data bytes
    #
    # Data bytes:
    #   [0]=ACTION REQUEST   = 0x09  (rACT=1, rGTO=1)
    #   [1]=GRIPPER OPTIONS  = 0x00
    #   [2]=OPTIONS 2        = 0x00
    #   [3]=POSITION REQUEST = rPR
    #   [4]=SPEED            = rSP
    #   [5]=FORCE            = rFR
    payload = bytes(
        [
            SLAVE_ID,
            0x10,
            0x03,
            0xE8,
            0x00,
            0x03,
            0x06,
            0x09,
            0x00,
            0x00,
            rPR,
            rSP,
            rFR,
        ]
    )
    return payload + modbus_crc(payload)


def send_ee_pass_through_with_timestamps(arm: Arm, message: bytes) -> Dict[str, Any]:
    """Send one EE pass-through message and return local send timestamps."""
    event = {
        "send_monotonic_ns": time.monotonic_ns(),
        "send_wall_ns": time.time_ns(),
        "message_hex": message.hex(" "),
    }
    arm.send_ee_pass_through_message(message)
    return event


def _response_data_bytes(resp: object) -> Optional[bytes]:
    if not isinstance(resp, dict) or "data" not in resp:
        return None
    data = resp["data"]
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, (list, tuple)):
        try:
            return bytes(data)
        except ValueError:
            return None
    return None


def is_hande_write_ack(raw: bytes) -> bool:
    """Return True for a valid Robotiq FC16 write ACK."""
    if len(raw) != 8:
        return False
    if raw[0] != SLAVE_ID or raw[1] != FC16_WRITE_MULTIPLE_REGISTERS:
        return False
    return modbus_crc(raw[:-2]) == raw[-2:]


def classify_hande_response(
    raw: bytes,
    function_code: int = 0x03,
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Classify one EE pass-through response payload."""
    if is_hande_write_ack(raw):
        return "fc16_ack", None
    parsed = parse_hande_status_response(raw)
    if parsed is None:
        return "invalid", None
    if parsed["function_code"] != function_code:
        return "other_status", parsed
    return "status", parsed


class RobotiqStatusMonitor:
    """Queue and demultiplex Robotiq status replies from the mixed EE stream.

    Dexcontrol's public `Arm.get_ee_pass_through_response()` is a single latest
    response slot. This monitor adds a second subscriber on the same
    `state/ee_pass_through/{side}` topic and queues every callback, filtering
    FC03/FC04 status replies away from FC16 write ACKs.
    """

    def __init__(
        self,
        arm: Arm,
        side: Optional[str] = None,
        function_code: int = 0x03,
        queue_size: int = 512,
    ) -> None:
        if function_code not in (0x03, 0x04):
            raise ValueError(f"function_code must be 0x03 or 0x04; got {function_code:#x}")
        if queue_size <= 0:
            raise ValueError(f"queue_size must be positive; got {queue_size}")

        self.arm = arm
        self.side = side or getattr(arm, "_side", None)
        if not self.side:
            raise ValueError("RobotiqStatusMonitor needs side='left' or side='right'")
        self.function_code = function_code
        self.queue_size = queue_size
        self.topic = f"state/ee_pass_through/{self.side}"
        self._lock = threading.Lock()
        self._statuses: Deque[Dict[str, Any]] = deque(maxlen=queue_size)
        self._pending_requests: Deque[Dict[str, Any]] = deque()
        self._counters: Counter[str] = Counter()
        self._next_request_id = 0
        node = getattr(arm, "_node", None)
        if node is None:
            raise ValueError("arm has no dexcomm node")
        self._subscriber = node.create_subscriber(
            topic=self.topic,
            callback=self.handle_response,
            decoder=EEPassThroughCmdCodec.decode,
            buffer_size=queue_size,
            channel_type="ring",
        )

    def send_status_request(self) -> Dict[str, Any]:
        """Send one FC03/FC04 status request and register it as pending."""
        message = build_hande_status_request(self.function_code)
        event = {
            "status_request_id": self._next_request_id,
            "function_code": self.function_code,
            "send_monotonic_ns": time.monotonic_ns(),
            "send_wall_ns": time.time_ns(),
            "message_hex": message.hex(" "),
        }
        with self._lock:
            self._next_request_id += 1
            self._pending_requests.append(event)
            self._counters["status_requests"] += 1
        self.arm.send_ee_pass_through_message(message)
        return event

    def handle_response(self, resp: object) -> None:
        """Dexcomm callback: classify and queue one EE pass-through response."""
        read_monotonic_ns = time.monotonic_ns()
        read_wall_ns = time.time_ns()
        raw = _response_data_bytes(resp)
        if raw is None:
            with self._lock:
                self._counters["missing_data_discarded"] += 1
            return

        kind, parsed = classify_hande_response(raw, self.function_code)
        with self._lock:
            if kind == "fc16_ack":
                self._counters["fc16_acks_discarded"] += 1
                return
            if kind == "invalid":
                self._counters["invalid_discarded"] += 1
                return
            if kind == "other_status":
                self._counters["other_status_discarded"] += 1
                return

            request_event = self._pending_requests.popleft() if self._pending_requests else None
            if request_event is None:
                self._counters["unmatched_statuses"] += 1
            parsed = dict(parsed or {})
            parsed["response_timestamp_ns"] = (
                resp.get("timestamp_ns") if isinstance(resp, dict) else None
            )
            parsed["response_sequence"] = resp.get("sequence") if isinstance(resp, dict) else None
            parsed["read_monotonic_ns"] = read_monotonic_ns
            parsed["read_wall_ns"] = read_wall_ns
            if request_event is not None:
                parsed["status_request_id"] = request_event["status_request_id"]
                parsed["request_send_monotonic_ns"] = request_event["send_monotonic_ns"]
                parsed["request_send_wall_ns"] = request_event["send_wall_ns"]
                parsed["status_request_hex"] = request_event["message_hex"]
            if len(self._statuses) == self.queue_size:
                self._counters["status_queue_overflow"] += 1
            self._statuses.append(parsed)
            self._counters["statuses_received"] += 1

    def drain_status_events(self, max_events: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return queued status events and remove them from the monitor."""
        with self._lock:
            n = len(self._statuses) if max_events is None else min(max_events, len(self._statuses))
            return [self._statuses.popleft() for _ in range(n)]

    def expire_timeouts(self, timeout_s: float) -> List[Dict[str, Any]]:
        """Expire pending status requests older than `timeout_s`."""
        if timeout_s < 0:
            raise ValueError(f"timeout_s must be nonnegative; got {timeout_s}")
        now_ns = time.monotonic_ns()
        timeout_ns = int(timeout_s * 1e9)
        expired: List[Dict[str, Any]] = []
        with self._lock:
            while self._pending_requests:
                event = self._pending_requests[0]
                if now_ns - int(event["send_monotonic_ns"]) < timeout_ns:
                    break
                event = self._pending_requests.popleft()
                expired.append(
                    {
                        "status_request_id": event["status_request_id"],
                        "function_code": event["function_code"],
                        "send_monotonic_ns": event["send_monotonic_ns"],
                        "send_wall_ns": event["send_wall_ns"],
                        "message_hex": event["message_hex"],
                        "timeout_monotonic_ns": now_ns,
                        "timeout_wall_ns": time.time_ns(),
                    }
                )
            self._counters["status_timeouts"] += len(expired)
        return expired

    def stats(self) -> Dict[str, int]:
        """Return monitor counters plus current queue depths."""
        with self._lock:
            stats = dict(self._counters)
            stats["queued_statuses"] = len(self._statuses)
            stats["pending_status_requests"] = len(self._pending_requests)
        return stats

    def close(self) -> None:
        """Best-effort subscriber cleanup for dexcomm versions that expose it."""
        subscriber = self._subscriber
        if subscriber is None:
            return
        for method_name in ("shutdown", "close", "undeclare"):
            method = getattr(subscriber, method_name, None)
            if callable(method):
                method()
                break
        self._subscriber = None


def discard_ee_pass_through_response(
    arm: Arm,
    expected_response: Optional[dict[str, Any]] = None,
) -> bool:
    """Clear dexcontrol's latest EE pass-through response slot.

    `Arm.get_ee_pass_through_response()` exposes a last-value cache, not a
    queue. Clearing the slot after a read prevents the same FC03 status from
    being interpreted as the current gripper state on a later loop iteration.

    When `expected_response` is given, the clear is guarded by object identity
    so a newer callback that arrived after the caller's read is preserved.
    """
    lock = getattr(arm, "_ee_pass_through_lock", None)
    if lock is None or not hasattr(arm, "_latest_ee_pass_through_data"):
        return False

    with lock:
        current = getattr(arm, "_latest_ee_pass_through_data", None)
        if expected_response is not None and current is not expected_response:
            return False
        arm._latest_ee_pass_through_data = None  # noqa: SLF001 - consume dexcontrol slot
    return True


def send_activate(right_arm: Arm) -> None:
    # Clear/reset
    send_ee_pass_through_with_timestamps(right_arm, HANDE_RESET_COMMAND)
    time.sleep(0.05)

    # Activate
    send_ee_pass_through_with_timestamps(right_arm, HANDE_ACTIVATE_COMMAND)
    time.sleep(2.5)


def build_hande_status_request(function_code: int = 0x03) -> bytes:
    """Build a Modbus read request for the Robotiq Hand-E status registers.

    Reads 3 registers starting at 0x07D0 (gripper status, fault/PR, position/current).
    The reply is 11 bytes: SLAVE FC 0x06 gSTA reserved gFLT gPR gPO gCU CRC_lo CRC_hi.

    Args:
        function_code: 0x03 (read holding registers, the value Robotiq's manual
                       uses) or 0x04 (read input registers).
    """
    if function_code not in (0x03, 0x04):
        raise ValueError(f"function_code must be 0x03 or 0x04; got {function_code:#x}")
    payload = bytes(
        [
            SLAVE_ID,
            function_code,
            STATUS_REG_ADDR_HI,
            STATUS_REG_ADDR_LO,
            0x00,
            STATUS_REG_COUNT,
        ]
    )
    return payload + modbus_crc(payload)


def parse_hande_status_response(raw: bytes) -> Optional[Dict[str, Any]]:
    """Parse an 11-byte Robotiq Hand-E FC03/FC04 status reply.

    Validates length, slave ID, function code, byte count, and CRC. Returns
    None on any failure so callers can mark the sample invalid without
    crashing — in particular, FC16 write ACKs (8 bytes, function 0x10) and
    truncated buffers are rejected here.

    Returns a dict with raw register bytes plus convenience fields:
        actual:     gPO / 255.0   (encoder position in [0, 1])
        cmd_echo:   gPR / 255.0   (last position request seen by the gripper)
        current_ma: gCU * 10      (motor current; gCU is in 10 mA increments)
    Plus decoded status bits gACT, gGTO, gSTA_bits, gOBJ, and fault nibbles.
    """
    if not isinstance(raw, (bytes, bytearray)):
        return None
    raw = bytes(raw)
    if len(raw) != 11:
        return None
    if raw[0] != SLAVE_ID:
        return None
    if raw[1] not in (0x03, 0x04):
        return None
    if raw[2] != 0x06:
        return None
    if modbus_crc(raw[:-2]) != raw[-2:]:
        return None

    gSTA, reserved, gFLT_byte, gPR, gPO, gCU = raw[3:9]
    return {
        "function_code": raw[1],
        "gSTA": gSTA,
        "gACT": gSTA & 0x01,
        "gGTO": (gSTA >> 3) & 0x01,
        "gSTA_bits": (gSTA >> 4) & 0x03,
        "gOBJ": (gSTA >> 6) & 0x03,
        "reserved": reserved,
        "gFLT": gFLT_byte & 0x0F,
        "kFLT": (gFLT_byte >> 4) & 0x0F,
        "gPR": gPR,
        "gPO": gPO,
        "gCU": gCU,
        "cmd_echo": gPR / 255.0,
        "actual": gPO / 255.0,
        "current_ma": gCU * 10,
        "raw_hex": raw.hex(" "),
    }


def response_token(resp: object) -> object:
    """Compute a freshness token for an EE pass-through response slot.

    Returns the `(timestamp_ns, sequence)` pair when either is present, else
    falls back to the raw bytes. Identical status payloads with new metadata
    are treated as fresh so a stationary gripper still produces sample events.
    """
    if not isinstance(resp, dict):
        return None
    timestamp_ns = resp.get("timestamp_ns")
    sequence = resp.get("sequence")
    if timestamp_ns is not None or sequence is not None:
        return timestamp_ns, sequence
    if "data" in resp:
        return bytes(resp["data"])
    return None


def poll_gripper_status(
    arm: Arm,
    function_code: int = 0x03,
    timeout_s: float = 0.05,
    sleep_s: float = 0.001,
) -> Optional[Dict[str, Any]]:
    """Send a status request and poll for the matching reply.

    `Arm.get_ee_pass_through_response()` is last-write-wins (single slot, no
    queue) and the same channel delivers FC16 write ACKs from any concurrent
    `send_ee_pass_through_message` calls. This helper:

      1. Clears any already-consumed response from the response slot.
      2. Sends the FC03/FC04 status request.
      3. Polls until the slot has a fresh response, then tries to parse it.
         If the new payload is something else (e.g. an FC16 ACK from another
         caller), the helper discards it and keeps waiting.

    Suitable for one-shot uses (e.g. warm-up). For continuous logging use
    `RobotiqStatusMonitor`, which has a dedicated subscriber and keeps every
    status callback as a one-shot event.

    Args:
        arm: dexcontrol Arm whose EE pass-through is enabled.
        function_code: 0x03 (default) or 0x04.
        timeout_s: total wait budget for a fresh, parseable reply.
        sleep_s: poll interval between slot checks.

    Returns:
        Parsed status dict on success; None on timeout / parse failure /
        EE pass-through disabled.
    """
    discard_ee_pass_through_response(arm)

    request_event = send_ee_pass_through_with_timestamps(
        arm, build_hande_status_request(function_code)
    )
    before_token = None

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        read_monotonic_ns = time.monotonic_ns()
        read_wall_ns = time.time_ns()
        resp = arm.get_ee_pass_through_response()
        if not isinstance(resp, dict) or "data" not in resp:
            time.sleep(sleep_s)
            continue
        token = response_token(resp)
        if token == before_token:
            discard_ee_pass_through_response(arm, resp)
            time.sleep(sleep_s)
            continue
        data = bytes(resp["data"])
        parsed = parse_hande_status_response(data)
        if parsed is None or parsed["function_code"] != function_code:
            # Likely an FC16 write ACK or a stale fragment — advance the
            # snapshot and wait for the actual status reply.
            before_token = token
            discard_ee_pass_through_response(arm, resp)
            continue
        parsed["response_timestamp_ns"] = resp.get("timestamp_ns")
        parsed["response_sequence"] = resp.get("sequence")
        parsed["request_send_monotonic_ns"] = request_event["send_monotonic_ns"]
        parsed["request_send_wall_ns"] = request_event["send_wall_ns"]
        parsed["read_monotonic_ns"] = read_monotonic_ns
        parsed["read_wall_ns"] = read_wall_ns
        discard_ee_pass_through_response(arm, resp)
        return parsed
    return None
