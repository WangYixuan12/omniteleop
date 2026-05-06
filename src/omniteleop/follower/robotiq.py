import time
from typing import Any, Dict, Optional

from dexcontrol.core.arm import Arm
from dexcontrol.robot import Robot

SLAVE_ID = 0x09

# Robotiq Hand-E status registers: 0x07D0..0x07D2 (3 registers, 6 bytes).
STATUS_REG_ADDR_HI = 0x07
STATUS_REG_ADDR_LO = 0xD0
STATUS_REG_COUNT = 3


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


def send_activate(right_arm: Arm) -> None:
    # Clear/reset
    cmd = bytes.fromhex("09 10 03 E8 00 03 06 00 00 00 00 00 00 73 30")
    right_arm.send_ee_pass_through_message(cmd)
    time.sleep(0.05)

    # Activate
    cmd = bytes.fromhex("09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1")
    right_arm.send_ee_pass_through_message(cmd)
    time.sleep(2.5)


def move_gripper(right_arm: Arm, pos: float, speed: float = 1.0, force: float = 1.0) -> None:
    cmd = build_hande_command(pos=pos, speed=speed, force=force)
    print("Sending:", cmd.hex(" "))
    right_arm.send_ee_pass_through_message(cmd)


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


def read_gripper_status_event(
    arm: Arm,
    last_token: object,
    function_code: int = 0x03,
) -> tuple[object, Optional[Dict[str, Any]], bool]:
    """Nonblocking single-shot read of the EE pass-through response slot.

    Returns `(new_token, parsed, advanced)`:
      - `new_token` is `last_token` if nothing new has arrived, else the
        freshly computed token (caller must persist it).
      - `parsed` is a status dict only when the new payload parses as the
        requested function code; FC16 ACKs and unparseable payloads return
        `None` here even though the token advanced.
      - `advanced` is True iff the token differs from `last_token`.

    The intended usage is a high-frequency drain loop: call repeatedly per
    side, persist `new_token`, append a sample row whenever `parsed` is not
    None. No blocking, no timeout, no synthetic NaN rows.
    """
    resp = arm.get_ee_pass_through_response()
    new_token = response_token(resp)
    if new_token == last_token or new_token is None:
        return last_token, None, False
    if not isinstance(resp, dict) or "data" not in resp:
        return new_token, None, True
    parsed = parse_hande_status_response(bytes(resp["data"]))
    if parsed is None or parsed["function_code"] != function_code:
        return new_token, None, True
    parsed["response_timestamp_ns"] = resp.get("timestamp_ns")
    parsed["response_sequence"] = resp.get("sequence")
    return new_token, parsed, True


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

      1. Snapshots the current contents of the response slot.
      2. Sends the FC03/FC04 status request.
      3. Polls until the slot has a fresh response, then tries to parse it.
         If the new payload is something else (e.g. an FC16 ACK from another
         caller), the helper updates its snapshot and keeps waiting.

    Suitable for one-shot uses (e.g. warm-up). For continuous logging use
    `read_gripper_status_event` in a drain loop instead — this helper's
    timeout-and-discard semantics drop late replies.

    Args:
        arm: dexcontrol Arm whose EE pass-through is enabled.
        function_code: 0x03 (default) or 0x04.
        timeout_s: total wait budget for a fresh, parseable reply.
        sleep_s: poll interval between slot checks.

    Returns:
        Parsed status dict on success; None on timeout / parse failure /
        EE pass-through disabled.
    """
    before_token = response_token(arm.get_ee_pass_through_response())

    arm.send_ee_pass_through_message(build_hande_status_request(function_code))

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = arm.get_ee_pass_through_response()
        if not isinstance(resp, dict) or "data" not in resp:
            time.sleep(sleep_s)
            continue
        token = response_token(resp)
        if token == before_token:
            time.sleep(sleep_s)
            continue
        data = bytes(resp["data"])
        parsed = parse_hande_status_response(data)
        if parsed is None or parsed["function_code"] != function_code:
            # Likely an FC16 write ACK or a stale fragment — advance the
            # snapshot and wait for the actual status reply.
            before_token = token
            continue
        parsed["response_timestamp_ns"] = resp.get("timestamp_ns")
        parsed["response_sequence"] = resp.get("sequence")
        return parsed
    return None

def main() -> None:
    bot = Robot()
    right_arm = bot.right_arm

    send_activate(right_arm)

    while True:
        print("Open -> 0.0")
        move_gripper(right_arm, pos=0.0, speed=1.0, force=1.0)
        time.sleep(1.0)

        print("Half close -> 0.5")
        move_gripper(right_arm, pos=0.5, speed=1.0, force=1.0)
        time.sleep(1.0)

        print("Close -> 1.0")
        move_gripper(right_arm, pos=1.0, speed=1.0, force=1.0)
        time.sleep(1.0)

    bot.shutdown()


if __name__ == "__main__":
    main()
