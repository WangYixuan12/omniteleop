import time

from dexcontrol.robot import Arm, Robot

SLAVE_ID = 0x09


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

    rPR = int(round(pos * 255))  # position request
    rSP = int(round(speed * 255))  # speed
    rFR = int(round(force * 255))  # force

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
    time.sleep(0.5)


def move_gripper(right_arm: Arm, pos: float, speed: float = 1.0, force: float = 1.0) -> None:
    cmd = build_hande_command(pos=pos, speed=speed, force=force)
    print("Sending:", cmd.hex(" "))
    right_arm.send_ee_pass_through_message(cmd)


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
