import time

from dexmotion.motion_manager import MotionManager


def main():
    # Initialize with locked joint regions
    mm = MotionManager(
        joint_regions_to_lock=[],
    )

    # Print robot type
    print(f"Robot type: {mm.robot_type}")

    # Print joint positions
    joint_pos_dict = mm.get_joint_pos_dict()
    print(f"Number of active joints: {len(joint_pos_dict)}")
    print(f"Joint positions: {joint_pos_dict}")

    while True:
        time.sleep(0.01)
        mm.


if __name__ == "__main__":
    main()
