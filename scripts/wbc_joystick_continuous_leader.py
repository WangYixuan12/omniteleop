#!/usr/bin/env python3
# ruff: noqa: SLF001  (reuses the loaded joystick leader's private helpers by design)
"""Joystick VR leader for continuous collection: left X ends a take, teleop keeps running.

Pairs with ``scripts/wbc_joystick_continuous_robot.py``. Calibration, the fixed-base-frame
hand mapping, the joystick base, the headset HUD, ``--align-reference`` and every flag are
``scripts/wbc_joystick_leader.py``; only the episode buttons differ:

  * hold RIGHT grip (``--hold-seconds``) in ``static`` -> calibrate -> (``align`` with
    ``--align-reference``) -> ``teleop``: the follower records the first take, as today;
  * left X in ``teleop`` -> ``teleop_idle``: the follower ends the take and saves it in the
    background while teleop continues (the stage shows on the headset HUD);
  * hold RIGHT grip in ``teleop_idle`` -> ``teleop``: the next take records from wherever
    the robot is -- no re-calibration, no align gate, no reset;
  * left X in ``teleop_idle`` is ignored, so left Y is the only way out of teleop. It stops
    both programs; a take still recording is saved.

Before the first take, left X (back to ``static``) and the left-trigger home request work
as in the stock leader. The stock joystick follower only ends a take on an e-stop, so it
would keep recording through ``teleop_idle``: run the continuous follower with this leader.

Run in the dexmate conda env, with the joystick leader's flags::

    python scripts/wbc_joystick_continuous_leader.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Load the joystick leader as a module (scripts/ is not a package), the way it loads the
# VR leader. Its main() is __main__-guarded, so this only binds its definitions.
_JOY_PATH = Path(__file__).resolve().parent / "wbc_joystick_leader.py"
_spec = importlib.util.spec_from_file_location("wbc_joystick_leader", _JOY_PATH)
_joy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_joy)

JoystickVRLeader = _joy.JoystickVRLeader
_button_rising_edge = _joy._base._button_rising_edge

IDLE_STAGE = "teleop_idle"  # teleop keeps streaming; the follower is not recording
_LOG = "[wbc_joystick_continuous_leader]"


class ContinuousJoystickVRLeader(JoystickVRLeader):
    """``JoystickVRLeader`` whose left X ends a take without leaving teleop."""

    # teleop_idle streams targets to an engaged (non-e-stop) follower, exactly like teleop.
    _TELEOP_STAGES = ("teleop", IDLE_STAGE)

    def _handle_episode_buttons(self, transforms) -> bool:
        """Left X ends a take; a RIGHT-grip hold starts the next one.

        Outside teleop (static / align) left X keeps its stock meaning: back to ``static``
        (e-stop). Inside teleop the tick is never consumed, so targets keep streaming.
        """
        if self.stage not in self._TELEOP_STAGES:
            return super()._handle_episode_buttons(transforms)
        x_now = bool(transforms["left_x_button"])
        x_pressed = _button_rising_edge(now=x_now, prev=self._prev_x)
        self._prev_x = x_now
        if self.stage == "teleop":
            if x_pressed:
                self.stage = IDLE_STAGE
                # As after the stock X: a grip still held from the last start must be
                # released before a new hold can start the next take.
                self._trigger_start = None
                self._trigger_requires_release = True
                print(f"\n{_LOG} episode ended -> {IDLE_STAGE} (the follower saves it; "
                      "teleop continues). Hold RIGHT grip to record the next episode.")
        elif self._trigger_held(transforms):  # teleop_idle: X is ignored, Y still exits
            self.stage = "teleop"
            print(f"\n{_LOG} RIGHT grip held -> teleop: recording the next episode.")
        return False


def main() -> None:
    # Reuse the joystick leader's CLI verbatim, with this class and help text. _joy is
    # this script's private instance of that module; wbc_joystick_leader.py is unchanged.
    _joy.__doc__ = __doc__
    _joy.JoystickVRLeader = ContinuousJoystickVRLeader
    _joy.main()


if __name__ == "__main__":
    main()
