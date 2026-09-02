"""Follower modules for robot control and safety processing."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .command_processor import CommandProcessor as CommandProcessor
    from .robot_controller import RobotController as RobotController

# Resolved on first attribute access so that importing a sibling submodule (e.g.
# ``omniteleop.follower.whole_body_ik``) does not drag in ``dexmotion`` via
# ``command_processor``.
_LAZY_ATTRS = {
    "CommandProcessor": (".command_processor", "CommandProcessor"),
    "command_processor_main": (".command_processor", "main"),
    "RobotController": (".robot_controller", "RobotController"),
    "robot_controller_main": (".robot_controller", "main"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    return getattr(import_module(module_name, __name__), attr)


def __dir__():
    return sorted(__all__)


__all__ = [
    "CommandProcessor",
    "command_processor_main",
    "RobotController",
    "robot_controller_main",
]
