from .base_task import SimTask, ExpertCommand
from .carry_task import CarryTask
from .pick_place_task import PickPlaceTask
from .mobile_pick_place_task import MobilePickPlaceTask

TASKS = {t.name: t for t in [CarryTask, PickPlaceTask, MobilePickPlaceTask]}
__all__ = ["SimTask", "ExpertCommand", "CarryTask", "PickPlaceTask", "MobilePickPlaceTask", "TASKS"]
