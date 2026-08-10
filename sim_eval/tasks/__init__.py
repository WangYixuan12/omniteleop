from .base_task import SimTask, ExpertCommand
from .carry_task import CarryTask
from .pick_place_task import PickPlaceTask
from .mobile_pick_place_task import MobilePickPlaceTask
from .long_horizon import LongHorizonPickPlace, LongHorizonSequence, LHSpec, LH_TASKS
from .nonprehensile import CloseFridgeTask, PushChairTask, NP_TASKS

TASKS = {t.name: t for t in [CarryTask, PickPlaceTask, MobilePickPlaceTask, *LH_TASKS,
                             LongHorizonSequence, *NP_TASKS]}
#: the long-horizon suite mirroring the real-robot plan (see long_horizon / nonprehensile).
#: `LongHorizonSequence` is deliberately NOT here: it re-runs the same three tasks in one rollout,
#: so counting it would double every aggregate score.
LONG_HORIZON = [t.name for t in (*LH_TASKS, *NP_TASKS)]
__all__ = ["SimTask", "ExpertCommand", "CarryTask", "PickPlaceTask", "MobilePickPlaceTask",
           "LongHorizonPickPlace", "LongHorizonSequence", "LHSpec", "CloseFridgeTask",
           "PushChairTask", "TASKS", "LONG_HORIZON"]
