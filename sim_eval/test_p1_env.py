"""P1 test: VegaOGEnv control loop reaches AND holds an EEF target (closed-loop @100Hz)."""
import os
os.environ["OMNIGIBSON_HEADLESS"] = "1"
import numpy as np
from vega_og_env import VegaOGEnv

P = lambda *a: print(*a, flush=True)
np.set_printoptions(precision=4, suppress=True)

env = VegaOGEnv(task=None, action_hz=100, physics_hz=200, pos_kp=4000, lock_base=True, wbc_port=5611)
env.reset()
L0, R0 = env.link_pose("L_ee"), env.link_pose("R_ee")
head0 = env.link_pose("zed_depth_frame")   # hold head at its nominal OG pose throughout
Ltgt = L0.copy(); Ltgt[:3, 3] += np.array([0.05, 0.08, -0.06])
Rtgt = R0.copy(); Rtgt[:3, 3] += np.array([0.05, -0.08, -0.06])
P("nominal L_ee", L0[:3, 3], "target", Ltgt[:3, 3])
for i in range(200):
    resp = env.wbc_tick(Ltgt, Rtgt, head_og=head0, grip_l=-1.0, grip_r=-1.0)
    if i % 20 == 0 or i == 199:
        le, re = env.link_pose("L_ee")[:3, 3], env.link_pose("R_ee")[:3, 3]
        P(f"  tick {i:3d}: L_ee err={np.linalg.norm(le-Ltgt[:3,3]):.4f} "
          f"R_ee err={np.linalg.norm(re-Rtgt[:3,3]):.4f} wbc l_err={resp['l_err']:.4f}")
le, re = env.link_pose("L_ee")[:3, 3], env.link_pose("R_ee")[:3, 3]
lerr, rerr = np.linalg.norm(le - Ltgt[:3, 3]), np.linalg.norm(re - Rtgt[:3, 3])
P(f"FINAL L_ee err={lerr:.4f} R_ee err={rerr:.4f} -> "
  f"{'PASS (closed-loop reaches+holds)' if lerr < 0.02 and rerr < 0.02 else 'CHECK gains/rate'}")
env.close()
