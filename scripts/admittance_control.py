# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Admittance control with pose-dependent wrist force/torque gravity compensation.

The wrist force/torque (F/T) sensors read the weight of everything mounted past
them (gripper + payload) plus a constant electrical bias. As the end-effector
re-orients, the gravity component rotates within the sensor frame, so a single
constant offset captured at startup is only correct at that one pose. This
script instead subtracts a *pose-dependent* gravity + bias estimate computed
from a calibration that is fit once and reloaded on every run.

Two subcommands::

    # 1) One-time calibration (moves both wrists through a set of poses, fits
    #    the gravity + bias model, and saves it). Re-run if the payload changes.
    python /home/dexmate/yixuan/omniteleop_yifan/scripts/admittance_control.py calibrate
    # higher wrist_delta leads to lower cond is better
    # (wrist_delta=0.30 for table, 0.60 for free space)

    # 2) Admittance control, loading the saved calibration.
    python /home/dexmate/yixuan/omniteleop_yifan/scripts/admittance_control.py run

Calibration model (fit per arm, in the raw sensor frame, using only FK so the
sensor's mount rotation never has to be known analytically):

    u(q)   = R_world<-ee(q).T @ g_hat        # gravity direction in the EE frame
    f_raw  = A @ u(q) + f_bias               # A absorbs mass * g * mount rotation
    tau_raw = B @ u(q) + tau_bias            # B absorbs the COM lever arm

Each row is a linear least-squares fit (12 unknowns, 3 equations per pose), so
it both rotates gravity with the wrist *and* separates the constant sensor bias
from gravity — neither of which a single startup reading can do.

Assumptions / caveats:
- The FK frames are expressed in the robot base frame and gravity is taken as
  base ``-Z`` (g_hat = [0, 0, -1]); this codebase is Z-up. A *constant* base
  tilt cancels (the same g_hat is used at calibration and run time); a base tilt
  that changes between calibration and run (e.g. a mobile base on a slope) is
  not accounted for.
- The calibration is specific to the mounted payload. Re-calibrate after
  changing the gripper or attaching/removing a tool.

For best performance, run on a Jetson or PC wired to the robot via ethernet, as
admittance control is sensitive to network latency.
"""

import time
from pathlib import Path

import numpy as np
from dexcomm import RateLimiter
from dexcontrol.core.arm import Arm
from dexcontrol.robot import Robot
from dexmotion.ik import LocalPinkIKSolver
from dexmotion.motion_manager import MotionManager
from loguru import logger
from tyro.extras import subcommand_cli_from_dict

# Run-time admittance law + gravity compensation live in the package so the
# calibration model fit here and the prediction applied by consumers (e.g.
# follower/vr_robot_controller.py) stay byte-for-byte identical. The fitting
# (fit_gravity_model / save_calibration) and the CLI stay below.
from omniteleop.follower.admittance import (
    DEFAULT_CALIB_PATH,
    GRAVITY_DIR,
    AdmittanceController,
    gravity_direction_in_ee,
    load_calibration,
    predict_gravity_wrench,
    preprocess_wrench,
)


def fit_gravity_model(
    u: np.ndarray, wrench: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Least-squares fit of the gravity + bias model from calibration samples.

    Solves, independently for force and torque, ``w = M @ u + bias`` where each
    is linear in the unknowns (M is 3x3, bias is 3,). See the module docstring
    for the derivation.

    Args:
        u: ``(N, 3)`` gravity-direction-in-EE samples (one per pose).
        wrench: ``(N, 6)`` mean raw wrench readings aligned with ``u``.

    Returns:
        Tuple of (model, diagnostics). ``model`` has keys ``A``, ``f_bias``,
        ``B``, ``tau_bias``; ``diagnostics`` has ``n``, ``cond``, ``force_rms``,
        ``torque_rms`` (residual RMS in N and Nm).

    Raises:
        ValueError: On shape mismatch or fewer than 4 poses.
    """
    u = np.asarray(u, dtype=float)
    wrench = np.asarray(wrench, dtype=float)
    if u.ndim != 2 or u.shape[1] != 3:
        raise ValueError(f"u must be (N, 3), got {u.shape}")
    if wrench.shape != (u.shape[0], 6):
        raise ValueError(
            f"wrench must be (N, 6) aligned with u {u.shape}, got {wrench.shape}"
        )
    n = u.shape[0]
    if n < 4:
        raise ValueError(f"need at least 4 calibration poses, got {n}")

    # Design matrix [u | 1]; columns 0:3 multiply the 3x3 matrix rows, column 3
    # is the per-axis bias.
    design = np.hstack([u, np.ones((n, 1))])  # (n, 4)
    cond = float(np.linalg.cond(design))

    pf, *_ = np.linalg.lstsq(design, wrench[:, :3], rcond=None)  # (4, 3)
    pt_, *_ = np.linalg.lstsq(design, wrench[:, 3:], rcond=None)  # (4, 3)

    # params[j, k] maps to row k of M (M[k, j]); params[3, k] is bias[k].
    model = {
        "A": pf[:3, :].T.copy(),
        "f_bias": pf[3, :].copy(),
        "B": pt_[:3, :].T.copy(),
        "tau_bias": pt_[3, :].copy(),
    }
    pred = np.hstack([design @ pf, design @ pt_])  # (n, 6)
    resid = wrench - pred
    diagnostics = {
        "n": n,
        "cond": cond,
        "force_rms": float(np.sqrt(np.mean(resid[:, :3] ** 2))),
        "torque_rms": float(np.sqrt(np.mean(resid[:, 3:] ** 2))),
    }
    return model, diagnostics


def save_calibration(path: Path, models: dict[str, dict[str, np.ndarray]]) -> None:
    """Save per-arm gravity calibration to a ``.npz`` file.

    Args:
        path: Destination path (parent directories are created).
        models: Mapping ``{"left": model, "right": model}`` (see
            :func:`fit_gravity_model`).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for arm in ("left", "right"):
        m = models[arm]
        arrays[f"{arm}_A"] = m["A"]
        arrays[f"{arm}_f_bias"] = m["f_bias"]
        arrays[f"{arm}_B"] = m["B"]
        arrays[f"{arm}_tau_bias"] = m["tau_bias"]
    arrays["gravity_dir"] = GRAVITY_DIR
    arrays["created_unix"] = np.array(time.time())
    np.savez(path, **arrays)


def _log_calibration_file_health(path: Path) -> None:
    """Log a compact health summary for a saved calibration file."""
    path = Path(path)
    if not path.exists():
        logger.error(f"Calibration file does not exist: {path}")
        return

    data = np.load(path)
    logger.info(f"Calibration file: {path}")
    if "created_unix" in data:
        created = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(float(data["created_unix"]))
        )
        logger.info(f"Calibration created: {created}")

    for arm in ("left", "right"):
        norms: list[float] = []
        missing: list[str] = []
        for key in ("A", "f_bias", "B", "tau_bias"):
            array_key = f"{arm}_{key}"
            if array_key not in data:
                missing.append(array_key)
                continue
            arr = np.asarray(data[array_key], dtype=float)
            norms.append(float(np.linalg.norm(arr)))
        if missing:
            logger.error(f"[{arm}] calibration missing arrays: {missing}")
            continue
        all_zero = max(norms) <= 1e-6
        log = logger.error if all_zero else logger.info
        log(
            f"[{arm}] calibration all_zero={all_zero} "
            f"A={norms[0]:.4g} f_bias={norms[1]:.4g} "
            f"B={norms[2]:.4g} tau_bias={norms[3]:.4g}"
        )


def _initialize_robot_and_motion_manager(
    bot: Robot,
) -> tuple[MotionManager, LocalPinkIKSolver]:
    """Initialize motion manager and IK solver.

    Args:
        bot: Robot instance.

    Returns:
        Tuple of (motion_manager, ik_solver).

    Raises:
        ValueError: If initialization fails.
    """
    if bot.has_component("torso"):
        qpos_dict = bot.get_joint_pos_dict(
            component=["head", "left_arm", "right_arm", "torso"]
        )
    else:
        qpos_dict = bot.get_joint_pos_dict(component=["head", "left_arm", "right_arm"])

    motion_manager = MotionManager(
        init_visualizer=False,
        initial_joint_configuration_dict=qpos_dict,
    )

    if motion_manager.pin_robot is None:
        raise ValueError("Motion Manager is not initialized")

    ik_solver = motion_manager.local_ik_solver
    if not isinstance(ik_solver, LocalPinkIKSolver):
        raise ValueError("Local IK solver is not initialized")

    return motion_manager, ik_solver


def _get_initial_ee_poses(motion_manager: MotionManager) -> dict[str, np.ndarray]:
    """Get initial end-effector poses.

    Args:
        motion_manager: Initialized motion manager.

    Returns:
        Dictionary mapping arm names to initial poses.
    """
    fk_result = motion_manager.fk(
        frame_names=motion_manager.target_frames,
        qpos=motion_manager.get_joint_pos(),
    )

    return {
        "left": fk_result["L_ee"].np,  # type: ignore
        "right": fk_result["R_ee"].np,  # type: ignore
    }


def _update_ee_poses_with_admittance(
    ee_pose: dict[str, np.ndarray],
    wrench_states: dict[str, dict],
    gravity_models: dict[str, dict[str, np.ndarray]],
    admittance_controller: AdmittanceController,
    init_ee_pose: dict[str, np.ndarray],
    zero_force: bool,
) -> None:
    """Update end-effector poses using admittance control.

    Args:
        ee_pose: Dictionary of current end-effector poses (modified in-place).
        wrench_states: Dictionary of wrench sensor states.
        gravity_models: Per-arm gravity + bias calibration models.
        admittance_controller: Admittance controller instance.
        init_ee_pose: Dictionary of initial end-effector poses.
        zero_force: Whether to use zero-force mode.
    """
    for arm in ("left", "right"):
        # Compute the gravity + bias wrench from the *measured* pose before it is
        # overwritten with the admittance-corrected pose below.
        grav_wrench = predict_gravity_wrench(gravity_models[arm], ee_pose[arm])
        wrench = preprocess_wrench(wrench_states[arm]["wrench"], grav_wrench, arm)
        new_pose = admittance_controller.get_admittance_pose(
            pose_cur=ee_pose[arm],
            wrench_ext=wrench,
            pose_des=None if zero_force else init_ee_pose[arm],
        )
        ee_pose[arm] = new_pose


def _send_joint_commands(
    target_qpos_dict: dict[str, float],
    wrench_states: dict[str, dict],
    arms: dict[str, Arm],
    need_button: bool,
) -> None:
    """Send joint position commands to robot arms.

    Args:
        target_qpos_dict: Dictionary of target joint positions.
        wrench_states: Dictionary of wrench sensor states.
        arms: Dictionary of arm objects.
        need_button: Whether button press is required for activation.
    """
    for arm in ("left", "right"):
        if wrench_states[arm]["blue_button"] or not need_button:
            prefix = "L" if arm == "left" else "R"
            target_qpos = [
                v for k, v in target_qpos_dict.items() if f"{prefix}_arm" in k
            ]
            arms[arm].set_joint_pos(target_qpos)


# ── Calibration ──────────────────────────────────────────────────────────────


def _wrist_delta_grid(delta: float) -> list[tuple[float, float, float]]:
    """Build a spread of wrist-joint (j5, j6, j7) offsets for calibration.

    Wrist-only perturbations re-orient the sensor (which excites the gravity
    model) while keeping the EE roughly in place (minimal collision risk).

    Args:
        delta: Offset magnitude in radians applied to wrist joints.

    Returns:
        List of (dj5, dj6, dj7) offset tuples, including the zero pose.
    """
    d = float(delta)
    return [
        (0.0, 0.0, 0.0),
        (+d, 0.0, 0.0),
        (-d, 0.0, 0.0),
        (0.0, +d, 0.0),
        (0.0, -d, 0.0),
        (0.0, 0.0, +d),
        (0.0, 0.0, -d),
        (+d, +d, 0.0),
        (-d, -d, 0.0),
        (+d, 0.0, +d),
        (-d, 0.0, -d),
        (0.0, +d, +d),
        (0.0, -d, -d),
        (+d, -d, +d),
        (-d, +d, -d),
    ]


def _interpolate_arms_to(
    arms: dict[str, Arm],
    targets: dict[str, np.ndarray],
    joint_delta: float = 0.02,
    wait_time: float = 0.05,
) -> None:
    """Smoothly move both arms to target joint configurations (blocking).

    Args:
        arms: Dictionary of arm objects.
        targets: Per-arm target joint positions (length-7 arrays).
        joint_delta: Max per-step joint increment (rad) for interpolation.
        wait_time: Per-step blocking time passed to ``set_joint_pos``.
    """
    for arm in ("left", "right"):
        target = np.asarray(targets[arm], dtype=float)
        current = np.asarray(arms[arm].get_joint_pos(), dtype=float)
        steps = max(1, int(np.max(np.abs(target - current)) / joint_delta))
        for i in range(steps):
            interp = current + (target - current) * (i + 1) / steps
            arms[arm].set_joint_pos(
                interp.tolist(), wait_time=wait_time, exit_on_reach=True
            )


def calibrate(
    calib_path: Path = DEFAULT_CALIB_PATH,
    wrist_delta: float = 0.60,
    settle_time: float = 0.8,
    samples_per_pose: int = 60,
    sample_dt: float = 0.005,
    yes: bool = False,
) -> None:
    """Calibrate and save the per-arm F/T gravity + bias model.

    Moves both wrists through a spread of orientations (offsets on joints 5-7
    from the current configuration), records the mean wrench and the gravity
    direction in the EE frame at each, fits the model (see module docstring),
    and saves it for :func:`run` to load.

    The arms move automatically — keep the workspace clear and be ready to
    e-stop. The model is specific to the mounted payload; re-run after changing
    the gripper or tool.

    Args:
        calib_path: Output path for the saved calibration ``.npz``.
        wrist_delta: Wrist-joint offset magnitude (rad) for the pose sweep.
        settle_time: Seconds to wait after reaching each pose before sampling.
        samples_per_pose: Number of wrench samples averaged at each pose.
        sample_dt: Delay (s) between wrench samples.
        yes: Skip the interactive motion confirmation prompt.
    """
    bot = Robot()

    if bot.left_arm.wrench_sensor is None or bot.right_arm.wrench_sensor is None:
        raise ValueError(
            "Calibration requires both left and right wrench sensors to be present."
        )

    motion_manager, _ = _initialize_robot_and_motion_manager(bot)
    arms = {"left": bot.left_arm, "right": bot.right_arm}
    arm_ee_name = {"left": "L_ee", "right": "R_ee"}

    start = {
        "left": np.asarray(bot.left_arm.get_joint_pos(), dtype=float),
        "right": np.asarray(bot.right_arm.get_joint_pos(), dtype=float),
    }
    for arm in ("left", "right"):
        if start[arm].shape != (7,):
            raise ValueError(
                f"Expected 7 joints for {arm} arm, got shape {start[arm].shape}"
            )

    deltas = _wrist_delta_grid(wrist_delta)

    if not yes:
        resp = input(
            f"Calibration will move BOTH wrists through {len(deltas)} poses "
            f"(+/-{wrist_delta} rad on joints 5-7). Clear the workspace. "
            f"Type 'yes' to proceed: "
        )
        if resp.strip().lower() not in ("y", "yes"):
            logger.info("Calibration aborted by user.")
            bot.shutdown()
            return

    u_samples: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    w_samples: dict[str, list[np.ndarray]] = {"left": [], "right": []}

    try:
        for idx, delta in enumerate(deltas):
            targets = {arm: start[arm].copy() for arm in ("left", "right")}
            for arm in ("left", "right"):
                targets[arm][4:7] += np.asarray(delta, dtype=float)
            logger.info(f"Pose {idx + 1}/{len(deltas)}: wrist delta {delta}")
            _interpolate_arms_to(arms, targets)
            time.sleep(settle_time)

            # FK from the ACTUAL reached configuration (robust to joint-limit
            # clamping or partial reach: the (u, wrench) pair is valid either way).
            actual = bot.get_joint_pos_dict(component=["left_arm", "right_arm"])
            motion_manager.set_joint_pos(actual)
            fk_result = motion_manager.fk(
                frame_names=motion_manager.target_frames,
                qpos=motion_manager.get_joint_pos(),
            )

            wrench_sum = {"left": np.zeros(6), "right": np.zeros(6)}
            for _ in range(samples_per_pose):
                for arm in ("left", "right"):
                    wrench_sum[arm] += arms[arm].wrench_sensor.get_wrench_state()
                time.sleep(sample_dt)

            for arm in ("left", "right"):
                pose = np.asarray(fk_result[arm_ee_name[arm]].np, dtype=float)
                u_samples[arm].append(gravity_direction_in_ee(pose))
                w_samples[arm].append(wrench_sum[arm] / samples_per_pose)

        logger.info("Returning arms to starting configuration ...")
        _interpolate_arms_to(arms, start)

        models: dict[str, dict[str, np.ndarray]] = {}
        for arm in ("left", "right"):
            wrench_array = np.array(w_samples[arm])
            if np.allclose(wrench_array, 0.0, atol=1e-6):
                raise ValueError(
                    f"[{arm}] collected all-zero wrench samples during calibration. "
                    f"The {arm} F/T sensor is likely disabled, disconnected, or "
                    "not publishing real data. Fix the sensor/mode before saving "
                    "a calibration."
                )
            model, diag = fit_gravity_model(
                np.array(u_samples[arm]), wrench_array
            )
            models[arm] = model
            logger.info(
                f"[{arm}] fit on {diag['n']} poses: residual force RMS "
                f"{diag['force_rms']:.3f} N, torque RMS {diag['torque_rms']:.3f} Nm, "
                f"design cond {diag['cond']:.1f}"
            )
            if diag["cond"] > 1e3:
                logger.warning(
                    f"[{arm}] poorly-conditioned calibration (cond "
                    f"{diag['cond']:.0f}); increase --wrist-delta or pose variety."
                )

        save_calibration(calib_path, models)
        logger.success(f"Saved F/T gravity calibration to {calib_path}")
    finally:
        bot.shutdown()


def run(
    calib_path: Path = DEFAULT_CALIB_PATH,
    zero_force: bool = True,
    kd_gain: float = 1.0,
    need_button: bool = False,
) -> None:
    """Run admittance control with pose-dependent gravity compensation.

    Two modes:
    1. Zero force control (default): The robot arm moves only in response to
       external forces detected by the force-torque sensor at the end effector,
       without trying to reach any target pose.
    2. Initial pose admittance control: The robot arm tries to go back to its
       initial pose while also responding to external forces.

    Args:
        calib_path: Path to the saved gravity calibration (see :func:`calibrate`).
        zero_force: If True, enables zero force control mode. If False, maintain
            initial pose while responding to external forces.
        kd_gain: Gain for the admittance controller. The larger the gain, the
            stiffer the robot arm will move under external force.
        need_button: If True, the robot arm will move only when the blue button
            is pressed. If False, the robot arm will move continuously.
    """
    if kd_gain < 0.3:
        logger.warning("kd_gain is too small. Setting to 0.3.")
        kd_gain = 0.3

    # Load the pose-dependent gravity + bias model before touching hardware so a
    # missing/invalid calibration fails fast.
    gravity_models = load_calibration(calib_path)
    logger.info(f"Loaded F/T gravity calibration from {calib_path}")

    # Initialize robot and components
    bot = Robot()
    arms = {"left": bot.left_arm, "right": bot.right_arm}
    arm_ee_name = {"left": "L_ee", "right": "R_ee"}

    # Validate wrench sensors
    if bot.left_arm.wrench_sensor is None or bot.right_arm.wrench_sensor is None:
        raise ValueError(
            "Admittance control requires both left and right wrench "
            "sensors to be present."
        )

    # Initialize motion manager and IK solver
    motion_manager, ik_solver = _initialize_robot_and_motion_manager(bot)

    # Get initial poses (admittance equilibrium for the non-zero-force mode)
    init_ee_pose = _get_initial_ee_poses(motion_manager)

    # Initialize admittance controller
    admittance_controller = AdmittanceController(
        zero_force_control=zero_force,
        kd_gain=kd_gain,
        dt=ik_solver.dt,
    )

    # Setup rate limiter
    ik_hz = 1 / ik_solver.dt
    rate_limiter = RateLimiter(ik_hz)

    logger.info("Admittance control started (pose-dependent gravity compensation).")
    pin_robot = motion_manager.pin_robot
    assert pin_robot is not None

    try:
        while True:
            # Get current wrench states
            wrench_states = {
                "left": bot.left_arm.wrench_sensor.get_state(),
                "right": bot.right_arm.wrench_sensor.get_state(),
            }

            # Check activation condition
            activated = any(
                wrench_states[arm]["blue_button"] for arm in ["left", "right"]
            )
            activated = True if not need_button else activated

            if activated:
                # Update motion manager with current joint positions
                motion_manager.set_joint_pos(
                    bot.get_joint_pos_dict(component=["left_arm", "right_arm"])
                )

                # Compute current end-effector poses
                ee_pose_result = motion_manager.fk(
                    frame_names=motion_manager.target_frames,
                    qpos=motion_manager.get_joint_pos(),
                )
                ee_pose: dict[str, np.ndarray] = {
                    arm: ee_pose_result[arm_ee_name[arm]].np  # type: ignore
                    for arm in ("left", "right")
                }

                # Update poses with admittance control
                _update_ee_poses_with_admittance(
                    ee_pose,
                    wrench_states,
                    gravity_models,
                    admittance_controller,
                    init_ee_pose,
                    zero_force,
                )

                # Solve inverse kinematics
                target_qpos_dict = ik_solver.solve_ik(
                    target_pose_dict={
                        "L_ee": ee_pose["left"],
                        "R_ee": ee_pose["right"],
                    },
                )[0]

                # Send joint commands
                _send_joint_commands(target_qpos_dict, wrench_states, arms, need_button)

            rate_limiter.sleep()

    except KeyboardInterrupt:
        logger.info("Admittance control stopped by user.")
    finally:
        bot.shutdown()


def diagnose(
    calib_path: Path = DEFAULT_CALIB_PATH,
    samples: int = 20,
    sample_dt: float = 0.05,
) -> None:
    """Print left/right admittance prerequisites: calibration, F/T mode, wrench, buttons."""
    _log_calibration_file_health(calib_path)

    bot = Robot()
    try:
        arms = {"left": bot.left_arm, "right": bot.right_arm}
        for arm_name, arm in arms.items():
            logger.info(f"[{arm_name}] arm active={arm.is_active()}")

            try:
                mode = arm.get_force_torque_sensor_mode()
                logger.info(f"[{arm_name}] force_torque_sensor_mode={mode}")
            except Exception as exc:  # noqa: BLE001 - diagnostic command must continue per arm
                logger.warning(
                    f"[{arm_name}] could not query force-torque sensor mode: {exc}"
                )

            sensor = arm.wrench_sensor
            if sensor is None:
                logger.error(f"[{arm_name}] no wrench sensor configured")
                continue

            logger.info(f"[{arm_name}] wrench sensor active={sensor.is_active()}")
            readings: list[np.ndarray] = []
            last_buttons: dict[str, bool] | None = None
            try:
                for _ in range(max(1, samples)):
                    state = sensor.get_state()
                    readings.append(np.asarray(state["wrench"], dtype=float))
                    last_buttons = {
                        "blue_button": bool(state["blue_button"]),
                        "green_button": bool(state["green_button"]),
                    }
                    time.sleep(sample_dt)
            except Exception as exc:  # noqa: BLE001 - diagnostic command must continue per arm
                logger.error(f"[{arm_name}] failed to read wrench/button state: {exc}")
                continue

            wrench = np.vstack(readings)
            mean = wrench.mean(axis=0)
            std = wrench.std(axis=0)
            all_zero = bool(np.allclose(wrench, 0.0, atol=1e-6))
            log = logger.error if all_zero else logger.info
            log(
                f"[{arm_name}] wrench all_zero={all_zero} "
                f"mean={np.array2string(mean, precision=3)} "
                f"std={np.array2string(std, precision=3)}"
            )
            logger.info(f"[{arm_name}] buttons={last_buttons}")
    finally:
        bot.shutdown()


if __name__ == "__main__":
    subcommand_cli_from_dict(
        {"run": run, "calibrate": calibrate, "diagnose": diagnose}
    )
