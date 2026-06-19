#!/usr/bin/env python3
"""Visualise a drive_box_record.py run: commanded box vs odom vs ZED.

Reads the 10 Hz HDF5 written by scripts/drive_box_record.py and produces:

  * a static PNG — XY trajectory (the commanded box polyline from the leg
    schedule, the swerve-odometry path, and the ZED path, all anchored at the
    same origin) plus x/y, yaw and speed-vs-time panels; and
  * an MP4 animation — the odom and ZED base markers driving their measured
    paths over the commanded box, with growing trails, heading arrows, a
    yaw-vs-time cursor, and (if recorded) the head-camera view in a left panel
    so you can confirm the head stayed still.

The "commanded trajectory" is reconstructed from ``legs`` (each straight/sideways
leg = speed x drive_time in the base frame; each turn rotates in place), i.e. the
intended ~0.8 m box that closes to origin + the 360 deg turn — not the raw cmd_vel
integral, which would also accumulate during the in-place sequential re-steer.

Run:
    conda run -n yixuan_yifan python scripts/plot_drive_box.py \
        --input ~/yixuan/Dexmate/SLAM/drive_box.hdf5
"""

from __future__ import annotations

import dataclasses
import pathlib
import subprocess
from typing import Optional

import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    input: str = "/home/dexmate/yixuan/Dexmate/SLAM/drive_box.hdf5"
    """HDF5 written by scripts/drive_box_record.py."""

    out_png: Optional[str] = None
    """Static figure (default: <input>.png)."""

    out_mp4: Optional[str] = None
    """Animation (default: <input>.mp4; use .gif if ffmpeg is unavailable)."""

    fps: int = 10
    """Animation playback fps (record is 10 Hz → fps=10 is real-time)."""

    head_inset: bool = True
    """Show the recorded head_left_rgb frame in a left panel of the animation, if present."""

    no_video: bool = False
    """Only write the static PNG (skip the MP4)."""


def _load(path: str) -> dict:
    import h5py

    p = pathlib.Path(path).expanduser()
    if not p.is_file():
        raise SystemExit(f"input not found: {p}")
    need = {
        "time/t_wall_s", "action/chassis/cmd_vel", "obs/odom/pose", "obs/odom/twist",
        "legs/label", "legs/cmd_vel",
        "legs/drive_time", "legs/t_issue", "legs/t_done",
    }
    # ZED base positioning was retired (→ wheel odometry); newer HDF5s omit these.
    optional = {"obs/zed/pose", "obs/zed/fresh"}
    out: dict = {}
    with h5py.File(p, "r") as f:
        missing = [k for k in need if k not in f]
        if missing:
            raise SystemExit(f"{p} missing datasets {missing}; is it a drive_box_record HDF5?")
        for k in need:
            out[k] = f[k][()]
        for k in optional:
            out[k] = f[k][()] if k in f else None
        out["head_rgb"] = f["obs/images/head_left_rgb"][()] if "obs/images/head_left_rgb" in f else None
        out["zed_cam"] = f["obs/zed/cam_pose"][()] if "obs/zed/cam_pose" in f else None
        out["attrs"] = dict(f.attrs)
    out["legs/label"] = [b.decode() if isinstance(b, bytes) else str(b) for b in out["legs/label"]]
    return out


def _commanded_box(legs_cmd: np.ndarray, legs_dt: np.ndarray) -> tuple[np.ndarray, list]:
    """Intended polyline of base positions + (center, dyaw) for in-place turns."""
    pos = np.zeros(2)
    yaw = 0.0
    corners = [pos.copy()]
    turns: list[tuple[np.ndarray, float]] = []
    for (vx, vy, wz), dt in zip(legs_cmd, legs_dt):
        if vx == 0.0 and vy == 0.0 and wz != 0.0:
            turns.append((pos.copy(), wz * dt))
            yaw += wz * dt
        else:
            c, s = np.cos(yaw), np.sin(yaw)
            pos = pos + np.array([[c, -s], [s, c]]) @ np.array([vx * dt, vy * dt])
            corners.append(pos.copy())
    return np.asarray(corners), turns


def _closure(pose: np.ndarray) -> tuple[float, float]:
    """|xy| (m) and yaw (deg) of the final sample relative to the origin."""
    return float(np.hypot(pose[-1, 0], pose[-1, 1])), float(np.degrees(pose[-1, 2]))


def static_plot(d: dict, out_png: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = d["time/t_wall_s"]
    cmd = d["action/chassis/cmd_vel"]
    odom = d["obs/odom/pose"]
    twist = d["obs/odom/twist"]
    has_zed = d.get("obs/zed/pose") is not None
    zed = d["obs/zed/pose"].astype(float) if has_zed else None
    zed_m = None
    if has_zed:
        zed_m = zed.copy()
        zed_m[~np.isfinite(zed_m).all(axis=1)] = np.nan
    zcam = d.get("zed_cam")
    zcam_m = None
    if zcam is not None:
        zcam_m = np.asarray(zcam, float).copy()
        zcam_m[~np.isfinite(zcam_m).all(axis=1)] = np.nan
    corners, turns = _commanded_box(d["legs/cmd_vel"], d["legs/drive_time"])

    fig = plt.figure(figsize=(13.5, 8.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(corners[:, 0], corners[:, 1], "--", color="0.5", lw=1.6, label="commanded box", zorder=2)
    ax.scatter(corners[:, 0], corners[:, 1], c="0.5", s=18, zorder=3)
    ax.plot(odom[:, 0], odom[:, 1], color="tab:blue", lw=1.8, label="odom (wheels)", zorder=4)
    if has_zed:
        ax.plot(zed_m[:, 0], zed_m[:, 1], color="tab:orange", lw=1.8, label="ZED (base)", zorder=4)
    if zcam_m is not None:
        ax.plot(zcam_m[:, 0], zcam_m[:, 1], color="tab:red", lw=1.2, ls=":",
                label="ZED cam (raw)", zorder=4)
    ax.scatter([0], [0], c="g", s=70, marker="o", label="origin", zorder=5)
    ax.scatter([odom[-1, 0]], [odom[-1, 1]], c="tab:blue", s=60, marker="X", zorder=6)
    if has_zed and np.isfinite(zed_m[-1, 0]):
        ax.scatter([zed_m[-1, 0]], [zed_m[-1, 1]], c="tab:orange", s=60, marker="X", zorder=6)
    for c, dyaw in turns:
        ax.annotate(f"turn {np.degrees(dyaw):+.0f}°", c, textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color="0.4")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    o_xy, o_yaw = _closure(odom)
    title = f"base XY — closure: odom |xy|={1000 * o_xy:.0f}mm yaw={o_yaw:+.1f}°"
    if has_zed:
        zc = "n/a" if not np.isfinite(zed_m[-1, 0]) else f"{1000 * _closure(zed_m)[0]:.0f}mm"
        title += f"  | zed |xy|={zc}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)

    axx = fig.add_subplot(gs[0, 1])
    for k, lbl, c in ((0, "x", "tab:green"), (1, "y", "tab:purple")):
        axx.plot(t, odom[:, k], color=c, lw=1.4, label=f"odom {lbl}")
        if has_zed:
            axx.plot(t, zed_m[:, k], color=c, lw=1.2, ls="--", label=f"zed {lbl}")
    axx.set_ylabel("pos [m]"); axx.grid(True, alpha=0.3); axx.legend(fontsize=7, ncol=2)
    axx.set_title("position vs time")

    axy = fig.add_subplot(gs[1, 1], sharex=axx)
    axy.plot(t, np.degrees(np.unwrap(odom[:, 2])), color="tab:blue", lw=1.4, label="odom yaw")
    if has_zed and np.isfinite(zed[:, 2]).all():
        axy.plot(t, np.degrees(np.unwrap(zed[:, 2])), color="tab:orange", lw=1.4, ls="--", label="zed yaw")
    axy.set_ylabel("yaw [deg]"); axy.grid(True, alpha=0.3); axy.legend(fontsize=8)

    axs = fig.add_subplot(gs[2, 1], sharex=axx)
    axs.plot(t, np.hypot(cmd[:, 0], cmd[:, 1]), color="k", lw=1.2, label="cmd |v|")
    axs.plot(t, np.hypot(twist[:, 0], twist[:, 1]), color="tab:blue", lw=1.4, label="odom |v|")
    axs.plot(t, np.abs(cmd[:, 2]), color="0.5", lw=1.0, ls=":", label="cmd |ω|")
    axs.plot(t, np.abs(twist[:, 2]), color="tab:red", lw=1.2, label="odom |ω|")
    for ti, td in zip(d["legs/t_issue"], d["legs/t_done"]):
        if np.isfinite(ti) and np.isfinite(td):
            axs.axvspan(ti, td, color="0.85", alpha=0.5, zorder=0)
    axs.set_xlabel("t [s]"); axs.set_ylabel("speed"); axs.grid(True, alpha=0.3)
    axs.legend(fontsize=7, ncol=2); axs.set_title("commanded vs measured speed (shaded = leg windows)")

    fig.suptitle(f"drive_box: {pathlib.Path(d['_path']).name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"wrote {out_png}")


def _pick_mp4_codec() -> str:
    """libx264 when the ffmpeg build has it; else software mpeg4 (the Jetson's
    ffmpeg ships no libx264), matching legacy/VSLAM/scripts/zed_base_pose_node.py."""
    from matplotlib import rcParams

    out = subprocess.run([rcParams["animation.ffmpeg_path"], "-hide_banner", "-encoders"],
                         capture_output=True, text=True, check=False).stdout
    return "libx264" if " libx264 " in out else "mpeg4"


def animate(d: dict, out_mp4: pathlib.Path, fps: int, head_inset: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    t = d["time/t_wall_s"]
    odom = d["obs/odom/pose"]
    has_zed = d.get("obs/zed/pose") is not None
    zed = d["obs/zed/pose"].astype(float) if has_zed else None
    twist = d["obs/odom/twist"]
    cmd = d["action/chassis/cmd_vel"]
    labels = d.get("legs/label", [])
    corners, turns = _commanded_box(d["legs/cmd_vel"], d["legs/drive_time"])
    zfinite = np.isfinite(zed).all(axis=1) if has_zed else None
    zcam = d.get("zed_cam")
    zcam = np.asarray(zcam, float) if zcam is not None else None
    zcam_finite = np.isfinite(zcam).all(axis=1) if zcam is not None else None
    rgb = d["head_rgb"] if (head_inset and d.get("head_rgb") is not None) else None

    if out_mp4.suffix.lower() == ".mp4" and not animation.FFMpegWriter.isAvailable():
        out_mp4 = out_mp4.with_suffix(".gif")
        print(f"ffmpeg unavailable → writing {out_mp4} instead")
    writer = (animation.FFMpegWriter(fps=fps, codec=_pick_mp4_codec(), bitrate=2400)
              if out_mp4.suffix.lower() == ".mp4" else animation.PillowWriter(fps=fps))

    # which leg is active at each record sample (for the on-screen label)
    leg_at = np.full(len(t), "", dtype=object)
    for lab, ti, td in zip(labels, d["legs/t_issue"], d["legs/t_done"]):
        if np.isfinite(ti) and np.isfinite(td):
            leg_at[(t >= ti) & (t <= td)] = lab

    has_head = rgb is not None
    fig = plt.figure(figsize=(16.0, 6.4) if has_head else (12.5, 6.4))
    if has_head:
        gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.25, 1.0])
        ax_cam = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1])
        axy = fig.add_subplot(gs[0, 2])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
        ax_cam = None
        ax = fig.add_subplot(gs[0, 0])
        axy = fig.add_subplot(gs[0, 1])

    cam_x_pts = zcam[zcam_finite, 0] if zcam is not None else np.empty(0)
    cam_y_pts = zcam[zcam_finite, 1] if zcam is not None else np.empty(0)
    zed_x_pts = zed[zfinite, 0] if has_zed else np.empty(0)
    zed_y_pts = zed[zfinite, 1] if has_zed else np.empty(0)
    allx = np.concatenate([odom[:, 0], zed_x_pts, cam_x_pts, corners[:, 0], [0]])
    ally = np.concatenate([odom[:, 1], zed_y_pts, cam_y_pts, corners[:, 1], [0]])
    cx, cy = (allx.min() + allx.max()) / 2, (ally.min() + ally.max()) / 2
    half = max(np.ptp(allx), np.ptp(ally)) / 2 * 1.2 + 0.1
    arrow = 0.16 * half
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.plot(corners[:, 0], corners[:, 1], "--", color="0.6", lw=1.4, zorder=1, label="commanded box")
    ax.scatter([0], [0], c="g", s=60, zorder=2, label="origin")
    od_trail, = ax.plot([], [], color="tab:blue", lw=1.7, zorder=3, label="odom")
    zd_trail = zd_dot = zd_arr = None
    if has_zed:
        zd_trail, = ax.plot([], [], color="tab:orange", lw=1.7, zorder=3, label="ZED (base)")
    zc_trail = None
    if zcam is not None:
        zc_trail, = ax.plot([], [], color="tab:red", lw=1.3, ls=":", zorder=3, label="ZED cam (raw)")
    od_dot = ax.scatter([], [], c="tab:blue", s=55, zorder=5)
    od_arr = ax.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0,
                       color="tab:blue", width=0.008, zorder=6)
    if has_zed:
        zd_dot = ax.scatter([], [], c="tab:orange", s=55, zorder=5)
        zd_arr = ax.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0,
                           color="tab:orange", width=0.008, zorder=6)
    ax.legend(loc="upper right", fontsize=8)

    yaw_o = np.degrees(np.unwrap(odom[:, 2]))
    axy.plot(t, yaw_o, color="tab:blue", lw=1.3, label="odom yaw")
    if has_zed:
        yaw_z = np.degrees(np.unwrap(np.where(zfinite, zed[:, 2], np.nan)))
        axy.plot(t, yaw_z, color="tab:orange", lw=1.3, ls="--", label="zed yaw")
    axy.set_xlabel("t [s]"); axy.set_ylabel("yaw [deg]"); axy.grid(True, alpha=0.3); axy.legend(fontsize=8)
    cursor = axy.axvline(t[0], color="0.35", lw=1.0)

    im = None
    if ax_cam is not None:
        ax_cam.axis("off")
        im = ax_cam.imshow(rgb[0])
        ax_cam.set_title("head cam", fontsize=9)

    frame_times = np.arange(t[0], t[-1] + 1e-9, 1.0 / fps)
    if len(frame_times) > 3000:
        frame_times = np.linspace(t[0], t[-1], 3000)
    print(f"rendering {len(frame_times)} frames → {out_mp4}")
    with writer.saving(fig, out_mp4.as_posix(), dpi=110):
        for tf in frame_times:
            i = int(np.clip(np.searchsorted(t, tf, "right") - 1, 0, len(t) - 1))
            od_trail.set_data(odom[: i + 1, 0], odom[: i + 1, 1])
            if has_zed:
                zf = zfinite[: i + 1]
                zd_trail.set_data(np.where(zf, zed[: i + 1, 0], np.nan),
                                  np.where(zf, zed[: i + 1, 1], np.nan))
            if zc_trail is not None:
                zcf = zcam_finite[: i + 1]
                zc_trail.set_data(np.where(zcf, zcam[: i + 1, 0], np.nan),
                                  np.where(zcf, zcam[: i + 1, 1], np.nan))
            od_dot.set_offsets([[odom[i, 0], odom[i, 1]]])
            od_arr.set_offsets([[odom[i, 0], odom[i, 1]]])
            od_arr.set_UVC(arrow * np.cos(odom[i, 2]), arrow * np.sin(odom[i, 2]))
            if has_zed and zfinite[i]:
                zd_dot.set_offsets([[zed[i, 0], zed[i, 1]]])
                zd_arr.set_offsets([[zed[i, 0], zed[i, 1]]])
                zd_arr.set_UVC(arrow * np.cos(zed[i, 2]), arrow * np.sin(zed[i, 2]))
            cursor.set_xdata([t[i], t[i]])
            if im is not None:
                im.set_data(rgb[i])
            spd = float(np.hypot(twist[i, 0], twist[i, 1]))
            fig.suptitle(f"t={t[i]:5.1f}s   {leg_at[i]:14s}   "
                         f"odom |v|={spd:.2f} m/s  yaw={yaw_o[i]:+.0f}°", fontsize=11)
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {out_mp4}")


def main() -> None:
    args = tyro.cli(Args)
    d = _load(args.input)
    d["_path"] = args.input
    inp = pathlib.Path(args.input).expanduser()
    out_png = pathlib.Path(args.out_png).expanduser() if args.out_png else inp.with_suffix(".png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    static_plot(d, out_png)
    if not args.no_video:
        out_mp4 = pathlib.Path(args.out_mp4).expanduser() if args.out_mp4 else inp.with_suffix(".mp4")
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        animate(d, out_mp4, args.fps, args.head_inset)


if __name__ == "__main__":
    main()
