"""Interactive terminal control for the whole-body-control SAPIEN demo.

``TargetController`` lets the operator drive the dual-arm end-effector targets from
the terminal:

- **arrow keys** nudge the selected target(s): Up/Down = +x/-x (forward/back),
  Left/Right = +y/-y (left/right);
- **r / f** = +z / -z (up / down);
- **1 / 2 / 3** = control the left / right / both arms;
- **p** = type exact x y z positions for the left and right targets;
- **space** = recenter to the starting targets;
- **q** (or Ctrl-C) = quit.

It is single-threaded: call :meth:`poll` once per control-loop iteration (it reads
whatever keys are pending without blocking). Use it as a context manager so the
terminal is always restored on exit -- including after Ctrl-C -- which keeps the
shell usable afterwards.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty

import numpy as np

# Arrow-key escape final bytes (ESC '[' <code>).
_ARROW_UP = ord("A")
_ARROW_DOWN = ord("B")
_ARROW_RIGHT = ord("C")
_ARROW_LEFT = ord("D")


class TargetController:
    """Keyboard control of the left/right end-effector target positions.

    Args:
        left0: initial left target position (world frame, meters).
        right0: initial right target position.
        step: nudge distance per key press (meters).
        log_prefix: prefix for printed help/status lines.
    """

    def __init__(
        self,
        left0: np.ndarray,
        right0: np.ndarray,
        step: float = 0.02,
        log_prefix: str = "[wbc]",
    ) -> None:
        self._left0 = np.asarray(left0, dtype=float).copy()
        self._right0 = np.asarray(right0, dtype=float).copy()
        self.left = self._left0.copy()
        self.right = self._right0.copy()
        self.step = step
        self.sel = "both"  # "left" | "right" | "both"
        self.quit = False
        self._log_prefix = log_prefix
        self._fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self._old: list | None = None
        self._esc_buf = b""  # carries an incomplete escape sequence between polls

    # -- context manager / terminal state --------------------------------------

    def __enter__(self) -> "TargetController":
        if self._fd is not None:
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)  # cbreak keeps ISIG, so Ctrl-C still interrupts
        self._print_help()
        return self

    def __exit__(self, *exc: object) -> bool:
        self._restore_cooked()
        return False  # never suppress exceptions (incl. KeyboardInterrupt)

    def _restore_cooked(self) -> None:
        if self._fd is not None and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def _set_cbreak(self) -> None:
        if self._fd is not None:
            tty.setcbreak(self._fd)

    def _print_help(self) -> None:
        print(
            f"{self._log_prefix} interactive: arrows=move (Up/Dn=x, L/R=y)  r/f=z  "
            "1/2/3=left/right/both  p=type xyz  space=recenter  q=quit"
        )

    # -- input handling --------------------------------------------------------

    def _read_available(self) -> bytes:
        """Read all currently-pending stdin bytes without blocking."""
        out = b""
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if not ready:
                break
            chunk = os.read(sys.stdin.fileno(), 1024)
            if not chunk:  # EOF (e.g. piped input exhausted, or Ctrl-D)
                self.quit = True
                break
            out += chunk
        return out

    def poll(self) -> None:
        """Process pending key presses (call once per loop iteration)."""
        data = self._esc_buf + self._read_available()
        self._esc_buf = b""
        i, n = 0, len(data)
        while i < n:
            byte = data[i]
            if byte == 0x1B:  # ESC -- expect an arrow sequence: ESC '[' <code>
                if i + 2 < n and data[i + 1] == ord("["):
                    self._handle_arrow(data[i + 2])
                    i += 3
                    continue
                if i + 2 >= n:  # incomplete sequence; finish parsing it next poll
                    self._esc_buf = data[i:]
                    break
                i += 1  # lone ESC / unrecognized sequence
                continue
            self._handle_key(data[i : i + 1])
            i += 1

    def _handle_arrow(self, code: int) -> None:
        if code == _ARROW_UP:
            self._nudge([+self.step, 0.0, 0.0])
        elif code == _ARROW_DOWN:
            self._nudge([-self.step, 0.0, 0.0])
        elif code == _ARROW_LEFT:
            self._nudge([0.0, +self.step, 0.0])
        elif code == _ARROW_RIGHT:
            self._nudge([0.0, -self.step, 0.0])

    def _handle_key(self, ch: bytes) -> None:
        if ch in (b"r", b"R"):
            self._nudge([0.0, 0.0, +self.step])
        elif ch in (b"f", b"F"):
            self._nudge([0.0, 0.0, -self.step])
        elif ch == b"1":
            self.sel = "left"
            print(f"\n{self._log_prefix} controlling: left")
        elif ch == b"2":
            self.sel = "right"
            print(f"\n{self._log_prefix} controlling: right")
        elif ch == b"3":
            self.sel = "both"
            print(f"\n{self._log_prefix} controlling: both")
        elif ch in (b"p", b"P"):
            self._prompt_positions()
        elif ch == b" ":
            self.left = self._left0.copy()
            self.right = self._right0.copy()
            print(f"\n{self._log_prefix} recentered targets")
        elif ch in (b"q", b"Q", b"\x03", b"\x04"):
            self.quit = True

    def _nudge(self, delta: list[float]) -> None:
        d = np.asarray(delta, dtype=float)
        if self.sel in ("left", "both"):
            self.left = self.left + d
        if self.sel in ("right", "both"):
            self.right = self.right + d

    def _prompt_positions(self) -> None:
        """Drop to line input and read exact x y z for the left/right targets."""
        self._restore_cooked()
        try:
            self.left = self._read_xyz("left ", self.left)
            self.right = self._read_xyz("right", self.right)
            print(
                f"{self._log_prefix} set left={np.round(self.left, 3)} "
                f"right={np.round(self.right, 3)}"
            )
        except (EOFError, KeyboardInterrupt):
            self.quit = True
        finally:
            self._set_cbreak()

    def _read_xyz(self, label: str, current: np.ndarray) -> np.ndarray:
        raw = input(f"\n  {label} x y z (blank=keep {np.round(current, 3)}): ").strip()
        if not raw:
            return current
        try:
            vals = [float(v) for v in raw.replace(",", " ").split()]
        except ValueError:
            print("  (could not parse 3 numbers; keeping current)")
            return current
        if len(vals) != 3:
            print("  (need exactly 3 numbers; keeping current)")
            return current
        return np.array(vals, dtype=float)
