"""Base routes planned on the scene's own traversability map.

Long-horizon tasks move the robot several metres between work stations, past furniture the
scripted expert knows nothing about. Hand-authored waypoints are guesswork -- Rs_int, for
instance, looks open-plan in the object list but a full-width wall (``walls_exiopd_0``, y in
[1.45, 1.64]) seals the kitchen off from the living room with a single doorway at x = -2.6.

So routes are PLANNED, on the ``floor_trav_*.png`` that ships with the scene: erode by the
chassis radius, BFS on a coarse grid, then shortcut the result to a handful of waypoints with a
line-of-sight test. That keeps the expert honest (it drives where the scene says floor is) and
makes a new task a pair of station coordinates rather than a hand-traced polyline.

The map is only consulted OFFLINE, when the task builds its waypoint list. Nothing here runs in
the control loop -- see `vega_og_env.base_keepouts` for the runtime guard.
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np

#: dataset resolution of the shipped layout PNGs (omnigibson.maps.TraversableMap)
MAP_RES = 0.01
#: Vega's chassis half-width plus a small margin (the WBC tracks the route with ~5 cm of lag).
ROBOT_RADIUS = 0.33
#: planning-grid cell; fine enough for 0.7 m doorways, coarse enough for an instant BFS
CELL = 0.05


def _dataset_path():
    path = os.environ.get("OMNIGIBSON_DATASET_PATH")
    if path:
        return path
    return os.path.expanduser("~/BEHAVIOR-1K/datasets/behavior-1k-assets")


class NavMap:
    """Traversable floor for one scene, eroded by the chassis radius.

    `clear_boxes` are world-frame (xmin, ymin, xmax, ymax) rectangles to punch back OPEN before
    eroding -- for furniture the task itself moves out of the way at reset (the blocking chairs),
    which the shipped map still shows as occupied.

    `block_boxes` are the mirror image: floor the shipped map shows as OPEN that the episode
    actually fills. Everything the shared layout spawns (`tasks.layout.STATIONS`) is in here --
    the map was baked before those existed, so without this the planner happily routes the
    chassis straight through the dish rack it is on its way to.
    """

    def __init__(self, scene_model, robot_radius=ROBOT_RADIUS, clear_boxes=(), map_name=None,
                 block_boxes=(), point_block_boxes=()):
        import cv2

        layout = os.path.join(_dataset_path(), "scenes", scene_model, "layout")
        name = map_name or "floor_trav_0.png"
        raw = cv2.imread(os.path.join(layout, name), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"no traversability map at {layout}/{name}")
        self.n = raw.shape[0]
        free = (raw == 255).astype(np.uint8)
        for boxes, value in ((clear_boxes, 1), (block_boxes, 0)):
            for xmin, ymin, xmax, ymax in boxes:
                r0, c0 = self._px(xmin, ymin)
                r1, c1 = self._px(xmax, ymax)
                free[min(r0, r1):max(r0, r1) + 1, min(c0, c1):max(c0, c1) + 1] = value
        k = int(np.ceil(2 * robot_radius / MAP_RES))          # cv2.erode shrinks by k/2 per side
        self.free_px = cv2.erode(free, np.ones((k, k), np.uint8))
        # These boxes constrain a point attached at a fixed offset from the chassis (for example,
        # a carried hand), not the chassis disc itself. Painting them before the erosion would
        # incorrectly grow them by ROBOT_RADIUS a second time and can close an otherwise physical
        # corridor. Callers include whatever hand / payload clearance they require in the boxes.
        for xmin, ymin, xmax, ymax in point_block_boxes:
            r0, c0 = self._px(xmin, ymin)
            r1, c1 = self._px(xmax, ymax)
            self.free_px[min(r0, r1):max(r0, r1) + 1,
                         min(c0, c1):max(c0, c1) + 1] = 0
        self.step = max(1, int(round(CELL / MAP_RES)))
        self.grid = self.free_px[::self.step, ::self.step] > 0   # (rows, cols) coarse free mask
        self.scene_model = scene_model

    # ---- coordinates ----
    def _px(self, x, y):
        """World XY -> (row, col) in the full-resolution map (omnigibson MapBase convention)."""
        return int(round(y / MAP_RES + self.n / 2)), int(round(x / MAP_RES + self.n / 2))

    def _cell(self, x, y):
        r, c = self._px(x, y)
        return r // self.step, c // self.step

    def _world(self, r, c):
        return ((c * self.step) - self.n / 2) * MAP_RES, ((r * self.step) - self.n / 2) * MAP_RES

    # ---- queries ----
    def free(self, x, y):
        r, c = self._px(x, y)
        return bool(0 <= r < self.n and 0 <= c < self.n and self.free_px[r, c])

    def nearest_free(self, x, y, max_r=1.5):
        """Closest traversable point to (x, y); the identity when it is already free."""
        if self.free(x, y):
            return np.array([x, y], dtype=float)
        best, best_d = None, np.inf
        for rad in np.arange(CELL, max_r, CELL):
            for ang in np.arange(0.0, 2 * np.pi, 0.15):
                p = (x + rad * np.cos(ang), y + rad * np.sin(ang))
                if self.free(*p) and rad < best_d:
                    best, best_d = p, rad
            if best is not None:
                return np.array(best, dtype=float)
        raise ValueError(f"no traversable point within {max_r} m of ({x:.2f}, {y:.2f}) "
                         f"in {self.scene_model}")

    def visible(self, a, b):
        """True when the straight segment a->b stays on traversable floor."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.linalg.norm(b - a) / (MAP_RES * 2)))
        return all(self.free(*(a + t * (b - a))) for t in np.linspace(0.0, 1.0, n))

    def route(self, start, goal):
        """Shortest traversable polyline start -> goal, shortcut to its corner waypoints.

        Returns an (N, 2) array beginning at `start` and ending at `goal` (both snapped to free
        floor). Raises when the two are not connected -- which is the signal that the two
        stations are in different rooms and the task needs a different destination.
        """
        start = self.nearest_free(*start)
        goal = self.nearest_free(*goal)
        if self.visible(start, goal):
            return np.stack([start, goal])
        path = self._bfs(self._cell(*start), self._cell(*goal))
        if path is None:
            raise ValueError(
                f"{self.scene_model}: no traversable route from ({start[0]:.2f},{start[1]:.2f}) "
                f"to ({goal[0]:.2f},{goal[1]:.2f}) -- different rooms?")
        pts = [start] + [np.array(self._world(r, c)) for r, c in path] + [goal]
        return np.stack(self._drop_short(self._shortcut(pts)))

    # ---- internals ----
    def _bfs(self, src, dst):
        grid = self.grid
        rows, cols = grid.shape
        if not (grid[src] and grid[dst]):
            return None
        prev = -np.ones(rows * cols, dtype=np.int64)
        q = deque([src[0] * cols + src[1]])
        prev[src[0] * cols + src[1]] = src[0] * cols + src[1]
        target = dst[0] * cols + dst[1]
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while q:
            cur = q.popleft()
            if cur == target:
                break
            r, c = divmod(cur, cols)
            for dr, dc in nbrs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc]:
                    idx = nr * cols + nc
                    if prev[idx] < 0:
                        prev[idx] = cur
                        q.append(idx)
        if prev[target] < 0:
            return None
        out, cur = [], target
        while True:
            r, c = divmod(cur, cols)
            out.append((r, c))
            if cur == prev[cur]:
                break
            cur = prev[cur]
        return out[::-1]

    @staticmethod
    def _drop_short(pts, min_leg=0.30):
        """Remove intermediate corners that leave a stub leg.

        The shortcut can end on a few-centimetre leg whose direction is numerical noise. A caller
        that steers by leg direction then commands a spurious turn: measured, an 8 cm final leg
        produced a -19 deg heading wedged between two -77 deg legs and spun the base.
        """
        out = [pts[0]]
        for p in pts[1:-1]:
            if float(np.linalg.norm(p - out[-1])) >= min_leg:
                out.append(p)
        if len(out) > 1 and float(np.linalg.norm(pts[-1] - out[-1])) < min_leg:
            out.pop()                       # the stub is the LAST leg: drop the corner before it
        out.append(pts[-1])
        return out

    def _shortcut(self, pts):
        """Greedy line-of-sight shortcut: keep only the corners the straight path cannot skip."""
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.visible(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out
