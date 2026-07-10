#!/usr/bin/env python3
"""
A* global path planning on the SLAM occupancy grid (map frame).
ME 5659 | Pradeep Sivaa Aiyanar

References:
  [1] Hart, Nilsson & Raphael (1968), "A Formal Basis for the Heuristic
      Determination of Minimum Cost Paths" — A* search.
  [2] Nav2 costmap_2d InflationLayer — graduated cost around lethal cells that
      biases the plan toward the centre of free space.

Per replan: inflate the latest /map with a scipy Euclidean distance transform
(cells within robot_radius + inflation_margin of an obstacle are lethal; free
cells beyond get a cost that decays with clearance), run 8-connected A*
(corner-cutting blocked) from the map->base_link pose to the goal, then
string-pull the grid path into straight line-of-sight segments.

The path is published in the map frame: the NMPC, EKF pose and RViz all work in
map after the Phase-2 SLAM fusion, so no per-cycle TF is needed and map drift
cannot rotate the plan.

Subscribes:  /map (OccupancyGrid, transient_local), /goal_pose (PoseStamped)
Publishes:   /global_path (Path), /global_path_markers (Marker, LINE_STRIP)
TF:          map->base_link (start pose)
"""

import heapq
import math
import time as _time

import numpy as np
from scipy.ndimage import distance_transform_edt

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time as RosTime
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy,
                       QoSReliabilityPolicy, QoSHistoryPolicy)

import tf2_ros

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker

SQRT2 = math.sqrt(2.0)


class GlobalPlanner(Node):
    """A* global planner over the inflated slam_toolbox occupancy grid."""

    def __init__(self):
        super().__init__('global_planner')

        # ── Parameters ────────────────────────────────────────────
        self.declare_parameter('robot_radius', 0.21)           # m (chassis half-diagonal)
        # Tuning: keep inflation small + downsample low, or obstacle halos merge
        # and close the ~3 m inter-cylinder gaps, forcing long detours.
        self.declare_parameter('inflation_margin', 0.15)       # m extra clearance
        self.declare_parameter('inflation_cost_weight', 4.0)   # soft-cost magnitude at edge
        self.declare_parameter('inflation_decay', 3.0)         # 1/m soft-cost decay
        self.declare_parameter('occupied_thresh', 65)          # cell >= -> lethal
        self.declare_parameter('allow_unknown', True)          # unknown (-1) traversable
        self.declare_parameter('planning_downsample', 2)       # coarsen grid NxN for speed
        self.declare_parameter('heuristic_weight', 1.0)        # >1 = weighted A* (faster)
        self.declare_parameter('allow_diagonal', True)         # 8-connected search
        self.declare_parameter('replan_period', 1.0)           # s between replans
        self.declare_parameter('goal_x', 13.0)                 # default goal (map frame)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('global_frame', 'map')          # grid + published path frame
        self.declare_parameter('control_frame', 'odom')        # legacy; unused since map-frame publish
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('goal_reached_tol', 0.35)       # m — stop replanning at goal
        # Path hysteresis: don't switch routes unless the current one is bad or
        # a new one is clearly better — stops A* flapping between equal routes.
        self.declare_parameter('path_hysteresis', True)
        self.declare_parameter('hysteresis_improve', 0.15)     # switch if new >15% shorter
        self.declare_parameter('hysteresis_max_stray', 0.8)    # m off-path -> abandon it

        self.robot_radius   = self.get_parameter('robot_radius').value
        self.infl_margin    = self.get_parameter('inflation_margin').value
        self.infl_weight    = self.get_parameter('inflation_cost_weight').value
        self.infl_decay     = self.get_parameter('inflation_decay').value
        self.occ_thresh     = int(self.get_parameter('occupied_thresh').value)
        self.allow_unknown  = bool(self.get_parameter('allow_unknown').value)
        self.downsample     = max(1, int(self.get_parameter('planning_downsample').value))
        self.heur_w         = float(self.get_parameter('heuristic_weight').value)
        self.diagonal       = bool(self.get_parameter('allow_diagonal').value)
        replan_period       = float(self.get_parameter('replan_period').value)
        self.global_frame   = self.get_parameter('global_frame').value
        self.control_frame  = self.get_parameter('control_frame').value
        self.base_frame     = self.get_parameter('robot_base_frame').value
        self.goal_tol       = float(self.get_parameter('goal_reached_tol').value)
        self._hysteresis    = bool(self.get_parameter('path_hysteresis').value)
        self._hyst_improve  = float(self.get_parameter('hysteresis_improve').value)
        self._hyst_stray    = float(self.get_parameter('hysteresis_max_stray').value)

        # Total clearance the robot centre must keep from any obstacle surface
        self.inflation_radius = self.robot_radius + self.infl_margin

        # ── Goal state (defined in the control/odom frame) ─────────
        self._goal_xy = np.array([
            float(self.get_parameter('goal_x').value),
            float(self.get_parameter('goal_y').value),
        ])

        # ── Cached map + inflation layer (rebuilt only on new map) ─
        self._grid_msg   = None      # latest OccupancyGrid
        self._lethal     = None      # bool (H,W) — inflated obstacles
        self._cost_field = None      # float32 (H,W) — soft inflation cost
        self._cell_size  = None      # m per (downsampled) planning cell
        self._origin_x   = 0.0       # map-frame origin of grid (m)
        self._origin_y   = 0.0
        self._map_dirty  = False     # new map arrived -> rebuild inflation

        # ── Path hysteresis state (last published path + its goal) ─
        self._last_path  = None      # list[(x,y)] currently committed (map frame)
        self._last_goal  = None      # goal that path was planned to (map frame)

        # ── TF (map->base_link start, map->odom path transform) ────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── QoS: slam_toolbox latches /map (transient_local) ───────
        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        # Latch the path/markers so late RViz subscribers still see them
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        # ── Subscriptions ──────────────────────────────────────────
        map_topic = self.get_parameter('map_topic').value
        self.create_subscription(OccupancyGrid, map_topic, self._map_cb, map_qos)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb, 10)

        # ── Publishers ─────────────────────────────────────────────
        self.path_pub   = self.create_publisher(Path,   '/global_path',         latched_qos)
        self.marker_pub = self.create_publisher(Marker, '/global_path_markers', latched_qos)

        # ── Replan timer ───────────────────────────────────────────
        self.create_timer(replan_period, self._replan)

        # ── Planning-time profiling (honest metrics for the report) ─
        self._plan_times = []

        self.get_logger().info(
            f'Global planner up | goal=({self._goal_xy[0]:.1f},{self._goal_xy[1]:.1f}) '
            f'{self.global_frame} | inflation={self.inflation_radius:.2f} m | downsample x{self.downsample}')

    # ── Callbacks ──────────────────────────────────────────────────
    def _map_cb(self, msg: OccupancyGrid):
        """Store the newest map and flag the inflation layer for rebuild."""
        self._grid_msg  = msg
        self._map_dirty = True

    def _goal_cb(self, msg: PoseStamped):
        """New goal from RViz '2D Nav Goal'.  Stored in the map (global) frame;
        transform if RViz publishes it in another frame."""
        gx, gy = msg.pose.position.x, msg.pose.position.y
        src = msg.header.frame_id
        if src and src != self.global_frame:
            tf = self._tf_xytheta(self.global_frame, src)
            if tf is not None:
                gx, gy = self._apply_xytheta(tf, gx, gy)
        self._goal_xy = np.array([gx, gy])
        self.get_logger().info(
            f'New goal: ({self._goal_xy[0]:.2f}, {self._goal_xy[1]:.2f}) '
            f'[{self.global_frame}] — replanning')
        self._replan()   # immediate replan, don't wait for the timer

    # ── TF helper ───────────────────────────────────────────────────
    def _tf_xytheta(self, target_frame, source_frame):
        """Return (tx, ty, yaw) of source_frame expressed in target_frame,
        i.e. the rigid transform that maps a point p_source -> p_target.
        Returns None if the transform is unavailable."""
        try:
            t = self._tf_buffer.lookup_transform(
                target_frame, source_frame, RosTime(),
                timeout=Duration(seconds=0.1))
        except Exception:
            return None
        tx = t.transform.translation.x
        ty = t.transform.translation.y
        q  = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return tx, ty, yaw

    @staticmethod
    def _apply_xytheta(tf_xyt, px, py):
        """Apply a 2D rigid transform (tx,ty,yaw) to point (px,py)."""
        tx, ty, yaw = tf_xyt
        c, s = math.cos(yaw), math.sin(yaw)
        return (tx + c * px - s * py,
                ty + s * px + c * py)

    # ── Inflation layer ─────────────────────────────────────────────
    def _rebuild_inflation(self):
        """Recompute the lethal mask + soft cost field from the latest map.

        Assumes the grid origin is axis-aligned (yaw = 0), which slam_toolbox
        always publishes for 2D maps.  Origin translation is honoured."""
        msg = self._grid_msg
        info = msg.info
        H, W = info.height, info.width
        res  = info.resolution
        self._origin_x = info.origin.position.x
        self._origin_y = info.origin.position.y

        raw = np.array(msg.data, dtype=np.int16).reshape(H, W)   # row-major, y-up

        # Lethal at native resolution: occupied cells; unknown optionally lethal
        lethal = raw >= self.occ_thresh
        if not self.allow_unknown:
            lethal |= (raw < 0)

        # Coarsen for planning speed: a coarse cell is lethal if ANY sub-cell is
        k = self.downsample
        if k > 1:
            Hc, Wc = H // k, W // k
            if Hc == 0 or Wc == 0:
                k = 1
            else:
                lethal = lethal[:Hc * k, :Wc * k].reshape(Hc, k, Wc, k).any(axis=(1, 3))
        self._cell_size = res * k

        # Euclidean distance (in metres) from every free cell to nearest obstacle
        if lethal.any():
            dist_m = distance_transform_edt(~lethal) * self._cell_size
        else:
            dist_m = np.full(lethal.shape, np.inf, dtype=np.float64)

        # Inflate: block everything within the robot-clearance radius
        lethal_inflated = dist_m <= self.inflation_radius

        # Graduated soft cost beyond the lethal ring (decays with clearance)
        cost = np.zeros(lethal.shape, dtype=np.float32)
        beyond = dist_m > self.inflation_radius
        cost[beyond] = self.infl_weight * np.exp(
            -self.infl_decay * (dist_m[beyond] - self.inflation_radius))

        self._lethal     = lethal_inflated
        self._cost_field = cost
        self._map_dirty  = False

    # ── Grid <-> world (map frame) ──────────────────────────────────
    def _world_to_cell(self, wx, wy):
        c = int((wx - self._origin_x) / self._cell_size)
        r = int((wy - self._origin_y) / self._cell_size)
        return r, c

    def _cell_to_world(self, r, c):
        wx = self._origin_x + (c + 0.5) * self._cell_size
        wy = self._origin_y + (r + 0.5) * self._cell_size
        return wx, wy

    def _in_bounds(self, r, c):
        H, W = self._lethal.shape
        return 0 <= r < H and 0 <= c < W

    def _nearest_free(self, r, c, max_ring=25):
        """Spiral outward from (r,c) to the closest non-lethal in-bounds cell.
        Handles the case where the robot or goal sits inside inflation."""
        if self._in_bounds(r, c) and not self._lethal[r, c]:
            return r, c
        for ring in range(1, max_ring + 1):
            for dr in range(-ring, ring + 1):
                for dc in range(-ring, ring + 1):
                    if max(abs(dr), abs(dc)) != ring:
                        continue   # only the ring perimeter
                    nr, nc = r + dr, c + dc
                    if self._in_bounds(nr, nc) and not self._lethal[nr, nc]:
                        return nr, nc
        return None

    # ── A* search ────────────────────────────────────────────────────
    def _astar(self, start, goal):
        """8- (or 4-) connected A* on the inflated grid. Returns list of
        (r,c) cells start->goal, or None if unreachable."""
        lethal = self._lethal
        cost_f = self._cost_field
        H, W = lethal.shape
        sr, sc = start
        gr, gc = goal
        cell = self._cell_size
        hw   = self.heur_w

        si = sr * W + sc
        gi = gr * W + gc

        g_score = np.full(H * W, np.inf, dtype=np.float64)
        came    = np.full(H * W, -1,     dtype=np.int64)
        closed  = np.zeros(H * W,        dtype=bool)

        g_score[si] = 0.0
        h0 = math.hypot(sr - gr, sc - gc) * cell * hw
        open_heap = [(h0, si)]

        if self.diagonal:
            moves = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                     (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2))
        else:
            moves = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0))

        found = False
        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if closed[cur]:
                continue
            closed[cur] = True
            if cur == gi:
                found = True
                break
            cr, cc = divmod(cur, W)
            gcur = g_score[cur]
            for dr, dc, mv in moves:
                nr, nc = cr + dr, cc + dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue
                if lethal[nr, nc]:
                    continue
                # Block diagonal corner-cutting through obstacle corners
                if mv > 1.0 and (lethal[cr, nc] or lethal[nr, cc]):
                    continue
                ni = nr * W + nc
                if closed[ni]:
                    continue
                # Scale by cell size so the soft cost is "extra cost per metre",
                # comparable to path length (per-cell it would dominate and detour).
                step = (mv + float(cost_f[nr, nc])) * cell
                ng = gcur + step
                if ng < g_score[ni]:
                    g_score[ni] = ng
                    came[ni] = cur
                    h = math.hypot(nr - gr, nc - gc) * cell * hw
                    heapq.heappush(open_heap, (ng + h, ni))

        if not found:
            return None

        # Reconstruct
        path = []
        cur = gi
        while cur != -1:
            path.append(divmod(cur, W))
            cur = came[cur]
        path.reverse()
        return path

    # ── Path post-processing ────────────────────────────────────────
    def _line_clear(self, a, b):
        """Bresenham line-of-sight check over the lethal grid (a,b are cells)."""
        lethal = self._lethal
        r0, c0 = a
        r1, c1 = b
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            if lethal[r, c]:
                return False
            if r == r1 and c == c1:
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    def _shortcut(self, cells):
        """String-pulling: greedily replace runs of cells with straight
        line-of-sight segments, collapsing the 8-connected staircase."""
        if len(cells) <= 2:
            return cells
        out = [cells[0]]
        i = 0
        n = len(cells)
        while i < n - 1:
            j = n - 1
            while j > i + 1:
                if self._line_clear(cells[i], cells[j]):
                    break
                j -= 1
            out.append(cells[j])
            i = j
        return out

    # ── Path hysteresis helpers ───────────────────────────────────────
    @staticmethod
    def _proj_on_path(path, px, py):
        """Project (px,py) onto the polyline. Return (seg_index, t, dist)."""
        best = (0, 0.0, float('inf'))
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
            d = math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
            if d < best[2]:
                best = (i, t, d)
        return best

    def _remaining_length(self, path, px, py):
        """Arc length from the robot's projection on `path` to its end (m)."""
        if len(path) < 2:
            return 0.0
        i, t, _ = self._proj_on_path(path, px, py)
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        proj = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        total = math.hypot(x2 - proj[0], y2 - proj[1])
        for k in range(i + 1, len(path) - 1):
            total += math.hypot(path[k + 1][0] - path[k][0],
                                path[k + 1][1] - path[k][1])
        return total

    def _path_still_valid(self, path, px, py):
        """True if the committed path is still usable: robot hasn't strayed off
        it and the remaining segments are collision-free on the CURRENT grid."""
        if path is None or len(path) < 2:
            return False
        i, _, dproj = self._proj_on_path(path, px, py)
        if dproj > self._hyst_stray:          # robot wandered off the path
            return False
        for k in range(i, len(path) - 1):     # re-check remaining segments
            a = self._world_to_cell(*path[k])
            b = self._world_to_cell(*path[k + 1])
            if not (self._in_bounds(*a) and self._in_bounds(*b)):
                return False
            if not self._line_clear(a, b):    # a new obstacle now blocks it
                return False
        return True

    # ── Replan ───────────────────────────────────────────────────────
    def _replan(self):
        if self._grid_msg is None:
            self.get_logger().warn('No /map yet — waiting for slam_toolbox',
                                    throttle_duration_sec=5.0)
            return

        if self._map_dirty or self._lethal is None:
            self._rebuild_inflation()

        # Start pose: robot origin in the map frame
        start_tf = self._tf_xytheta(self.global_frame, self.base_frame)
        if start_tf is None:
            self.get_logger().warn(
                f'No {self.global_frame}->{self.base_frame} TF — cannot plan',
                throttle_duration_sec=5.0)
            return
        start_map = (start_tf[0], start_tf[1])

        # Goal is defined in the map frame (same as the NMPC), so no transform.
        goal_map = (float(self._goal_xy[0]), float(self._goal_xy[1]))

        # Skip replanning once we are basically on top of the goal
        if math.hypot(start_map[0] - goal_map[0],
                      start_map[1] - goal_map[1]) < self.goal_tol:
            return

        s_rc = self._world_to_cell(*start_map)
        g_rc = self._world_to_cell(*goal_map)

        start_cell = self._nearest_free(*s_rc)
        goal_cell  = self._nearest_free(*g_rc)
        if start_cell is None or goal_cell is None:
            self.get_logger().warn('Start or goal has no free cell nearby — skipping replan',
                                    throttle_duration_sec=3.0)
            return

        t0 = _time.perf_counter()
        cells = self._astar(start_cell, goal_cell)
        t_ms = (_time.perf_counter() - t0) * 1e3

        if cells is None:
            self.get_logger().warn('A*: no path found to goal', throttle_duration_sec=3.0)
            return

        smoothed = self._shortcut(cells)

        # Cells -> world (map frame, published as-is); no map->odom keeps drift
        # from rotating the path.
        pts = [self._cell_to_world(r, c) for (r, c) in smoothed]

        # Snap the final waypoint exactly to the commanded goal so the NMPC
        # terminal target matches the goal-reached check (both in map).
        pts[-1] = (float(self._goal_xy[0]), float(self._goal_xy[1]))

        # ── Path hysteresis ──────────────────────────────────────────
        # A* can flip between near-equal-length homotopy routes each replan,
        # jerking the MPCC target. Commit to the current path; switch only if the
        # goal changed, the old path is blocked / strayed off, or the new plan is
        # meaningfully (>hyst_improve) shorter.
        goal_changed = (self._last_goal is None or
                        math.hypot(goal_map[0] - self._last_goal[0],
                                   goal_map[1] - self._last_goal[1]) > 1e-3)
        kept = False
        if (self._hysteresis and not goal_changed
                and self._path_still_valid(self._last_path, *start_map)):
            old_len = self._remaining_length(self._last_path, *start_map)
            new_len = self._remaining_length(pts, *start_map)
            if new_len >= old_len * (1.0 - self._hyst_improve):
                pts = self._last_path        # keep the committed route
                kept = True

        if not kept:
            self._last_path = pts
        self._last_goal = goal_map

        self._publish_path(pts)
        self._publish_marker(pts)

        # Profiling
        self._plan_times.append(t_ms)
        if len(self._plan_times) > 50:
            self._plan_times.pop(0)
        length = sum(math.hypot(pts[i + 1][0] - pts[i][0],
                                pts[i + 1][1] - pts[i][1])
                     for i in range(len(pts) - 1))
        self.get_logger().info(
            f'A* path: {len(cells)} cells -> {len(pts)} waypoints, '
            f'{length:.1f} m, solve {t_ms:.1f} ms '
            f'[{"kept" if kept else "new"}]', throttle_duration_sec=2.0)

    # ── Publishing ───────────────────────────────────────────────────
    def _publish_path(self, pts):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.global_frame
        for (x, y) in pts:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)

    def _publish_marker(self, pts):
        m = Marker()
        m.header.frame_id = self.global_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'global_path'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.08                 # line width (m)
        m.color.r = 0.1
        m.color.g = 1.0
        m.color.b = 0.3
        m.color.a = 0.9
        m.pose.orientation.w = 1.0
        for (x, y) in pts:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.02
            m.points.append(p)
        self.marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
