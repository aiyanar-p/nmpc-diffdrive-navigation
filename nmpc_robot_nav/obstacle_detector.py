#!/usr/bin/env python3
"""LiDAR obstacle detector: cluster /scan into circles, smooth over time, publish
to /obstacles for the NMPC.

Per scan: filter by range, project polar -> Cartesian in the odom frame, cluster
points (greedy BFS) into (cx, cy, r), then EMA-smooth each obstacle across frames
so jitter does not make the controller over-correct. Output is a fixed-length
Float64MultiArray [x1,y1,r1, ...] plus RViz markers.

Subscribes: /scan (LaserScan), /odom (Odometry).
Publishes:  /obstacles (Float64MultiArray), /obstacle_markers (MarkerArray).
"""

import rclpy
from rclpy.node import Node
import numpy as np
import math

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray


# ── EMA obstacle tracker ──────────────────────────────────────────────────────
class EMATracker:
    """Temporal smoother for detected obstacles.

    Greedily associates each new cluster with the nearest existing track (within
    match_dist), updates matched tracks with an exponential moving average, ages
    out tracks unseen for max_miss_frames, and spawns tracks for new detections.
    """

    def __init__(self, alpha: float = 0.35, match_dist: float = 1.0,
                 max_miss_frames: int = 4):
        self.alpha      = alpha           # EMA weight on the new measurement
        self.match_dist = match_dist      # max distance to associate a cluster
        self.max_miss   = max_miss_frames

        # Each track: {'cx', 'cy', 'r', 'miss'}
        self._tracks = []

    def update(self, raw_obstacles: list,
               robot_pos: tuple = None) -> list:
        """Associate raw (cx,cy,r) detections with tracks, EMA-smooth, and return
        the active (cx, cy, r) list.

        robot_pos enables blind-zone persistence: an obstacle within the LiDAR
        min_range shadow can't be seen but hasn't gone away, so its miss counter
        is frozen and it is never dropped while the robot is close — the NMPC
        keeps the constraint exactly when it matters most.
        """
        matched_track_idx = set()
        matched_obs_idx   = set()

        # ── Associate detections with existing tracks ─────────────
        if self._tracks and raw_obstacles:
            track_xy = np.array([[t['cx'], t['cy']] for t in self._tracks])
            obs_xy   = np.array([[o[0],   o[1]]   for o in raw_obstacles])

            # Pairwise distances (tracks × observations)
            diff = track_xy[:, None, :] - obs_xy[None, :, :]   # (T, O, 2)
            dists = np.hypot(diff[:, :, 0], diff[:, :, 1])      # (T, O)

            # Greedy nearest-neighbour assignment
            while True:
                if dists.size == 0:
                    break
                ti, oi = np.unravel_index(np.argmin(dists), dists.shape)
                if dists[ti, oi] > self.match_dist:
                    break
                cx_r, cy_r, r_r = raw_obstacles[oi]
                t = self._tracks[ti]
                t['cx']   = self.alpha * cx_r + (1 - self.alpha) * t['cx']
                t['cy']   = self.alpha * cy_r + (1 - self.alpha) * t['cy']
                t['r']    = self.alpha * r_r  + (1 - self.alpha) * t['r']
                t['miss'] = 0
                matched_track_idx.add(ti)
                matched_obs_idx.add(oi)
                dists[ti, :] = np.inf
                dists[:, oi] = np.inf

        # ── Age unmatched tracks ──────────────────────────────────
        # Freeze aging inside the LiDAR blind zone (obstacle_radius + min_range,
        # conservatively 1.5 m) so a close, unseen obstacle is not dropped.
        for ti, t in enumerate(self._tracks):
            if ti not in matched_track_idx:
                if robot_pos is not None:
                    d = math.hypot(t['cx'] - robot_pos[0],
                                   t['cy'] - robot_pos[1])
                    if d < 1.5:          # inside blind zone — keep track alive
                        continue
                t['miss'] += 1

        # ── Remove stale tracks ───────────────────────────────────
        self._tracks = [t for t in self._tracks if t['miss'] <= self.max_miss]

        # ── Spawn new tracks for unmatched observations ───────────
        for oi, obs in enumerate(raw_obstacles):
            if oi not in matched_obs_idx:
                self._tracks.append(
                    {'cx': obs[0], 'cy': obs[1], 'r': obs[2], 'miss': 0})

        # ── Return ALL active tracks (including blind-zone ones) ───
        # Tracks in the blind zone have miss==0 (frozen) so they appear here.
        return [(t['cx'], t['cy'], t['r'])
                for t in self._tracks if t['miss'] == 0]

    def reset(self):
        self._tracks.clear()


# ── ObstacleDetector node ─────────────────────────────────────────────────────
class ObstacleDetector(Node):

    def __init__(self):
        super().__init__('obstacle_detector')

        # ── Parameters ───────────────────────────────────────────
        self.declare_parameter('max_range',              10.0)
        self.declare_parameter('min_range',               0.15)
        self.declare_parameter('cluster_tol',             0.4)
        self.declare_parameter('min_cluster_pts',         2)
        self.declare_parameter('max_cluster_pts',         40)
        self.declare_parameter('max_obstacles',           10)
        self.declare_parameter('obstacle_radius_padding', 0.05)
        self.declare_parameter('max_obstacle_radius',     0.8)
        self.declare_parameter('ema_alpha',               0.35)
        self.declare_parameter('ema_match_dist',          1.0)
        self.declare_parameter('ema_max_miss_frames',     4)

        self.max_range   = self.get_parameter('max_range').value
        self.min_range   = self.get_parameter('min_range').value
        self.cluster_tol = self.get_parameter('cluster_tol').value
        self.min_pts     = int(self.get_parameter('min_cluster_pts').value)
        self.max_pts     = int(self.get_parameter('max_cluster_pts').value)
        self.max_obs     = int(self.get_parameter('max_obstacles').value)
        self.r_pad       = self.get_parameter('obstacle_radius_padding').value
        self.max_r       = self.get_parameter('max_obstacle_radius').value

        # ── EMA tracker ──────────────────────────────────────────
        self._tracker = EMATracker(
            alpha          = self.get_parameter('ema_alpha').value,
            match_dist     = self.get_parameter('ema_match_dist').value,
            max_miss_frames= int(self.get_parameter('ema_max_miss_frames').value),
        )

        # ── Robot pose (from /odom — always available) ────────────
        self._rx  = 0.0
        self._ry  = 0.0
        self._rth = 0.0
        self._odom_ready = False

        # ── Publishers ───────────────────────────────────────────
        self.obs_pub    = self.create_publisher(
            Float64MultiArray, '/obstacles', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/obstacle_markers', 10)

        # ── Subscriptions ────────────────────────────────────────
        self.create_subscription(Odometry,  '/odom',  self._odom_cb,  10)
        self.create_subscription(LaserScan, '/scan',  self._scan_cb,  10)

        self.get_logger().info('ObstacleDetector ready (EMA smoothing enabled).')

    # ── Odometry callback — updates robot pose ────────────────────
    def _odom_cb(self, msg: Odometry):
        self._rx  = msg.pose.pose.position.x
        self._ry  = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._rth = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._odom_ready = True

    # ── Main scan callback ────────────────────────────────────────
    def _scan_cb(self, msg: LaserScan):
        if not self._odom_ready:
            self._publish_empty()
            return

        # ── Filter scan ──────────────────────────────────────────
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        valid  = np.isfinite(ranges) & (ranges >= self.min_range) & \
                 (ranges <= self.max_range)
        ranges = ranges[valid]
        angles = angles[valid]

        if len(ranges) == 0:
            self._publish_empty()
            return

        # ── Polar → Cartesian in odom frame ──────────────────────
        xl = ranges * np.cos(angles)
        yl = ranges * np.sin(angles)
        cos_th = math.cos(self._rth)
        sin_th = math.sin(self._rth)
        xm = self._rx + cos_th * xl - sin_th * yl
        ym = self._ry + sin_th * xl + cos_th * yl

        points = np.column_stack([xm, ym])

        # ── Greedy BFS clustering ────────────────────────────────
        clusters = self._cluster(points)

        # ── Cluster → raw (cx, cy, r) ─────────────────────────────
        raw = []
        for pts in clusters:
            if len(pts) < self.min_pts or len(pts) > self.max_pts:
                continue
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
            r = float(np.max(dists)) + self.r_pad
            if r > self.max_r:
                continue
            raw.append((cx, cy, r))

        # ── EMA smoothing (pass robot pos for blind-zone persistence) ──
        smoothed = self._tracker.update(raw,
                                        robot_pos=(self._rx, self._ry))

        # ── Sort by distance to robot, keep max_obs ───────────────
        smoothed.sort(
            key=lambda o: (o[0] - self._rx)**2 + (o[1] - self._ry)**2)
        smoothed = smoothed[:self.max_obs]

        # ── Pad to max_obs ────────────────────────────────────────
        while len(smoothed) < self.max_obs:
            smoothed.append((1000.0, 1000.0, 0.1))

        # ── Publish ───────────────────────────────────────────────
        flat = []
        for (cx, cy, r) in smoothed:
            flat.extend([cx, cy, r])
        msg_out      = Float64MultiArray()
        msg_out.data = flat
        self.obs_pub.publish(msg_out)

        self._publish_markers(smoothed)

    # ── Greedy BFS clustering ─────────────────────────────────────
    def _cluster(self, points: np.ndarray):
        n = len(points)
        assigned = np.full(n, -1, dtype=int)
        cluster_id = 0
        for i in range(n):
            if assigned[i] != -1:
                continue
            assigned[i] = cluster_id
            queue = [i]
            while queue:
                curr = queue.pop()
                dists = np.hypot(
                    points[:, 0] - points[curr, 0],
                    points[:, 1] - points[curr, 1])
                neighbours = np.where(
                    (dists < self.cluster_tol) & (assigned == -1))[0]
                assigned[neighbours] = cluster_id
                queue.extend(neighbours.tolist())
            cluster_id += 1
        return [points[assigned == cid] for cid in range(cluster_id)]

    # ── Empty obstacle list ───────────────────────────────────────
    def _publish_empty(self):
        flat = [1000.0, 1000.0, 0.1] * self.max_obs
        msg = Float64MultiArray()
        msg.data = flat
        self.obs_pub.publish(msg)

    # ── RViz markers ─────────────────────────────────────────────
    def _publish_markers(self, obstacles):
        ma  = MarkerArray()
        now = self.get_clock().now().to_msg()

        del_marker        = Marker()
        del_marker.action = Marker.DELETEALL
        del_marker.header.frame_id = 'odom'
        del_marker.header.stamp    = now
        ma.markers.append(del_marker)

        for idx, (cx, cy, r) in enumerate(obstacles):
            if cx > 900:
                continue
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp    = now
            m.ns     = 'obstacles'
            m.id     = idx
            m.type   = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x  = cx
            m.pose.position.y  = cy
            m.pose.position.z  = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = 2.0 * r
            m.scale.y = 2.0 * r
            m.scale.z = 1.0
            m.color.r = 1.0
            m.color.g = 0.4
            m.color.b = 0.0
            m.color.a = 0.45
            m.lifetime.sec = 1
            ma.markers.append(m)

        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
