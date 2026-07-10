#!/usr/bin/env python3
"""
nmpc_controller.py — Nonlinear MPC with real-time obstacle avoidance
ME 5659 | Pradeep Sivaa Aiyanar

References:
  [1] Mayne et al. (2000) — Constrained MPC: Stability and Optimality
  [2] Qasim et al. (2024) — NMPC-Based Trajectory Tracking and Obstacle
      Avoidance for Mobile Robots (CasADi + circle constraints)

NO pre-planned trajectory.  The robot is given only a goal position.
At every timestep, the NMPC solves:

  min   Σ_{k=0}^{N-1} [ (x_k-x_g)^T Q (x_k-x_g) + u_k^T R u_k ]
         + (x_N-x_g)^T Q_N (x_N-x_g)
         + w_s * Σ_j s_j²          (slack penalty for obstacle softening)

  s.t.  x_{k+1}  = f(x_k, u_k)               (unicycle, Forward Euler)
        v ∈ [0, v_max],  ω ∈ [-ω_max, ω_max]
        ‖(x_k-ox_j, y_k-oy_j)‖² ≥ (r_j+margin)² - s_j  ∀ j,k
        s_j ≥ 0                              (slack ≥ 0)
        x_0 = x_current

Soft obstacle constraints prevent infeasibility in dense environments.
At runtime, the max_obs closest obstacles are selected from /obstacles.

Subscribes:   /odom            nav_msgs/Odometry
              /obstacles        std_msgs/Float64MultiArray
              /goal_pose        geometry_msgs/PoseStamped  (optional)
Publishes:    /cmd_vel          geometry_msgs/Twist
              /nmpc_pred_path   nav_msgs/Path  (predicted horizon, RViz)
              /actual_path      nav_msgs/Path  (driven path, RViz)
              /goal_marker      visualization_msgs/Marker
Robot pose comes from the TF tree (map → base_link), which is the
SLAM-corrected pose. This replaces raw wheel odometry and ensures the
NMPC plans in the consistent map frame.
"""

import rclpy
from rclpy.node import Node
import rclpy.duration
from rclpy.time import Time as RosTime
import numpy as np
import math
import time as _time   # wall-clock timer for RTI profiling
import casadi as ca

import tf2_ros

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker


# ── Extended Kalman Filter for unicycle pose [x, y, θ] ───────────────────────
class EKFPose:
    """
    EKF for a differential-drive robot with state x = [x, y, θ].

    Prediction:  unicycle model driven by the last applied [v, ω]
    Update:      direct pose measurement from odometry / TF

    Noise tuning (all in metres / radians):
      Q  — process noise   (how much we trust the model)
      R  — measurement noise (how much we trust the sensor)
    """

    def __init__(self,
                 q_xy: float = 0.02,   # position process noise (m)
                 q_th: float = 0.005,  # heading process noise  (rad)
                 r_xy: float = 0.05,   # position meas. noise   (m)
                 r_th: float = 0.02):  # heading meas. noise    (rad)
        self.Q = np.diag([q_xy, q_xy, q_th])
        self.R = np.diag([r_xy, r_xy, r_th])
        self.x = np.zeros(3)        # state estimate  [x, y, θ]
        self.P = np.eye(3) * 0.5    # covariance (start uncertain)
        self._ready = False

    def initialize(self, x0: np.ndarray):
        self.x = x0.copy()
        self.P = np.eye(3) * 0.1
        self._ready = True

    @property
    def ready(self):
        return self._ready

    def predict(self, v: float, omega: float, dt: float):
        """Unicycle prediction step."""
        if not self._ready:
            return
        x, y, th = self.x
        # State prediction
        self.x = np.array([
            x  + v * math.cos(th) * dt,
            y  + v * math.sin(th) * dt,
            th + omega * dt,
        ])
        self.x[2] = math.atan2(math.sin(self.x[2]), math.cos(self.x[2]))

        # Linearised state-transition Jacobian  F = ∂f/∂x
        F = np.array([
            [1.0,  0.0, -v * math.sin(th) * dt],
            [0.0,  1.0,  v * math.cos(th) * dt],
            [0.0,  0.0,  1.0],
        ])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray):
        """
        Measurement update.  z = [x_meas, y_meas, θ_meas] from odom / TF.
        """
        if not self._ready:
            self.initialize(z)
            return

        H = np.eye(3)                          # measurement model is identity
        innov = z - H @ self.x
        innov[2] = math.atan2(math.sin(innov[2]), math.cos(innov[2]))  # wrap θ

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)   # Kalman gain

        self.x = self.x + K @ innov
        self.x[2] = math.atan2(math.sin(self.x[2]), math.cos(self.x[2]))
        self.P = (np.eye(3) - K @ H) @ self.P


class NMPCController(Node):

    def __init__(self):
        super().__init__('nmpc_controller')

        # ── Parameters ───────────────────────────────────────────
        self.declare_parameter('N',              20)
        self.declare_parameter('Ts',             0.1)
        self.declare_parameter('Q_diag',         [12.0, 12.0, 0.5])
        self.declare_parameter('R_diag',         [0.5, 1.0])
        self.declare_parameter('R_du_diag',      [0.3, 0.1])
        self.declare_parameter('q_terminal',     10.0)
        self.declare_parameter('v_max',          1.0)
        self.declare_parameter('v_min',          0.0)
        self.declare_parameter('omega_max',      1.2)
        self.declare_parameter('safety_margin',  0.30)
        self.declare_parameter('max_obstacles',  6)
        self.declare_parameter('slack_penalty',  5000.0)
        self.declare_parameter('goal_x',         13.0)
        self.declare_parameter('goal_y',         0.0)
        self.declare_parameter('goal_tol',       0.35)
        # "Arrived & settled" fallback: the speed taper + sim deadband can stall
        # the robot a few cm short of the tight goal_tol.  If it is within
        # goal_settle_tol AND has stopped moving for settle_cycles, declare
        # success at the honest settled distance (so it never hangs near the goal).
        self.declare_parameter('goal_settle_tol', 0.12)   # m — "close enough" band
        self.declare_parameter('settle_speed',    0.02)   # m/s — |v| below = stopped
        self.declare_parameter('settle_cycles',   10)     # cycles stopped => settled
        self.declare_parameter('blocked_cycles',  25)     # cycles stopped & FAR => blocked
        # Reverse-capable terminal controller: for the final approach, hand off
        # from NMPC/MPCC to a simple go-to-point law that can DRIVE or REVERSE
        # straight to the goal — fixes forward-only hunting when a goal is
        # approached at an awkward angle (it just backs up instead of looping).
        self.declare_parameter('use_terminal_ctrl', True)
        self.declare_parameter('terminal_radius',   0.5)  # m — engage within this range
        self.declare_parameter('term_k_rho',        1.5)  # distance gain (v = k_rho*rho)
        self.declare_parameter('term_k_alpha',      2.5)  # heading gain  (w = k_alpha*a)
        self.declare_parameter('v_rev_max',         0.2)  # m/s — max reverse speed
        self.declare_parameter('solver_max_iter',200)
        self.declare_parameter('use_warm_start', True)
        self.declare_parameter('use_rti',        True)
        # ── Global-path (carrot) tracking ────────────────────────
        # Follow a lookahead point along /global_path (A*) instead of aiming
        # straight at the far goal.  This is the global(A*)+local(NMPC) split.
        self.declare_parameter('use_global_path', True)
        self.declare_parameter('lookahead_dist',  1.2)   # m — carrot distance ahead
        # ── SLAM pose fusion (Phase 2: drift-corrected map-frame control) ──
        # Fuse SLAM map->base_link into the EKF so the NMPC controls in the
        # drift-free map frame and reaches the true world goal.  Gating rejects
        # jumps so map drift/loop-closure cannot destabilise the control state.
        self.declare_parameter('fuse_slam_pose',  True)
        self.declare_parameter('fuse_gate_pos',   0.5)   # m   — reject jumps beyond this
        self.declare_parameter('fuse_gate_th',    0.5)   # rad
        self.declare_parameter('fuse_max_rejects', 10)   # cycles before force re-sync
        # ── MPCC (Model Predictive Contouring Control) ───────────
        # Track the A* path as a contour instead of chasing a carrot: add an
        # arc-length progress state and penalise contour (lateral) + lag
        # (longitudinal) error while maximising progress.  Carrot NMPC remains
        # the fallback (no path / near goal / solver fail) — flip use_mpcc off
        # to revert entirely.
        self.declare_parameter('use_mpcc',    True)
        self.declare_parameter('mpcc_qc',     30.0)   # contour-error weight (stay ON path)
        self.declare_parameter('mpcc_ql',     10.0)   # lag-error weight (progress tracks robot)
        self.declare_parameter('mpcc_qprog',  6.0)    # progress reward (move along path)
        self.declare_parameter('mpcc_qth',    1.0)    # heading-alignment weight (small)
        self.declare_parameter('mpcc_reach',  3.0)    # m — path window fit ahead of robot
        self.declare_parameter('mpcc_max_iter', 40)   # cap IPOPT iters → bound solve time
        # ── Rotate-in-place (turn toward off-heading goals before driving) ──
        self.declare_parameter('rotate_in_place', True)
        self.declare_parameter('rotate_thresh',   1.2)   # rad — turn in place if |herr|>this
        self.declare_parameter('rotate_gain',     1.5)   # rad/s per rad of heading error

        self.N          = int(self.get_parameter('N').value)
        self.Ts         = self.get_parameter('Ts').value
        Q_d             = list(self.get_parameter('Q_diag').value)
        R_d             = list(self.get_parameter('R_diag').value)
        R_du_d          = list(self.get_parameter('R_du_diag').value)
        q_term          = self.get_parameter('q_terminal').value
        self.v_max      = self.get_parameter('v_max').value
        self.v_min      = self.get_parameter('v_min').value
        self.omega_max  = self.get_parameter('omega_max').value
        self.margin     = self.get_parameter('safety_margin').value
        self.max_obs    = int(self.get_parameter('max_obstacles').value)
        self.slack_w    = self.get_parameter('slack_penalty').value
        self.goal       = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            0.0,    # goal heading — robot just needs to arrive, not align
        ])
        self.goal_tol   = self.get_parameter('goal_tol').value
        self.goal_settle_tol = float(self.get_parameter('goal_settle_tol').value)
        self.settle_speed    = float(self.get_parameter('settle_speed').value)
        self.settle_cycles   = int(self.get_parameter('settle_cycles').value)
        self.blocked_cycles  = int(self.get_parameter('blocked_cycles').value)
        self.use_terminal_ctrl = bool(self.get_parameter('use_terminal_ctrl').value)
        self.terminal_radius   = float(self.get_parameter('terminal_radius').value)
        self.term_k_rho        = float(self.get_parameter('term_k_rho').value)
        self.term_k_alpha      = float(self.get_parameter('term_k_alpha').value)
        self.v_rev_max         = float(self.get_parameter('v_rev_max').value)
        max_iter        = int(self.get_parameter('solver_max_iter').value)
        self.warm_start = self.get_parameter('use_warm_start').value
        self.use_rti    = self.get_parameter('use_rti').value
        self.use_global_path = bool(self.get_parameter('use_global_path').value)
        self.lookahead       = float(self.get_parameter('lookahead_dist').value)
        self.fuse_slam       = bool(self.get_parameter('fuse_slam_pose').value)
        self.fuse_gate_pos   = float(self.get_parameter('fuse_gate_pos').value)
        self.fuse_gate_th    = float(self.get_parameter('fuse_gate_th').value)
        self.fuse_max_rej    = int(self.get_parameter('fuse_max_rejects').value)
        self.use_mpcc        = bool(self.get_parameter('use_mpcc').value)
        self.mpcc_qc         = float(self.get_parameter('mpcc_qc').value)
        self.mpcc_ql         = float(self.get_parameter('mpcc_ql').value)
        self.mpcc_qprog      = float(self.get_parameter('mpcc_qprog').value)
        self.mpcc_qth        = float(self.get_parameter('mpcc_qth').value)
        self.mpcc_reach      = float(self.get_parameter('mpcc_reach').value)
        self.mpcc_max_iter   = int(self.get_parameter('mpcc_max_iter').value)
        self.rotate_in_place = bool(self.get_parameter('rotate_in_place').value)
        self.rotate_thresh   = float(self.get_parameter('rotate_thresh').value)
        self.rotate_gain     = float(self.get_parameter('rotate_gain').value)

        # ── Build CasADi solver ──────────────────────────────────
        Q    = np.diag(Q_d)
        Q_N  = q_term * Q
        R    = np.diag(R_d)
        R_du = np.diag(R_du_d)
        self._build_solver(Q, Q_N, R, R_du, max_iter)
        if self.use_mpcc:
            self._build_mpcc_solver(R, R_du, min(max_iter, self.mpcc_max_iter))

        # ── TF2 (SLAM-corrected pose: map → base_link) ───────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── EKF pose filter ──────────────────────────────────────
        self._ekf = EKFPose(q_xy=0.02, q_th=0.005, r_xy=0.05, r_th=0.02)
        # Last applied command (v, ω) — used for EKF prediction step
        self._last_v     = 0.0
        self._last_omega = 0.0
        # Measured wheel-odometry body velocity — drives the EKF prediction step
        self._odom_v     = 0.0
        self._odom_w     = 0.0

        # ── State / warm-start storage ───────────────────────────
        self._x_curr    = np.zeros(3)    # [x, y, θ] — EKF-filtered pose
        self._all_obstacles = np.zeros((10, 3))  # raw from detector (up to 10)
        self._all_obstacles[:, 0] = 1000.0
        self._all_obstacles[:, 1] = 1000.0
        self._all_obstacles[:, 2] = 0.1

        # Active obstacles passed to NMPC (closest max_obs)
        self._obstacles = np.zeros((self.max_obs, 3))
        self._obstacles[:, 0] = 1000.0
        self._obstacles[:, 1] = 1000.0
        self._obstacles[:, 2] = 0.1

        # Warm-start: previous solution
        self._U_prev      = np.zeros((2, self.N))
        self._X_prev      = np.zeros((3, self.N + 1))
        # Last applied control action — used for Δu input rate penalty
        self._u_prev_ctrl = np.zeros(2)
        # Per-timestep per-obstacle slack warm-start (shape: max_obs × N)
        self._S_prev      = np.zeros((self.max_obs, self.N))
        # MPCC warm-start: progress state + progress-rate + augmented X/U/slack
        self._mU_prev     = np.zeros((2, self.N))
        self._mX_prev     = np.zeros((3, self.N + 1))
        self._mPp_prev    = np.zeros(self.N + 1)
        self._mVp_prev    = np.zeros(self.N)
        self._mS_prev     = np.zeros((self.max_obs, self.N))
        # EMA command smoother: filtered v and ω published to robot
        # EMA smoother only on v (linear velocity) — suppresses start-stop chattering.
        # ω (angular) is passed through UNFILTERED so the robot can turn sharply
        # the moment IPOPT commands it; any EMA lag on ω delays the avoidance
        # turn by α·Ts seconds, causing the robot to clip obstacles at speed.
        self._v_smooth   = 0.0
        self._v_alpha    = 0.5   # EMA weight for v (τ ≈ 0.14 s = 1.4 control steps)

        # Global path from A* planner (Nx2, control/odom frame). None until the
        # first /global_path arrives — controller then falls back to the raw goal.
        self._global_path   = None

        # SLAM-fusion state
        self._fuse_rejects  = 0
        self._map_from_odom = None    # cached (tx,ty,yaw) map<-odom for obstacle TF

        self._initialized   = False
        self._goal_reached  = False
        self._min_dist_goal = float('inf')   # closest approach to goal (accuracy metric)
        self._stuck_ctr     = 0              # consecutive "stopped" cycles (settle/blocked)
        self._blocked_warned = False         # warned once that the goal is unreachable
        self._rotating      = False          # rotate-in-place hysteresis flag
        self._consec_fails  = 0
        self._actual_path        = Path()
        # Control frame: the drift-corrected SLAM map frame when fusion is on
        # (Phase 2), else raw odom.  Everything the NMPC reasons about — pose,
        # goal, path, obstacles, viz — lives in this frame.
        self._active_frame       = 'map' if self.fuse_slam else 'odom'
        self._actual_path.header.frame_id = self._active_frame

        # ── RTI solve-time profiling ──────────────────────────────
        # Stores last 100 solve times (ms) for mean/max reporting
        self._solve_times   = []
        self._solve_budget  = self.Ts * 1000.0   # ms — full control period

        # ── ROS subscriptions ────────────────────────────────────
        # /odom kept for fallback only; primary pose comes from TF
        self.create_subscription(Odometry,          '/odom',       self._odom_cb,  10)
        self.create_subscription(Float64MultiArray, '/obstacles',  self._obs_cb,   10)
        self.create_subscription(PoseStamped,       '/goal_pose',  self._goal_cb,  10)
        self.create_subscription(Path,              '/global_path', self._path_cb, 10)

        # ── ROS publishers ───────────────────────────────────────
        self.cmd_pub        = self.create_publisher(Twist,  '/cmd_vel',         10)
        self.pred_path_pub  = self.create_publisher(Path,   '/nmpc_pred_path',  10)
        self.actual_path_pub= self.create_publisher(Path,   '/actual_path',     10)
        self.goal_marker_pub= self.create_publisher(Marker, '/goal_marker',     10)

        # ── Control timer at 1/Ts Hz ─────────────────────────────
        self.create_timer(self.Ts, self._control_loop)
        self.create_timer(2.0,     self._diag_cb)

        # Publish goal marker once on startup
        self.create_timer(2.0, self._publish_goal_marker_once)
        self._goal_marker_published = False

        self.get_logger().info(
            f'NMPC ready | N={self.N} | Ts={self.Ts}s | '
            f'goal=({self.goal[0]:.1f},{self.goal[1]:.1f})'
        )

    # ──────────────────────────────────────────────────────────────
    def _build_solver(self, Q, Q_N, R, R_du, max_iter):
        """Construct the CasADi Opti NLP once at startup."""
        N   = self.N
        Ts  = self.Ts
        M   = self.max_obs

        opti = ca.Opti()

        # Decision variables
        X = opti.variable(3, N + 1)   # predicted states
        U = opti.variable(2, N)       # predicted controls
        S = opti.variable(M, N)       # per-timestep per-obstacle slack (M×N)

        # Parameters (updated each solve)
        x0_p    = opti.parameter(3)           # current state
        xg_p    = opti.parameter(3)           # goal state
        u_prev_p = opti.parameter(2)          # last applied control (for Δu cost)
        ox_p    = opti.parameter(M)           # obstacle x
        oy_p    = opti.parameter(M)           # obstacle y
        or_p    = opti.parameter(M)           # obstacle radius

        # ── Cost ─────────────────────────────────────────────────
        # Input rate penalty (Δu): penalise successive control changes to
        # suppress velocity chattering near obstacles.
        # Based on: De Souza et al. (2022) "Smooth Reference Tracking of a
        # Mobile Robot using NMPC" — adding ||u_k - u_{k-1}||²_R_du to the
        # stage cost directly reduces chattering without changing constraints.
        cost = 0
        for k in range(N):
            dx = X[:, k] - xg_p
            dx_wrapped = ca.vertcat(
                dx[0],
                dx[1],
                ca.atan2(ca.sin(dx[2]), ca.cos(dx[2]))
            )
            cost += dx_wrapped.T @ ca.DM(Q) @ dx_wrapped
            cost += U[:, k].T @ ca.DM(R) @ U[:, k]
            # Δu penalty: first step uses last applied control, rest use prev step
            u_km1  = u_prev_p if k == 0 else U[:, k - 1]
            du     = U[:, k] - u_km1
            cost  += du.T @ ca.DM(R_du) @ du

        # Terminal cost
        dx_T = X[:, N] - xg_p
        dx_T_wrapped = ca.vertcat(
            dx_T[0],
            dx_T[1],
            ca.atan2(ca.sin(dx_T[2]), ca.cos(dx_T[2]))
        )
        cost += dx_T_wrapped.T @ ca.DM(Q_N) @ dx_T_wrapped

        # ── Slack penalty (per-timestep, per-obstacle) ────────────
        # High penalty on any constraint violation — keeps S near zero
        cost += self.slack_w * ca.sumsqr(S)

        # ── Repulsive exponential potential ────────────────────────
        # The hard constraint below has zero gradient until the trajectory
        # enters the exclusion zone. This soft potential adds a nonzero,
        # distance-decaying gradient everywhere (~10% strength at 1.5 m past
        # the boundary) so IPOPT steers around obstacles pre-emptively.
        rep_w     = 200.0   # weight per (horizon step × obstacle)
        rep_alpha = 1.5     # 1/m: ~10% gradient strength at 1.5m past constraint
        for j in range(M):
            for k in range(N + 1):
                dx_obs = X[0, k] - ox_p[j]
                dy_obs = X[1, k] - oy_p[j]
                dist   = ca.sqrt(dx_obs**2 + dy_obs**2 + 1e-6)
                r_safe = or_p[j] + self.margin
                cost  += rep_w * ca.exp(-rep_alpha * (dist - r_safe))

        opti.minimize(cost)

        # ── Dynamics constraints (Forward-Euler unicycle) ─────────
        for k in range(N):
            x_next = X[:, k] + Ts * ca.vertcat(
                U[0, k] * ca.cos(X[2, k]),
                U[0, k] * ca.sin(X[2, k]),
                U[1, k]
            )
            opti.subject_to(X[:, k + 1] == x_next)

        # ── Initial state constraint ──────────────────────────────
        opti.subject_to(X[:, 0] == x0_p)

        # ── Input constraints ─────────────────────────────────────
        opti.subject_to(opti.bounded(self.v_min,      U[0, :], self.v_max))
        opti.subject_to(opti.bounded(-self.omega_max, U[1, :], self.omega_max))

        # ── Soft obstacle avoidance — per-timestep per-obstacle slack ──
        # dist_sq(k,j) >= (r_j + margin)² - S[j,k-1],  S[j,k-1] >= 0
        # Distance-squared keeps a nonzero gradient outside the exclusion zone
        # (unlike an fmax barrier), so IPOPT avoids obstacles pre-emptively.
        # Per-timestep slack S[j,k] (M×N) avoids the trivial per-obstacle
        # minimum where one large slack satisfies every step at once.
        # Inactive obstacles sit at (1000,1000): satisfied with S=0 at no cost.
        margin = self.margin
        opti.subject_to(opti.bounded(0, ca.vec(S), ca.inf))   # flatten S for the bound
        for j in range(M):
            for k in range(1, N + 1):
                dist_sq = (X[0, k] - ox_p[j])**2 + (X[1, k] - oy_p[j])**2
                r_sq    = (or_p[j] + margin)**2
                opti.subject_to(dist_sq >= r_sq - S[j, k - 1])

        # ── Solver selection ─────────────────────────────────────
        if self.use_rti:
            # RTI (Real-Time Iteration, Diehl et al. 2005): exactly 3 SQP steps
            # via the built-in qrqp QP solver, ~8-15 ms on this hardware.
            # error_on_fail=False so we always read the best available iterate.
            opti.solver('sqpmethod', {
                'max_iter':              3,        # 3 SQP steps (RTI variant)
                'qpsol':                 'qrqp',
                'qpsol_options': {
                    'print_iter':    False,
                    'print_header':  False,
                    'error_on_fail': False,
                },
                'hessian_approximation': 'exact',
                'print_header':          False,
                'print_time':            False,
                'print_iteration':       False,
                'verbose':               False,
                'error_on_fail':         False,
            })
        else:
            # Warm-started IPOPT with early exit (acceptable_tol after a few
            # iterations): ~25-40 ms per solve vs ~800 ms at max_iter=300.
            # Acceptable-quality iterates are fine for receding-horizon control.
            opti.solver('ipopt', {
                'print_time': 0,
                'ipopt': {
                    'max_iter':              max_iter,
                    'print_level':           0,
                    'sb':                    'yes',
                    'tol':                   1e-4,
                    'acceptable_tol':        1e-3,
                    'acceptable_iter':       5,       # exit after 5 consecutive acceptable iters (~15-20 total)
                    'warm_start_init_point': 'yes',
                }
            })

        # Initialise slack to zero — warm feasible start
        opti.set_initial(S, np.zeros((M, N)))

        # Store references for solve()
        self._opti     = opti
        self._X_var    = X
        self._U_var    = U
        self._S_var    = S
        self._x0_p     = x0_p
        self._xg_p     = xg_p
        self._u_prev_p = u_prev_p
        self._ox_p     = ox_p
        self._oy_p     = oy_p
        self._or_p     = or_p

        self.get_logger().info('CasADi NMPC solver built.')

    def _build_mpcc_solver(self, R, R_du, max_iter):
        """Model Predictive Contouring Control solver.  An arc-length progress
        state s advances along a cubic parametrisation (px(s), py(s)) of the A*
        path (coefficients set each cycle); the cost penalises CONTOUR (lateral)
        + LAG (longitudinal) error to the path while REWARDING progress, so the
        robot follows the path itself instead of chasing a moving carrot point.
        Refs: Lam/Liniger MPCC; Brito et al. (2019) MPCC collision avoidance."""
        N, Ts, M = self.N, self.Ts, self.max_obs
        opti = ca.Opti()

        X  = opti.variable(3, N + 1)     # [x, y, psi]
        U  = opti.variable(2, N)         # [v, omega]
        Pp = opti.variable(1, N + 1)     # arc-length progress (local σ, starts at 0)
        Vp = opti.variable(1, N)         # progress rate  ds/dt
        S  = opti.variable(M, N)         # obstacle slack

        x0_p     = opti.parameter(3)
        cx_p     = opti.parameter(4)     # px(σ) = cx0 + cx1 σ + cx2 σ² + cx3 σ³
        cy_p     = opti.parameter(4)
        smax_p   = opti.parameter(1)     # end of the fitted path window
        u_prev_p = opti.parameter(2)
        ox_p     = opti.parameter(M)
        oy_p     = opti.parameter(M)
        or_p     = opti.parameter(M)

        qc, ql, qprog, qth = self.mpcc_qc, self.mpcc_ql, self.mpcc_qprog, self.mpcc_qth
        R_dm, Rdu_dm = ca.DM(R), ca.DM(R_du)

        def ref(s):
            rx  = cx_p[0] + cx_p[1]*s + cx_p[2]*s**2 + cx_p[3]*s**3
            ry  = cy_p[0] + cy_p[1]*s + cy_p[2]*s**2 + cy_p[3]*s**3
            drx = cx_p[1] + 2*cx_p[2]*s + 3*cx_p[3]*s**2
            dry = cy_p[1] + 2*cy_p[2]*s + 3*cy_p[3]*s**2
            return rx, ry, ca.atan2(dry, drx)

        cost = 0
        for k in range(N):
            rx, ry, phi = ref(Pp[0, k])
            ex, ey = X[0, k] - rx, X[1, k] - ry
            e_c =  ca.sin(phi)*ex - ca.cos(phi)*ey        # contour (lateral) error
            e_l = -ca.cos(phi)*ex - ca.sin(phi)*ey        # lag (longitudinal) error
            cost += qc*e_c**2 + ql*e_l**2
            dth = ca.atan2(ca.sin(X[2, k]-phi), ca.cos(X[2, k]-phi))
            cost += qth*dth**2
            cost += U[:, k].T @ R_dm @ U[:, k]
            u_km1 = u_prev_p if k == 0 else U[:, k-1]
            du = U[:, k] - u_km1
            cost += du.T @ Rdu_dm @ du
        rxN, ryN, phiN = ref(Pp[0, N])
        exN, eyN = X[0, N] - rxN, X[1, N] - ryN
        cost += qc*(ca.sin(phiN)*exN - ca.cos(phiN)*eyN)**2
        cost += ql*(-ca.cos(phiN)*exN - ca.sin(phiN)*eyN)**2
        cost -= qprog * Pp[0, N]                           # reward total progress

        # Obstacle slack + soft repulsive field (same shaping as the carrot NMPC)
        cost += self.slack_w * ca.sumsqr(S)
        rep_w, rep_alpha = 200.0, 1.5
        for j in range(M):
            for k in range(N + 1):
                dist = ca.sqrt((X[0, k]-ox_p[j])**2 + (X[1, k]-oy_p[j])**2 + 1e-6)
                cost += rep_w * ca.exp(-rep_alpha * (dist - (or_p[j] + self.margin)))

        opti.minimize(cost)

        # Dynamics: unicycle + progress integrator
        for k in range(N):
            x_next = X[:, k] + Ts * ca.vertcat(
                U[0, k]*ca.cos(X[2, k]),
                U[0, k]*ca.sin(X[2, k]),
                U[1, k])
            opti.subject_to(X[:, k+1] == x_next)
            opti.subject_to(Pp[0, k+1] == Pp[0, k] + Ts*Vp[0, k])

        opti.subject_to(X[:, 0] == x0_p)
        opti.subject_to(Pp[0, 0] == 0.0)
        opti.subject_to(opti.bounded(self.v_min,      U[0, :], self.v_max))
        opti.subject_to(opti.bounded(-self.omega_max, U[1, :], self.omega_max))
        opti.subject_to(opti.bounded(0.0, Vp, self.v_max))    # forward progress only
        opti.subject_to(Pp >= 0.0)
        opti.subject_to(Pp <= smax_p)                          # stay within fitted window

        opti.subject_to(opti.bounded(0, ca.vec(S), ca.inf))
        for j in range(M):
            for k in range(1, N + 1):
                dist_sq = (X[0, k]-ox_p[j])**2 + (X[1, k]-oy_p[j])**2
                opti.subject_to(dist_sq >= (or_p[j] + self.margin)**2 - S[j, k-1])

        opti.solver('ipopt', {
            'print_time': 0,
            'ipopt': {
                'max_iter':              max_iter,
                'print_level':           0,
                'sb':                    'yes',
                'tol':                   1e-4,
                'acceptable_tol':        1e-3,
                'acceptable_iter':       5,
                'warm_start_init_point': 'yes',
            }
        })
        opti.set_initial(S, np.zeros((M, N)))

        self._m_opti = opti
        self._mX_var, self._mU_var, self._mS_var = X, U, S
        self._mPp_var, self._mVp_var = Pp, Vp
        self._m_x0, self._m_cx, self._m_cy, self._m_smax = x0_p, cx_p, cy_p, smax_p
        self._m_uprev, self._m_ox, self._m_oy, self._m_or = u_prev_p, ox_p, oy_p, or_p
        self.get_logger().info('CasADi MPCC solver built.')

    # ── Callbacks ─────────────────────────────────────────────────
    def _odom_cb(self, msg):
        """Odometry now only signals that the sim is live so the control loop can
        start.  The EKF measurement comes from the SLAM map->base_link pose,
        fused (gated) in the control loop — see _fuse_gated / _get_pose_from_tf.
        We no longer feed raw odom pose here: that pinned the state to the
        drifting odom frame, which is exactly what made the robot miss the
        world goal."""
        self._odom_v = msg.twist.twist.linear.x     # wheel-odom body velocity
        self._odom_w = msg.twist.twist.angular.z    # (drives EKF prediction)
        if not self._initialized:
            self._initialized = True     # allow control loop to start

    def _get_pose_from_tf(self):
        """Get robot pose: prefer map frame (SLAM), fall back to odom.
        Rejects transforms older than a few control cycles to avoid stale SLAM
        data (TF_OLD_DATA) corrupting the EKF state estimate."""
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        max_age = 5.0 * self.Ts   # tolerate a little SLAM TF lag

        for frame in ('map', 'odom'):
            try:
                t = self._tf_buffer.lookup_transform(
                    frame, 'base_link',
                    RosTime(),
                    timeout=rclpy.duration.Duration(seconds=0.05))

                # Reject stale transforms — SLAM often publishes lagged TF
                tf_sec = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
                if now_sec - tf_sec > max_age:
                    continue

                x  = t.transform.translation.x
                y  = t.transform.translation.y
                q  = t.transform.rotation
                th = math.atan2(2.0*(q.w*q.z + q.x*q.y),
                                1.0 - 2.0*(q.y*q.y + q.z*q.z))
                return np.array([x, y, th]), True
            except Exception:
                continue
        return self._x_curr, False

    def _fuse_gated(self, z):
        """Fuse an absolute pose measurement z=[x,y,θ] (map frame) into the EKF,
        gating out large single-cycle jumps so a bad scan match / TF spike cannot
        corrupt the smooth control state.  A *sustained* offset (e.g. a real loop
        closure) is force-accepted after fuse_max_rej cycles so the estimate
        re-syncs instead of diverging from SLAM forever."""
        if not self._ekf.ready:
            self._ekf.initialize(z)
            return
        dxy = math.hypot(z[0] - self._ekf.x[0], z[1] - self._ekf.x[1])
        dth = abs(math.atan2(math.sin(z[2] - self._ekf.x[2]),
                             math.cos(z[2] - self._ekf.x[2])))
        if dxy > self.fuse_gate_pos or dth > self.fuse_gate_th:
            self._fuse_rejects += 1
            if self._fuse_rejects < self.fuse_max_rej:
                return                      # transient spike — reject, keep predicting
            self._fuse_rejects = 0          # sustained — accept to re-sync
        else:
            self._fuse_rejects = 0
        self._ekf.update(z)

    def _lookup_map_from_odom(self):
        """Return (tx,ty,yaw) of the map←odom transform, used to express the
        odom-frame /obstacles in the map control frame.  None if unavailable
        (caller then assumes identity — valid at spawn when map≈odom)."""
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'odom', RosTime(),
                timeout=rclpy.duration.Duration(seconds=0.02))
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.transform.translation.x, t.transform.translation.y, yaw)

    def _record_actual_path(self):
        """Append the current (map-frame) pose to the driven-path trail."""
        ps = PoseStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = self._active_frame
        ps.pose.position.x = float(self._x_curr[0])
        ps.pose.position.y = float(self._x_curr[1])
        self._actual_path.header.frame_id = self._active_frame
        self._actual_path.poses.append(ps)
        if len(self._actual_path.poses) > 3000:
            self._actual_path.poses.pop(0)

    def _obs_cb(self, msg):
        data = np.array(msg.data).reshape(-1, 3)
        n = min(len(data), 10)
        self._all_obstacles[:n] = data[:n]
        if n < 10:
            self._all_obstacles[n:, 0] = 1000.0
            self._all_obstacles[n:, 1] = 1000.0
            self._all_obstacles[n:, 2] = 0.1

    def _select_closest_obstacles(self):
        """Select the max_obs closest real obstacles to the robot."""
        rx, ry = self._x_curr[0], self._x_curr[1]
        real = self._all_obstacles[self._all_obstacles[:, 0] < 900]

        out = np.zeros((self.max_obs, 3))
        out[:, 0] = 1000.0
        out[:, 1] = 1000.0
        out[:, 2] = 0.1

        if len(real) == 0:
            return out

        # /obstacles arrive in the odom frame, but the control frame is now map.
        # Express them in map so the NMPC constraint ‖pose_map − obs‖ is
        # frame-consistent.  Identity fallback is valid at spawn (map≈odom).
        if self.fuse_slam and self._map_from_odom is not None:
            tx, ty, yaw = self._map_from_odom
            c, s = math.cos(yaw), math.sin(yaw)
            ox = tx + c * real[:, 0] - s * real[:, 1]
            oy = ty + s * real[:, 0] + c * real[:, 1]
            real = np.column_stack([ox, oy, real[:, 2]])

        dists = np.hypot(real[:, 0] - rx, real[:, 1] - ry)
        order = np.argsort(dists)
        top = real[order[:self.max_obs]]
        n = len(top)
        out[:n] = top
        return out

    def _goal_cb(self, msg):
        """Update goal from RViz '2D Nav Goal'.  Stored in the control frame
        (map); transform if RViz publishes it in a different frame."""
        gx, gy = msg.pose.position.x, msg.pose.position.y
        src = msg.header.frame_id
        if src and src not in (self._active_frame, ''):
            try:
                t = self._tf_buffer.lookup_transform(
                    self._active_frame, src, RosTime(),
                    timeout=rclpy.duration.Duration(seconds=0.2))
                q = t.transform.rotation
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                c, s = math.cos(yaw), math.sin(yaw)
                gx, gy = (t.transform.translation.x + c * gx - s * gy,
                          t.transform.translation.y + s * gx + c * gy)
            except Exception:
                self.get_logger().warn(
                    f'Goal frame "{src}" has no TF to {self._active_frame}; using raw values')
        self.goal[0] = gx
        self.goal[1] = gy
        self._goal_reached = False
        self._min_dist_goal = float('inf')     # reset accuracy metric for the new goal
        self._stuck_ctr = 0                    # reset settle/blocked detector for new goal
        self._blocked_warned = False
        self.get_logger().info(
            f'New goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f}) [{self._active_frame}]')
        self._publish_goal_marker(self.goal[0], self.goal[1])

    def _path_cb(self, msg: Path):
        """Store the latest A* global path (control/odom frame) for carrot
        tracking.  An empty path clears it and the controller falls back to
        aiming directly at the goal."""
        if len(msg.poses) < 2:
            self._global_path = None
            return
        self._global_path = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses])

    # ── Global-path carrot (lookahead) ────────────────────────────
    def _carrot_on_path(self):
        """Pure-pursuit style lookahead: project the robot onto the global
        path, then walk forward by `lookahead` metres and return that point.
        With string-pulled (sparse) waypoints we project onto segments, not
        just vertices, so the carrot is smooth along long straight runs."""
        path = self._global_path
        rx, ry = self._x_curr[0], self._x_curr[1]

        # ── Nearest point across all segments (project onto each) ──
        best_d2, best_i, best_pt = float('inf'), 0, (path[0, 0], path[0, 1])
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            dx, dy = x2 - x1, y2 - y1
            seg2 = dx * dx + dy * dy
            t = 0.0 if seg2 < 1e-9 else max(0.0, min(1.0, ((rx - x1) * dx + (ry - y1) * dy) / seg2))
            px, py = x1 + t * dx, y1 + t * dy
            d2 = (rx - px) ** 2 + (ry - py) ** 2
            if d2 < best_d2:
                best_d2, best_i, best_pt = d2, i, (px, py)

        # ── Walk forward `lookahead` metres from the projected point ──
        acc = 0.0
        px, py = best_pt
        for i in range(best_i, len(path) - 1):
            x1, y1 = (px, py) if i == best_i else (path[i, 0], path[i, 1])
            x2, y2 = path[i + 1]
            seg = math.hypot(x2 - x1, y2 - y1)
            if acc + seg >= self.lookahead:
                r = (self.lookahead - acc) / seg if seg > 1e-6 else 1.0
                return (x1 + r * (x2 - x1), y1 + r * (y2 - y1))
            acc += seg
        # Past the end of the path -> aim at the final waypoint (the goal)
        return (float(path[-1, 0]), float(path[-1, 1]))

    def _get_target(self):
        """NMPC reference [x, y, theta].  Target = carrot on /global_path when
        available (global A* + local NMPC), else the raw goal (pure
        goal-directed fallback).  Heading reference points toward the target."""
        if (not self.use_global_path) or self._global_path is None:
            cx, cy = float(self.goal[0]), float(self.goal[1])
        else:
            cx, cy = self._carrot_on_path()
        theta_ref = math.atan2(cy - self._x_curr[1], cx - self._x_curr[0])
        return np.array([cx, cy, theta_ref])

    def _desired_heading(self):
        """Heading the robot should face: toward the carrot on /global_path if
        available, else straight at the goal.  Used by the rotate-in-place guard."""
        rx, ry = self._x_curr[0], self._x_curr[1]
        if self.use_global_path and self._global_path is not None:
            cx, cy = self._carrot_on_path()
        else:
            cx, cy = float(self.goal[0]), float(self.goal[1])
        return math.atan2(cy - ry, cx - rx)

    # ── Solvers ───────────────────────────────────────────────────
    def _solve_carrot(self):
        """Carrot/lookahead NMPC solve (fallback + no-path/near-goal mode).
        Returns (U_sol, X_sol, S_sol), or None on total solver failure."""
        target = self._get_target()

        self._opti.set_value(self._x0_p,    self._x_curr)
        self._opti.set_value(self._xg_p,    target)
        self._opti.set_value(self._u_prev_p, self._u_prev_ctrl)
        self._opti.set_value(self._ox_p,    self._obstacles[:, 0])
        self._opti.set_value(self._oy_p,    self._obstacles[:, 1])
        self._opti.set_value(self._or_p,    self._obstacles[:, 2])

        if self.warm_start:
            if not np.any(self._X_prev != 0):
                th0 = math.atan2(target[1] - self._x_curr[1],
                                 target[0] - self._x_curr[0])
                v0  = self.v_max * 0.5
                U_ws       = np.zeros((2, self.N))
                U_ws[0, :] = v0
                X_ws       = np.zeros((3, self.N + 1))
                X_ws[:, 0] = self._x_curr
                for k in range(self.N):
                    X_ws[0, k+1] = X_ws[0, k] + v0 * math.cos(th0) * self.Ts
                    X_ws[1, k+1] = X_ws[1, k] + v0 * math.sin(th0) * self.Ts
                    X_ws[2, k+1] = th0
            else:
                U_ws = np.hstack([self._U_prev[:, 1:], self._U_prev[:, -1:]])
                X_ws = np.hstack([self._X_prev[:, 1:], self._X_prev[:, -1:]])
            S_ws = np.zeros((self.max_obs, self.N))
            self._opti.set_initial(self._U_var, U_ws)
            self._opti.set_initial(self._X_var, X_ws)
            self._opti.set_initial(self._S_var, S_ws)

        t0 = _time.perf_counter()
        try:
            sol = self._opti.solve()
            U_sol = np.array(sol.value(self._U_var))
            X_sol = np.array(sol.value(self._X_var))
            S_sol = np.array(sol.value(self._S_var))
            self._consec_fails = 0
        except Exception:
            try:
                U_sol = np.array(self._opti.debug.value(self._U_var))
                X_sol = np.array(self._opti.debug.value(self._X_var))
                S_sol = np.array(self._opti.debug.value(self._S_var))
                if not self.use_rti:
                    self._consec_fails += 1
                    if self._consec_fails <= 5:
                        self.get_logger().warn(
                            f'NMPC: partial iterate used (fail #{self._consec_fails})')
            except Exception as e2:
                self.get_logger().warn(f'Solver recovery failed entirely: {e2}')
                return None

        t_ms = (_time.perf_counter() - t0) * 1e3
        self._solve_times.append(t_ms)
        if len(self._solve_times) > 100:
            self._solve_times.pop(0)

        self._U_prev, self._X_prev, self._S_prev = U_sol, X_sol, S_sol
        return U_sol, X_sol, S_sol

    def _fit_path_poly(self):
        """Project the robot onto /global_path and fit a cubic px(σ), py(σ) over
        a forward window (σ = local arc length from the projection point).
        Returns (cx, cy, reach) with coeffs low→high order, or None if the path
        is absent / too short / near its end (carrot handles those cases)."""
        path = self._global_path
        if path is None or len(path) < 2:
            return None
        rx, ry = self._x_curr[0], self._x_curr[1]
        seg = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total < 0.5:
            return None
        # nearest point on the polyline → global arc length s0
        best_d2, s0 = float('inf'), 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((rx-x1)*dx + (ry-y1)*dy) / L2))
            px, py = x1 + t*dx, y1 + t*dy
            d2 = (rx-px)**2 + (ry-py)**2
            if d2 < best_d2:
                best_d2, s0 = d2, cum[i] + t*math.sqrt(L2)
        reach = min(self.mpcc_reach, total - s0)
        if reach < 0.5:
            return None                      # near end → carrot/terminal converges
        sig = np.linspace(0.0, reach, max(8, int(reach / 0.1)))
        abs_s = s0 + sig
        xs = np.interp(abs_s, cum, path[:, 0])
        ys = np.interp(abs_s, cum, path[:, 1])
        cx = np.polyfit(sig, xs, 3)[::-1].copy()   # low→high order
        cy = np.polyfit(sig, ys, 3)[::-1].copy()
        return cx, cy, float(reach)

    def _solve_mpcc(self, cx, cy, smax):
        """MPCC solve: set path params, warm-start, solve.  Returns
        (U_sol, X_sol, S_sol) or None on failure (caller falls back to carrot)."""
        o = self._m_opti
        o.set_value(self._m_x0,    self._x_curr)
        o.set_value(self._m_cx,    cx)
        o.set_value(self._m_cy,    cy)
        o.set_value(self._m_smax,  smax)
        o.set_value(self._m_uprev, self._u_prev_ctrl)
        o.set_value(self._m_ox,    self._obstacles[:, 0])
        o.set_value(self._m_oy,    self._obstacles[:, 1])
        o.set_value(self._m_or,    self._obstacles[:, 2])

        if self.warm_start and np.any(self._mX_prev != 0):
            o.set_initial(self._mX_var, np.hstack([self._mX_prev[:, 1:], self._mX_prev[:, -1:]]))
            o.set_initial(self._mU_var, np.hstack([self._mU_prev[:, 1:], self._mU_prev[:, -1:]]))
            Pp_ws = np.concatenate([self._mPp_prev[1:], self._mPp_prev[-1:]])
            Pp_ws = np.clip(Pp_ws - Pp_ws[0], 0.0, smax)      # re-zero local progress
            o.set_initial(self._mPp_var, Pp_ws.reshape(1, -1))
            o.set_initial(self._mVp_var,
                          np.hstack([self._mVp_prev[1:], self._mVp_prev[-1:]]).reshape(1, -1))
            o.set_initial(self._mS_var, np.hstack([self._mS_prev[:, 1:], self._mS_prev[:, -1:]]))
        else:
            v0 = 0.5 * self.v_max
            X0 = np.repeat(self._x_curr.reshape(3, 1), self.N + 1, axis=1)
            o.set_initial(self._mX_var, X0)
            o.set_initial(self._mPp_var,
                          np.linspace(0.0, min(smax, v0*self.N*self.Ts), self.N + 1).reshape(1, -1))
            o.set_initial(self._mVp_var, np.full((1, self.N), v0))
            o.set_initial(self._mS_var, np.zeros((self.max_obs, self.N)))

        t0 = _time.perf_counter()
        try:
            sol = o.solve()
            getv = sol.value
        except Exception:
            try:
                o.debug.value(self._mU_var)          # probe: does a best iterate exist?
                getv = o.debug.value
                self.get_logger().warn('MPCC: partial iterate used', throttle_duration_sec=2.0)
            except Exception:
                self.get_logger().warn('MPCC solve failed — carrot fallback',
                                       throttle_duration_sec=2.0)
                return None
        U_sol  = np.array(getv(self._mU_var)).reshape(2, self.N)
        X_sol  = np.array(getv(self._mX_var)).reshape(3, self.N + 1)
        S_sol  = np.array(getv(self._mS_var)).reshape(self.max_obs, self.N)
        Pp_sol = np.array(getv(self._mPp_var)).reshape(-1)
        Vp_sol = np.array(getv(self._mVp_var)).reshape(-1)

        t_ms = (_time.perf_counter() - t0) * 1e3
        self._solve_times.append(t_ms)
        if len(self._solve_times) > 100:
            self._solve_times.pop(0)

        self._mU_prev, self._mX_prev, self._mS_prev = U_sol, X_sol, S_sol
        self._mPp_prev, self._mVp_prev = Pp_sol, Vp_sol
        return U_sol, X_sol, S_sol

    # ── Reverse-capable terminal controller ───────────────────────
    @staticmethod
    def _wrap(a):
        """Wrap an angle to (-pi, pi]."""
        return math.atan2(math.sin(a), math.cos(a))

    def _terminal_clear(self):
        """True if the robot is in the open (no obstacle close) so the simple
        terminal controller can run.  Near obstacles we keep the full NMPC so
        obstacle avoidance/constraints stay active."""
        real = self._obstacles[self._obstacles[:, 0] < 900]
        for ox, oy, orad in real:
            if math.hypot(self._x_curr[0] - ox,
                          self._x_curr[1] - oy) < orad + self.margin + 0.4:
                return False
        return True

    def _terminal_control(self, rho):
        """Go-to-point law for the final approach.  Picks forward or reverse by
        whichever needs less turning, so a forward-only robot that overshot an
        awkwardly-approached goal simply BACKS UP instead of looping/hunting."""
        dx = self.goal[0] - self._x_curr[0]
        dy = self.goal[1] - self._x_curr[1]
        alpha = self._wrap(math.atan2(dy, dx) - self._x_curr[2])  # heading err to goal
        if abs(alpha) <= math.pi / 2:                # goal ahead -> drive forward
            direction = 1.0
        else:                                        # goal behind -> reverse in
            direction = -1.0
            alpha = self._wrap(alpha - math.pi)      # steer the rear toward the goal
        v = direction * self.term_k_rho * rho
        v = max(-self.v_rev_max, min(self.v_max, v))
        v *= max(0.0, math.cos(alpha))               # slow down while still turning
        w = max(-self.omega_max, min(self.omega_max, self.term_k_alpha * alpha))
        return v, w

    # ── Main control loop ─────────────────────────────────────────
    def _control_loop(self):
        if not self._initialized:
            return

        # ── EKF prediction using last applied command ─────────────
        self._ekf.predict(self._odom_v, self._odom_w, self.Ts)

        # ── SLAM pose fusion (Phase 2): drift-corrected map-frame state ──
        # Fuse the SLAM map→base_link pose as a GATED measurement.  This is the
        # piece that was missing before: gating rejects large single-cycle jumps
        # (bad scan matches / TF spikes) so they cannot corrupt the smooth control
        # state, while sustained offsets are eventually accepted so the estimate
        # re-syncs after a loop closure.  With fusion on, _x_curr is a stable
        # estimate in the map frame, so the NMPC drives to the *true world* goal
        # instead of a drifting odom target.  Cache map←odom for obstacle TF.
        self._map_from_odom = self._lookup_map_from_odom()
        if self.fuse_slam:
            z, ok = self._get_pose_from_tf()
            if ok:
                self._fuse_gated(z)

        # Use EKF-filtered pose for planning
        self._x_curr = self._ekf.x.copy()

        # Record the driven path in the (map) control frame for RViz
        self._record_actual_path()

        # Check goal reached
        dist_to_goal = math.hypot(
            self._x_curr[0] - self.goal[0],
            self._x_curr[1] - self.goal[1])
        self._min_dist_goal = min(self._min_dist_goal, dist_to_goal)  # closest approach

        # Success = inside the tight tolerance, OR "arrived & settled": within a
        # small band and actually stopped (|v|≈0) for settle_cycles.  The second
        # path catches the case where the speed taper + sim deadband stall the
        # robot a few cm short of goal_tol so it never crosses the tight radius.
        stopped = (abs(self._odom_v) < self.settle_speed
                   and abs(self._odom_w) < 0.1)   # not translating AND not rotating
        self._stuck_ctr = self._stuck_ctr + 1 if stopped else 0
        within_band = dist_to_goal < self.goal_settle_tol
        settled = within_band and self._stuck_ctr >= self.settle_cycles

        if dist_to_goal < self.goal_tol or settled:
            if not self._goal_reached:
                self._goal_reached = True
                how = 'exact' if dist_to_goal < self.goal_tol else 'settled'
                self.get_logger().info(
                    f'\n{"="*50}\n'
                    f'  GOAL REACHED!  ({how})\n'
                    f'  Position : ({self._x_curr[0]:.3f}, {self._x_curr[1]:.3f})\n'
                    f'  Goal     : ({self.goal[0]:.3f}, {self.goal[1]:.3f})\n'
                    f'  Final error   : {dist_to_goal:.3f} m  ({dist_to_goal*100:.1f} cm)\n'
                    f'  Closest approach: {self._min_dist_goal*100:.1f} cm\n'
                    f'{"="*50}'
                )
            self.cmd_pub.publish(Twist())   # stop
            return

        # Blocked: the robot has stopped for a long time but is NOT near the goal
        # (outside the settle band).  This almost always means the goal was placed
        # too close to an obstacle — the 0.5 m safety margin can't be crossed — so
        # it stalls a few tens of cm short.  Warn ONCE so it doesn't sit silently
        # with no GOAL REACHED (this is what happened with the (13,-3.9) goal).
        if (not self._blocked_warned and not within_band
                and self._stuck_ctr >= self.blocked_cycles):
            self._blocked_warned = True
            # Distinguish a genuinely blocked goal (hemmed in by an obstacle) from
            # a convergence failure on an OPEN goal (forward-only overshoot/hunting
            # after an awkward approach) — don't falsely blame obstacles.
            real = self._obstacles[self._obstacles[:, 0] < 900]
            near_obs = any(
                math.hypot(self.goal[0] - ox, self.goal[1] - oy) < orad + self.margin + 0.15
                for ox, oy, orad in real)
            reason = ('the goal is too close to an obstacle (0.5 m safety margin cannot '
                      'be crossed) — pick a goal ≥1 m from any cylinder' if near_obs else
                      'the goal is in open space but the approach was awkward for this '
                      'forward-only robot (it overshot and could not re-converge) — try '
                      'a goal with a clearer, more direct approach')
            self.get_logger().warn(
                f'GOAL NOT REACHED — got within {self._min_dist_goal*100:.0f} cm, then '
                f'stopped {dist_to_goal*100:.0f} cm away for '
                f'{self.blocked_cycles*self.Ts:.0f} s. Likely {reason}.')

        # ── Reverse-capable terminal controller (final parking) ──────
        # For the last stretch, hand off from NMPC/MPCC to a go-to-point law
        # that can drive OR reverse straight to the clicked goal, so an
        # awkwardly-approached goal is reached instead of hunted (scenario 3).
        # Only engages in the open — near obstacles the NMPC keeps handling
        # avoidance/constraints.  Runs after the goal-reached check above, so it
        # only acts in the band  goal_tol < dist < terminal_radius.
        if (self.use_terminal_ctrl and dist_to_goal < self.terminal_radius
                and self._terminal_clear()):
            v, w = self._terminal_control(dist_to_goal)
            tw = Twist()
            tw.linear.x  = float(v)
            tw.angular.z = float(w)
            self.cmd_pub.publish(tw)
            self._u_prev_ctrl = np.array([v, w])
            return

        # ── Rotate in place toward the target when badly mis-aligned ──
        # The robot drives forward only (v_min≥0), so it cannot tightly reach a
        # goal that is beside/behind it — it arcs into a loop.  When the heading
        # error to the path/goal is large, turn in place first (v=0), then drive.
        # Hysteresis (0.4×) prevents chattering; turning in place is collision-safe
        # (the ~circular footprint centre does not move).
        if self.rotate_in_place:
            dh   = self._desired_heading()
            herr = math.atan2(math.sin(dh - self._x_curr[2]),
                              math.cos(dh - self._x_curr[2]))
            thr  = self.rotate_thresh * (0.4 if self._rotating else 1.0)
            if abs(herr) > thr:
                self._rotating = True
                w = max(-self.omega_max, min(self.omega_max, self.rotate_gain * herr))
                tw = Twist()
                tw.angular.z = float(w)
                self.cmd_pub.publish(tw)
                self._u_prev_ctrl = np.array([0.0, w])
                # cold-start the NMPC once aligned (fresh trajectory from new heading)
                self._X_prev  = np.zeros((3, self.N + 1))
                self._mX_prev = np.zeros((3, self.N + 1))
                return
            self._rotating = False

        # ── Select closest obstacles for this step ────────────────
        self._obstacles = self._select_closest_obstacles()

        # ── Solve: MPCC path-following first, carrot NMPC as fallback ─
        # MPCC tracks the A* path as a contour (arc-length progress + contour/
        # lag error).  When there is no usable path (startup, near goal, or MPCC
        # solver failure) we fall back to the carrot NMPC, which handles pure
        # goal-directed control + reactive obstacle avoidance.
        res = None
        if self.use_mpcc:
            poly = self._fit_path_poly()
            if poly is not None:
                res = self._solve_mpcc(*poly)
        if res is None:
            res = self._solve_carrot()
        if res is None:
            self.cmd_pub.publish(Twist())   # total solver failure → stop safely
            return
        U_sol, X_sol, S_sol = res

        # Slack is a SQUARED-distance give on the (r+margin) constraint; small
        # values are the soft margin flexing near an obstacle, not contact.
        # margin has a 0.20 m buffer, so warn only when the give is large enough
        # to actually eat into real clearance (~0.25 m² ≈ >10 cm into the buffer).
        max_slack = float(np.max(S_sol))
        if max_slack > 0.25:
            self.get_logger().warn(
                f'Obstacle margin eaten: max_slack={max_slack:.3f} m² — clearance low')

        # Taper linear speed only within 1 m of goal so that the tapering
        # does not interfere with obstacle-corridor navigation at x=10-11
        # where dist_to_goal ≈ 2 m (the old 2 m threshold caused stuttering
        # by reducing speed right in the middle of the dense obstacle field).
        approach_scale = min(1.0, dist_to_goal / 1.0)
        v_raw  = float(U_sol[0, 0]) * approach_scale
        v_cmd     = float(np.clip(v_raw,         self.v_min, self.v_max))
        omega_cmd = float(np.clip(U_sol[1, 0], -self.omega_max, self.omega_max))

        # ── EMA smoother on v only — ω passes through unfiltered ──────
        # Smoothing v suppresses start-stop chattering from NMPC barrier
        # gradient oscillations (the original stuttering complaint).
        # ω must NOT be filtered: any lag on the steering command delays
        # the avoidance turn by α·Ts·v_max metres, causing the robot to
        # clip obstacles before the turn takes effect.
        self._v_smooth = self._v_alpha * v_cmd + (1.0 - self._v_alpha) * self._v_smooth
        v_pub   = float(np.clip(self._v_smooth, self.v_min, self.v_max))
        om_pub  = float(np.clip(omega_cmd,     -self.omega_max, self.omega_max))

        # ── Publish /cmd_vel ──────────────────────────────────────
        cmd             = Twist()
        cmd.linear.x    = v_pub
        cmd.angular.z   = om_pub
        self.cmd_pub.publish(cmd)

        # Save filtered command for EKF prediction and Δu penalty
        self._last_v       = v_pub
        self._last_omega   = om_pub
        self._u_prev_ctrl  = np.array([v_pub, om_pub])

        # ── Publish predicted path (RViz) ─────────────────────────
        pred_path = Path()
        pred_path.header.stamp    = self.get_clock().now().to_msg()
        pred_path.header.frame_id = self._active_frame
        for k in range(self.N + 1):
            ps               = PoseStamped()
            ps.header        = pred_path.header
            ps.pose.position.x = float(X_sol[0, k])
            ps.pose.position.y = float(X_sol[1, k])
            pred_path.poses.append(ps)
        self.pred_path_pub.publish(pred_path)

        # ── Publish actual path (every call) ──────────────────────
        self._actual_path.header.stamp = self.get_clock().now().to_msg()
        self.actual_path_pub.publish(self._actual_path)

    # ── Diagnostics ───────────────────────────────────────────────
    def _diag_cb(self):
        active_obs = np.sum(self._all_obstacles[:, 0] < 900)
        dist = math.hypot(self._x_curr[0]-self.goal[0],
                          self._x_curr[1]-self.goal[1])
        self._min_dist_goal = min(self._min_dist_goal, dist)   # capture settled error too
        ctrl = 'MPCC' if (self.use_mpcc and self._global_path is not None) else 'carrot'
        acc_str = (f'| ACCURACY: settled {dist*100:.1f} cm, best {self._min_dist_goal*100:.1f} cm'
                   if self._goal_reached else '')

        # Solve-time statistics
        if self._solve_times:
            t_mean = np.mean(self._solve_times)
            t_max  = np.max(self._solve_times)
            t_budget = self._solve_budget
            timing_str = (f'| {ctrl} solve: mean={t_mean:.1f}ms '
                          f'max={t_max:.1f}ms '
                          f'budget={t_budget:.0f}ms '
                          f'({100*t_mean/t_budget:.0f}% used)')
        else:
            timing_str = ''

        self.get_logger().info(
            f'pos=({self._x_curr[0]:.2f},{self._x_curr[1]:.2f}) '
            f'goal=({self.goal[0]:.1f},{self.goal[1]:.1f}) '
            f'dist={dist:.2f}m | obs_active={active_obs} {timing_str} {acc_str}'
        )

    # ── Goal marker ───────────────────────────────────────────────
    def _publish_goal_marker_once(self):
        if not self._goal_marker_published:
            self._publish_goal_marker(self.goal[0], self.goal[1])
            self._goal_marker_published = True

    def _publish_goal_marker(self, gx, gy):
        m = Marker()
        m.header.frame_id = self._active_frame
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns              = 'goal'
        m.id              = 0
        m.type            = Marker.CYLINDER
        m.action          = Marker.ADD
        m.pose.position.x = gx
        m.pose.position.y = gy
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 0.1
        m.color.r = 0.0
        m.color.g = 0.9
        m.color.b = 0.2
        m.color.a = 0.85
        m.lifetime.sec = 0
        self.goal_marker_pub.publish(m)

        # Text label
        t = Marker()
        t.header = m.header
        t.ns     = 'goal'
        t.id     = 1
        t.type   = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose.position.x = gx
        t.pose.position.y = gy
        t.pose.position.z = 0.6
        t.pose.orientation.w = 1.0
        t.scale.z  = 0.4
        t.color.r  = 1.0
        t.color.g  = 1.0
        t.color.b  = 1.0
        t.color.a  = 1.0
        t.text     = 'GOAL'
        t.lifetime.sec = 0
        self.goal_marker_pub.publish(t)


def main(args=None):
    rclpy.init(args=args)
    node = NMPCController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
