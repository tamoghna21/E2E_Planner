"""Pure-pursuit + PID tracking-controller "expert" (Phase 4.5 conditional fix).

Replaces the open-loop kinematic-inversion pseudo-action expert (src/data/collect_expert.py's
original approach), which Phase 4.5's decisive test showed cannot reliably reproduce the logged
route through real physics (49.9% mean route completion on its own training scenarios, 3/6 ending
in collision).

This controller is *closed-loop*: at every step it looks at the vehicle's actual current state
(not the logged state) and computes the action that steers it back toward the logged reference
path -- pure pursuit for steering, a PID on a target speed profile for throttle/brake. Because it
reacts to whatever state the car is actually in, it (a) produces labels that are physically
consistent by construction (the action recorded is the action that was actually applied through
EnvInputPolicy/real physics, not inverted after the fact), and (b) can be queried at any
off-trajectory state, which is what makes DAgger (Phase 6a) meaningful later.
"""
import numpy as np


class ReferencePath:
    """Arc-length-parameterized logged trajectory: position, heading, speed vs. progress."""

    def __init__(self, position, heading, speed, valid):
        valid = np.asarray(valid).astype(bool)
        self.position = np.asarray(position)[valid]
        self.heading = np.asarray(heading)[valid]
        self.speed = np.asarray(speed)[valid]

        deltas = np.diff(self.position, axis=0)
        seg_len = np.linalg.norm(deltas, axis=1)
        self.cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.length = self.cum_len[-1]

    def nearest_index(self, point):
        d = np.linalg.norm(self.position - np.asarray(point)[None, :], axis=1)
        return int(np.argmin(d))

    def progress_at(self, point):
        return self.cum_len[self.nearest_index(point)]

    def point_at_progress(self, s):
        s = np.clip(s, 0.0, self.length)
        idx = np.searchsorted(self.cum_len, s)
        idx = min(idx, len(self.position) - 1)
        return self.position[idx]

    def speed_at_progress(self, s):
        s = np.clip(s, 0.0, self.length)
        idx = np.searchsorted(self.cum_len, s)
        idx = min(idx, len(self.speed) - 1)
        return self.speed[idx]


class PurePursuitPIDController:
    def __init__(self, ref_path: ReferencePath, wheelbase, max_steering_rad,
                 lookahead_min=5.0, lookahead_gain=0.5,
                 kp=0.6, ki=0.05, kd=0.05, dt=0.1):
        self.ref = ref_path
        self.wheelbase = wheelbase
        self.max_steering_rad = max_steering_rad
        self.lookahead_min = lookahead_min
        self.lookahead_gain = lookahead_gain
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def act(self, position, heading_theta, speed):
        progress = self.ref.progress_at(position)
        lookahead = self.lookahead_min + self.lookahead_gain * speed
        target = self.ref.point_at_progress(progress + lookahead)

        dx_world, dy_world = target[0] - position[0], target[1] - position[1]
        cos_h, sin_h = np.cos(-heading_theta), np.sin(-heading_theta)
        dx_local = dx_world * cos_h - dy_world * sin_h
        dy_local = dx_world * sin_h + dy_world * cos_h
        ld = max(np.hypot(dx_local, dy_local), 1e-3)

        curvature = 2.0 * dy_local / (ld ** 2)
        steering = np.clip(np.arctan(self.wheelbase * curvature) / self.max_steering_rad, -1.0, 1.0)

        target_speed = self.ref.speed_at_progress(progress + self.lookahead_min)
        error = target_speed - speed
        self._integral = np.clip(self._integral + error * self.dt, -10.0, 10.0)
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error
        throttle_brake = np.clip(self.kp * error + self.ki * self._integral + self.kd * derivative, -1.0, 1.0)

        return np.array([float(steering), float(throttle_brake)], dtype=np.float32)


def build_controller_for_scenario(env):
    """Construct a PurePursuitPIDController from the currently-loaded scenario's logged SDC track."""
    scenario = env.engine.data_manager.current_scenario
    sdc_id = str(scenario["metadata"]["sdc_id"])
    track = scenario["tracks"][sdc_id]
    state = track["state"]
    position = np.asarray(state["position"])[:, :2]
    heading = np.asarray(state["heading"])
    velocity = np.asarray(state["velocity"])
    speed = np.linalg.norm(velocity, axis=1)
    valid = np.asarray(state["valid"])

    ref_path = ReferencePath(position, heading, speed, valid)
    wheelbase = env.agent.FRONT_WHEELBASE + env.agent.REAR_WHEELBASE
    max_steering_rad = np.deg2rad(env.agent.max_steering)
    return PurePursuitPIDController(ref_path, wheelbase, max_steering_rad)
