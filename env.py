"""
CarEnv: wraps Car + Track behind a reset()/step(action) interface, the
standard shape used by reinforcement learning code (matches gymnasium's
convention, without depending on the library). This is what the neural
network actually talks to - it never touches Car or Track directly.

Every tunable value (track shape, car physics, sensors, reward shaping,
episode length) comes from config.py - edit that file, not this one.
"""

import math

import config as cfg
from car import Car
from track import build_track_from_config, ProgressTracker


# Each entry: (accelerate, brake, turn_left, turn_right)
ACTIONS = [
    (True, False, False, False),   # 0: accelerate straight
    (True, False, True, False),    # 1: accelerate + turn left
    (True, False, False, True),    # 2: accelerate + turn right
    (False, True, False, False),   # 3: brake
    (False, False, False, False),  # 4: coast
]
ACTION_NAMES = ["straight", "left", "right", "brake", "coast"]


class CarEnv:
    def __init__(self):
        self.width = cfg.SCREEN_WIDTH
        self.height = cfg.SCREEN_HEIGHT
        self.dt = cfg.ENV_DT
        self.max_steps = cfg.ENV_MAX_STEPS

        self.track = build_track_from_config(cfg)

        # Start on the track's first checkpoint, facing toward the second -
        # works for any track shape, since checkpoints are evenly spaced
        # samples of the original centerline.
        self._start_x, self._start_y = self.track.checkpoints[0]
        next_x, next_y = self.track.checkpoints[1]
        self._start_angle = math.atan2(next_y - self._start_y, next_x - self._start_x)

        self.sensor_angles_deg = cfg.SENSOR_ANGLES_DEG
        self.sensor_range = cfg.SENSOR_RANGE

        self.car = None
        self.progress = None
        self.step_count = 0

    def reset(self):
        """Start a new episode: put the car back at the start line, reset
        progress tracking, and return the initial state."""
        self.car = Car(
            x=self._start_x, y=self._start_y, angle=self._start_angle,
            acceleration=cfg.CAR_ACCELERATION, brake_strength=cfg.CAR_BRAKE_STRENGTH,
            friction=cfg.CAR_FRICTION, max_speed=cfg.CAR_MAX_SPEED,
            max_reverse_speed=cfg.CAR_MAX_REVERSE_SPEED, turn_rate=cfg.CAR_TURN_RATE,
            length=cfg.CAR_LENGTH, width=cfg.CAR_WIDTH,
        )
        self.progress = ProgressTracker(self.track, checkpoint_radius=cfg.TRACK_CHECKPOINT_RADIUS)
        self.step_count = 0
        return self._get_state()

    def step(self, action):
        """Apply one action for one frame. Returns (state, reward, done, info)."""
        accelerate, brake, turn_left, turn_right = ACTIONS[action]

        # Distance to the current target checkpoint BEFORE moving, so we can
        # reward the car for closing that distance this frame.
        target = self.track.checkpoints[self.progress.next_checkpoint]
        dist_before = math.hypot(self.car.x - target[0], self.car.y - target[1])

        self.car.update(self.dt, accelerate, brake, turn_left, turn_right)
        self.step_count += 1

        dist_after = math.hypot(self.car.x - target[0], self.car.y - target[1])
        on_track = self.track.is_on_track(self.car.x, self.car.y)

        prev_checkpoint = self.progress.next_checkpoint
        self.progress.update(self.car.x, self.car.y)
        checkpoint_hit = self.progress.next_checkpoint != prev_checkpoint

        # --- Reward shaping (constants from config.py) ---
        reward = 0.0
        reward += (dist_before - dist_after) * cfg.REWARD_PROGRESS_SCALE
        reward += cfg.REWARD_PER_FRAME_PENALTY
        if checkpoint_hit:
            reward += cfg.REWARD_CHECKPOINT_BONUS

        done = False
        if not on_track:
            reward += cfg.REWARD_OFF_TRACK_PENALTY
            done = True
        if self.step_count >= self.max_steps:
            done = True

        info = {
            "on_track": on_track,
            "laps_completed": self.progress.laps_completed,
            "checkpoint_hit": checkpoint_hit,
        }
        return self._get_state(), reward, done, info

    def _get_state(self):
        """Build the normalized state vector the network will see:
        [sensor distances (0-1), speed (-1 to 1), angle to next checkpoint (-1 to 1)]"""
        sensors = self.track.get_sensor_readings(
            self.car.x, self.car.y, self.car.angle,
            self.sensor_angles_deg, self.sensor_range,
        )
        normalized_sensors = [d / self.sensor_range for d in sensors]

        normalized_speed = self.car.speed / self.car.max_speed

        target = self.track.checkpoints[self.progress.next_checkpoint]
        target_angle = math.atan2(target[1] - self.car.y, target[0] - self.car.x)
        angle_diff = (target_angle - self.car.angle + math.pi) % (2 * math.pi) - math.pi
        normalized_angle_diff = angle_diff / math.pi

        return normalized_sensors + [normalized_speed, normalized_angle_diff]

    @property
    def state_size(self):
        return len(self.sensor_angles_deg) + 2

    @property
    def action_size(self):
        return len(ACTIONS)
