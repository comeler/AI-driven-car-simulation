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
from track import build_track_from_config, LapProgress


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

        # Start at the first centerline point, facing along the track -
        # works for any track shape and either direction (clockwise flag).
        self._start_x, self._start_y = self.track.centerline[0]
        next_x, next_y = self.track.centerline[1]
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
        self.progress = LapProgress(self.track)
        self.progress.reset(self.car.x, self.car.y)
        self.step_count = 0
        return self._get_state()

    def step(self, action):
        """Apply one action for one frame. Returns (state, reward, done, info)."""
        accelerate, brake, turn_left, turn_right = ACTIONS[action]
        self.car.update(self.dt, accelerate, brake, turn_left, turn_right)
        self.step_count += 1

        # delta = signed forward distance traveled along the track this
        # frame (continuous projection onto the nearest point of the
        # centerline curve - NOT distance to any fixed checkpoint, so lateral
        # position / which line through a corner the car takes is free).
        delta, lap_completed = self.progress.update(self.car.x, self.car.y)
        on_track = self.track.is_on_track(self.car.x, self.car.y)

        # --- Reward shaping (constants from config.py) ---
        reward = 0.0
        reward += delta * cfg.REWARD_PROGRESS_SCALE
        reward += cfg.REWARD_PER_FRAME_PENALTY
        if lap_completed:
            reward += cfg.REWARD_LAP_BONUS

        done = False
        if not on_track:
            reward += cfg.REWARD_OFF_TRACK_PENALTY
            done = True
        if self.step_count >= self.max_steps:
            done = True

        info = {
            "on_track": on_track,
            "laps_completed": self.progress.laps_completed,
            "lap_completed": lap_completed,
        }
        return self._get_state(), reward, done, info

    def _get_state(self):
        """Build the normalized state vector the network will see:
        [sensor distances (0-1), speed (-1 to 1), heading misalignment with
        the track's local direction of travel (-1 to 1)]"""
        sensors = self.track.get_sensor_readings(
            self.car.x, self.car.y, self.car.angle,
            self.sensor_angles_deg, self.sensor_range,
        )
        normalized_sensors = [d / self.sensor_range for d in sensors]

        normalized_speed = self.car.speed / self.car.max_speed

        angle_diff = (self.progress.track_heading - self.car.angle + math.pi) % (2 * math.pi) - math.pi
        normalized_angle_diff = angle_diff / math.pi

        return normalized_sensors + [normalized_speed, normalized_angle_diff]

    @property
    def state_size(self):
        return len(self.sensor_angles_deg) + 2

    @property
    def action_size(self):
        return len(ACTIONS)
