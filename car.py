"""
Car class: represents the car's physical state (position, speed, direction)
and how it responds to control inputs each frame. Physics constants are
passed in as keyword arguments (with sensible fallback defaults) - CarEnv
and Simulation both pull the real values from config.py and pass them in,
so tuning car handling only ever means editing config.py.
"""

import math


class Car:
    def __init__(self, x, y, angle=0.0, *,
                 acceleration=200.0, brake_strength=300.0, friction=80.0,
                 max_speed=300.0, max_reverse_speed=-100.0, turn_rate=2.5,
                 length=40, width=20):
        # Position (pixels)
        self.x = x
        self.y = y

        # Orientation (radians). 0 = facing right (+x axis), increases clockwise
        # on screen (because pygame's y axis points down).
        self.angle = angle

        # Current speed (pixels per second). Can be negative (reverse).
        self.speed = 0.0

        # --- Tunable physics parameters (normally supplied from config.py) ---
        self.acceleration = acceleration        # px/s^2 while accelerating
        self.brake_strength = brake_strength     # px/s^2 while braking
        self.friction = friction                 # px/s^2 natural slowdown when coasting
        self.max_speed = max_speed
        self.max_reverse_speed = max_reverse_speed
        self.turn_rate = turn_rate               # radians/s at full steering, scaled by speed

        # --- Shape (used for drawing, and later for collision detection) ---
        self.length = length
        self.width = width

        # History of positions, used to draw the trajectory trail
        self.trajectory = []
        self.max_trajectory_points = 2000

    def update(self, dt, accelerate=False, brake=False, turn_left=False, turn_right=False):
        """Advance the car's state by dt seconds, given control inputs."""

        # 1. Update speed based on accelerate / brake / coasting
        if accelerate:
            self.speed += self.acceleration * dt
        elif brake:
            if self.speed > 0:
                self.speed -= self.brake_strength * dt
            else:
                self.speed -= self.acceleration * dt  # lets you reverse by holding brake at standstill
        else:
            # Natural friction pulls speed back toward 0
            if self.speed > 0:
                self.speed = max(0.0, self.speed - self.friction * dt)
            elif self.speed < 0:
                self.speed = min(0.0, self.speed + self.friction * dt)

        self.speed = max(self.max_reverse_speed, min(self.max_speed, self.speed))

        # 2. Update steering angle. Turn rate is scaled by current speed so the
        # car can't spin in place while stationary, like a real vehicle.
        speed_factor = self.speed / self.max_speed
        if turn_left:
            if speed_factor == 0 :
                self.angle = 0
            elif speed_factor < 0.2 :
                self.angle -= self.turn_rate *speed_factor * dt
            else :
                self.angle -= (self.turn_rate *speed_factor * dt)*(-speed_factor+1/0.8)
        if turn_right:
            if speed_factor == 0 :
                self.angle = 0
            elif speed_factor < 0.2 :
                self.angle += self.turn_rate *speed_factor * dt
            else :
                self.angle += (self.turn_rate *speed_factor * dt)*(-speed_factor+1/0.8)


        # 3. Update position from speed + angle
        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt

        # 4. Record trajectory for drawing
        self.trajectory.append((self.x, self.y))
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)

    def get_corners(self):
        """Return the 4 corners of the car's rectangle in world space, so it
        can be drawn rotated to match self.angle."""
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        half_len, half_wid = self.length / 2, self.width / 2

        local_corners = [
            (half_len, -half_wid),
            (half_len, half_wid),
            (-half_len, half_wid),
            (-half_len, -half_wid),
        ]
        return [
            (self.x + lx * cos_a - ly * sin_a, self.y + lx * sin_a + ly * cos_a)
            for lx, ly in local_corners
        ]
