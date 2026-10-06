"""
Simulation: owns the pygame window, the game loop, input handling and drawing.
Keeping this separate from Car and Track keeps "physics", "track geometry" and
"rendering/game loop" decoupled - useful later when we swap manual keyboard
input for an AI agent, or swap this track for another one.
"""

import math
import pygame

import config as cfg
from car import Car
from track import build_track_from_config, ProgressTracker


class Simulation:
    def __init__(self):
        pygame.init()
        self.width = cfg.SCREEN_WIDTH
        self.height = cfg.SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("2D Autonomous Car - Manual Control")
        self.clock = pygame.time.Clock()

        self.track = build_track_from_config(cfg)
        self.progress = ProgressTracker(self.track, checkpoint_radius=cfg.TRACK_CHECKPOINT_RADIUS)

        # Start on the track's first checkpoint, facing toward the second.
        start_x, start_y = self.track.checkpoints[0]
        next_x, next_y = self.track.checkpoints[1]
        start_angle = math.atan2(next_y - start_y, next_x - start_x)
        self.car = Car(
            x=start_x, y=start_y, angle=start_angle,
            acceleration=cfg.CAR_ACCELERATION, brake_strength=cfg.CAR_BRAKE_STRENGTH,
            friction=cfg.CAR_FRICTION, max_speed=cfg.CAR_MAX_SPEED,
            max_reverse_speed=cfg.CAR_MAX_REVERSE_SPEED, turn_rate=cfg.CAR_TURN_RATE,
            length=cfg.CAR_LENGTH, width=cfg.CAR_WIDTH,
        )

        self.sensor_angles_deg = cfg.SENSOR_ANGLES_DEG
        self.sensor_range = cfg.SENSOR_RANGE

        self.running = True

        # Colors
        self.bg_color = (30, 30, 30)
        self.car_color_on_track = (0, 200, 255)
        self.car_color_off_track = (220, 60, 60)
        self.trajectory_color = (90, 90, 90)
        self.wall_color = (200, 200, 200)
        self.checkpoint_color = (60, 140, 60)
        self.sensor_color = (255, 210, 0)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        accelerate = keys[pygame.K_UP] or keys[pygame.K_w]
        brake = keys[pygame.K_DOWN] or keys[pygame.K_s]
        turn_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        turn_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        return accelerate, brake, turn_left, turn_right

    def draw_boundary(self, points, color):
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, True, points, 2)

    def draw_sensors(self, readings):
        for offset_deg, dist in zip(self.sensor_angles_deg, readings):
            angle = self.car.angle + math.radians(offset_deg)
            end_x = self.car.x + math.cos(angle) * dist
            end_y = self.car.y + math.sin(angle) * dist
            pygame.draw.line(self.screen, self.sensor_color, (self.car.x, self.car.y), (end_x, end_y), 1)
            pygame.draw.circle(self.screen, self.sensor_color, (int(end_x), int(end_y)), 3)

    def draw(self, sensor_readings, on_track):
        self.screen.fill(self.bg_color)

        self.draw_boundary(self.track.inner, self.wall_color)
        self.draw_boundary(self.track.outer, self.wall_color)

        for cx, cy in self.track.checkpoints:
            pygame.draw.circle(self.screen, self.checkpoint_color, (int(cx), int(cy)), 4)
        # Highlight the checkpoint the car is currently heading toward
        next_cp = self.track.checkpoints[self.progress.next_checkpoint]
        pygame.draw.circle(self.screen, (255, 255, 0), (int(next_cp[0]), int(next_cp[1])), 7, 2)

        if len(self.car.trajectory) > 1:
            pygame.draw.lines(self.screen, self.trajectory_color, False, self.car.trajectory, 2)

        self.draw_sensors(sensor_readings)

        car_color = self.car_color_on_track if on_track else self.car_color_off_track
        pygame.draw.polygon(self.screen, car_color, self.car.get_corners())

        font = pygame.font.SysFont("consolas", 18)
        lines = [
            f"Speed: {self.car.speed:5.1f} px/s",
            f"On track: {on_track}",
            f"Lap progress: {self.progress.progress_fraction * 100:5.1f}%",
            f"Laps completed: {self.progress.laps_completed}",
            f"Sensors: " + ", ".join(f"{d:.0f}" for d in sensor_readings),
        ]
        for i, text in enumerate(lines):
            surf = font.render(text, True, (255, 255, 255))
            self.screen.blit(surf, (10, 10 + i * 20))

        pygame.display.flip()

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # seconds since last frame, capped at 60 FPS

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            accelerate, brake, turn_left, turn_right = self.handle_input()
            self.car.update(dt, accelerate, brake, turn_left, turn_right)

            on_track = self.track.is_on_track(self.car.x, self.car.y)
            self.progress.update(self.car.x, self.car.y)
            sensor_readings = self.track.get_sensor_readings(
                self.car.x, self.car.y, self.car.angle,
                self.sensor_angles_deg, self.sensor_range,
            )

            self.draw(sensor_readings, on_track)

        pygame.quit()
