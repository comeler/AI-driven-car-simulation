"""
watch_best.py: load a trained genetic algorithm's best weights and watch it
drive, rendered with pygame. Reuses CarEnv for all physics/track/progress
logic (so what you see matches exactly what the network was trained/scored
on), with a lightweight version of the Step 1-2 drawing code on top.
Network shape (HIDDEN_SIZE) and weights path come from config.py, same as
train.py - always in sync, no manual matching needed.
"""

import math

import numpy as np
import pygame

import config as cfg
from env import CarEnv
from neural_network import NeuralNetwork


def draw(screen, env, sensor_readings, on_track):
    screen.fill((30, 30, 30))

    if len(env.track.inner) > 1:
        pygame.draw.lines(screen, (200, 200, 200), True, env.track.inner, 2)
    if len(env.track.outer) > 1:
        pygame.draw.lines(screen, (200, 200, 200), True, env.track.outer, 2)

    for cx, cy in env.track.checkpoints:
        pygame.draw.circle(screen, (60, 140, 60), (int(cx), int(cy)), 4)
    next_cp = env.track.checkpoints[env.progress.next_checkpoint]
    pygame.draw.circle(screen, (255, 255, 0), (int(next_cp[0]), int(next_cp[1])), 7, 2)

    if len(env.car.trajectory) > 1:
        pygame.draw.lines(screen, (90, 90, 90), False, env.car.trajectory, 2)

    for offset_deg, dist in zip(env.sensor_angles_deg, sensor_readings):
        angle = env.car.angle + math.radians(offset_deg)
        end_x = env.car.x + math.cos(angle) * dist
        end_y = env.car.y + math.sin(angle) * dist
        pygame.draw.line(screen, (255, 210, 0), (env.car.x, env.car.y), (end_x, end_y), 1)

    color = (0, 200, 255) if on_track else (220, 60, 60)
    pygame.draw.polygon(screen, color, env.car.get_corners())

    font = pygame.font.SysFont("consolas", 18)
    lines = [
        f"Speed: {env.car.speed:5.1f} px/s",
        f"On track: {on_track}",
        f"Lap progress: {env.progress.progress_fraction * 100:5.1f}%",
        f"Laps completed: {env.progress.laps_completed}",
    ]
    for i, text in enumerate(lines):
        surf = font.render(text, True, (255, 255, 255))
        screen.blit(surf, (10, 10 + i * 20))

    pygame.display.flip()


def main():
    weights = np.load(cfg.BEST_WEIGHTS_PATH)
    env = CarEnv()
    net = NeuralNetwork(env.state_size, cfg.HIDDEN_SIZE, env.action_size, weights=weights)

    pygame.init()
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Best trained car")
    clock = pygame.time.Clock()

    state = env.reset()
    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = net.choose_action(state)
        state, reward, done, info = env.step(action)

        sensor_readings = env.track.get_sensor_readings(
            env.car.x, env.car.y, env.car.angle, env.sensor_angles_deg, env.sensor_range,
        )
        draw(screen, env, sensor_readings, info["on_track"])

        if done:
            state = env.reset()

    pygame.quit()


if __name__ == "__main__":
    main()
