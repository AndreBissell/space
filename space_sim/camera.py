import numpy as np
import pygame
from config import PLANET_RADIUS, G

TURN_SPEED    = np.radians(1.0)
THRUST        = 0.5
DRAG          = 0.98
MAX_SPEED     = 25.0
INITIAL_SPEED = 2.5

INITIAL_POS   = np.array([0.0, 0.0, -(PLANET_RADIUS + 250.0)], dtype=np.float64)
INITIAL_YAW   = np.pi
INITIAL_PITCH = 0.0


def _initial_velocity():
    cy, sy = np.cos(INITIAL_YAW), np.sin(INITIAL_YAW)
    cp, sp = np.cos(INITIAL_PITCH), np.sin(INITIAL_PITCH)
    forward = np.array([sy * cp, sp, cy * cp], dtype=np.float64)
    return forward * INITIAL_SPEED


class Camera:
    def __init__(self):
        self.pos      = INITIAL_POS.copy()
        self.yaw      = INITIAL_YAW
        self.pitch    = INITIAL_PITCH
        self.velocity = _initial_velocity()
        self.speed    = INITIAL_SPEED

    def reset(self):
        self.pos      = INITIAL_POS.copy()
        self.yaw      = INITIAL_YAW
        self.pitch    = INITIAL_PITCH
        self.velocity = _initial_velocity()
        self.speed    = INITIAL_SPEED

    def update(self, keys, gravity_sources):
        # --- Rotation ---
        if keys[pygame.K_LEFT]:
            self.yaw -= TURN_SPEED
        if keys[pygame.K_RIGHT]:
            self.yaw += TURN_SPEED
        if keys[pygame.K_UP]:
            self.pitch += TURN_SPEED
        if keys[pygame.K_DOWN]:
            self.pitch -= TURN_SPEED
        self.pitch = float((self.pitch + np.pi) % (2 * np.pi) - np.pi)

        # --- Gravity from all massive bodies ---
        for center, mass in gravity_sources:
            diff    = center - self.pos
            dist_sq = float(np.dot(diff, diff))
            if dist_sq > 1.0:
                dist = np.sqrt(dist_sq)
                self.velocity += (diff / dist) * (G * mass / dist_sq)

        # --- Thrust ---
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        forward = np.array([sy * cp, sp, cy * cp], dtype=np.float64)
        if keys[pygame.K_SPACE]:
            self.velocity += forward * THRUST

        # --- Drag & speed cap ---
        self.velocity *= DRAG
        self.speed = float(np.linalg.norm(self.velocity))
        if self.speed > MAX_SPEED:
            self.velocity *= MAX_SPEED / self.speed
            self.speed = MAX_SPEED

        # --- Move ---
        self.pos += self.velocity
