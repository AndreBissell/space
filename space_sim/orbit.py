import numpy as np
from config import G


class Orbit:
    """
    Computes circular-orbit values around a body whose mass scales as radius^3
    (same-density assumption used throughout the sim: mass = G * radius^3).

    Usage:
        earth_orbit = Orbit(PLANET_RADIUS)
        v = earth_orbit.speed(r)          # circular orbital speed at distance r
        T = earth_orbit.period(r)         # orbital period in frames
        v_esc = earth_orbit.escape_speed(r)
        r = earth_orbit.radius_for_speed(v)  # radius that gives circular speed v
    """

    def __init__(self, body_radius):
        self.gm = G * float(body_radius) ** 3

    def speed(self, r):
        """Circular orbital speed at distance r."""
        return float(np.sqrt(self.gm / r))

    def period(self, r):
        """Orbital period in simulation frames at distance r."""
        return float(2.0 * np.pi * r / self.speed(r))

    def escape_speed(self, r):
        """Minimum speed needed to escape from distance r."""
        return float(np.sqrt(2.0 * self.gm / r))

    def radius_for_speed(self, v):
        """Orbital radius at which circular speed equals v."""
        return float(self.gm / v ** 2)

    def summary(self, r):
        """Return a dict of all key values at distance r."""
        return {
            "r":            r,
            "speed":        self.speed(r),
            "period_frames": self.period(r),
            "escape_speed": self.escape_speed(r),
        }
