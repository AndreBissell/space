import numpy as np
from config import COL_EXPLO_HOT, COL_EXPLO_FIRE, COL_EXPLO_EMBER, HIGH_SPEED_CRASH

_PALETTE = np.array([COL_EXPLO_HOT, COL_EXPLO_FIRE, COL_EXPLO_EMBER], dtype=np.float32)


class Explosion:
    """
    Particle burst spawned on collision.
    Particle count scales with impact speed; hot (yellow) particles fade to red ember.
    """

    def __init__(self, pos, impact_speed):
        big = impact_speed >= HIGH_SPEED_CRASH
        n   = int(np.clip(impact_speed * 45, 80, 800)) if big else int(np.clip(impact_speed * 20, 40, 300))

        self.positions  = np.tile(pos.astype(np.float32), (n, 1))

        # Radial velocities — faster particles for bigger crashes
        cos_t  = np.random.uniform(-1.0, 1.0, n)
        phi    = np.random.uniform(0, 2 * np.pi, n)
        sin_t  = np.sqrt(1.0 - cos_t ** 2)
        speed  = np.random.uniform(0.4, 1.8, n) * impact_speed * 0.35

        self.velocities = np.column_stack([
            sin_t * np.cos(phi) * speed,
            cos_t * speed,
            sin_t * np.sin(phi) * speed,
        ]).astype(np.float32)

        self.max_life  = np.random.uniform(45, 110, n).astype(np.float32)
        self.lifetimes = self.max_life.copy()

        # Hotter colours more likely near impact; cool embers drift further
        weights = np.array([0.45, 0.35, 0.20]) if big else np.array([0.25, 0.45, 0.30])
        choice  = np.random.choice(3, size=n, p=weights)
        self.colors = _PALETTE[choice].astype(np.float32)
        self.sizes  = np.random.uniform(1.5, 4.5, n).astype(np.float32)
        self.active = True

    def update(self):
        if not self.active:
            return
        self.positions  += self.velocities
        self.velocities *= 0.92          # atmospheric drag on debris
        self.lifetimes  -= 1.0
        alive = self.lifetimes > 0
        if not np.any(alive):
            self.active = False
            return
        self.positions  = self.positions[alive]
        self.velocities = self.velocities[alive]
        self.lifetimes  = self.lifetimes[alive]
        self.max_life   = self.max_life[alive]
        self.colors     = self.colors[alive]
        self.sizes      = self.sizes[alive]

    def get_render_data(self):
        """
        Returns (positions, prescaled_colors, sizes).
        Colors are pre-multiplied by lifetime alpha so draw_particles
        should receive alpha=255 (no extra depth fading for explosions).
        """
        if not self.active or len(self.positions) == 0:
            return None
        alpha = (self.lifetimes / self.max_life)[:, None]
        prescaled = np.clip(self.colors * alpha, 0, 255).astype(np.uint8)
        return self.positions.copy(), prescaled, self.sizes.copy()
