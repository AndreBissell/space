import numpy as np
import pygame
from config import (PLANET_RADIUS, MAX_DEPTH, FOV,
                    COL_EARTH_OCEAN, COL_EARTH_LAND, COL_EARTH_CLOUD,
                    COL_EARTH_ICE,   COL_EARTH_ATMO,
                    PLANET_PARTICLES,
                    COL_CORE, COL_MID, COL_OUTER)


class EarthVisual:
    """
    Solid Earth rendered as a shaded filled circle with projected surface features.

    Features (continents, ice caps, clouds) are pre-generated as 3D points on the
    unit sphere.  Each frame they're projected to screen so they shift with perspective
    and disappear around the limb — no surface triangulation needed.

    draw() writes into rock_surf (opaque) and inv_mask (black silhouette), so the
    planet properly occludes stars and other glow particles behind it.
    draw() returns (cx, cy, proj_r) so the caller can paint the atmosphere halo
    onto glow_surf before masking.
    """

    _COLORS = {
        'land':  np.array(COL_EARTH_LAND,  dtype=np.float32),
        'cloud': np.array(COL_EARTH_CLOUD, dtype=np.float32),
        'ice':   np.array(COL_EARTH_ICE,   dtype=np.float32),
    }

    def __init__(self):
        rng = np.random.default_rng(seed=7)  # fixed seed → same Earth every run

        positions, sizes, kinds = [], [], []

        # 10 continent blobs spread across both hemispheres
        for _ in range(10):
            cos_t = rng.uniform(-0.78, 0.78)
            phi   = rng.uniform(0, 2 * np.pi)
            sin_t = np.sqrt(max(0.0, 1.0 - cos_t ** 2))
            positions.append([sin_t * np.cos(phi), cos_t, sin_t * np.sin(phi)])
            sizes.append(rng.uniform(0.09, 0.22))
            kinds.append('land')

        # North and south polar ice caps
        positions.append([0.0,  1.0, 0.0])
        sizes.append(0.20)
        kinds.append('ice')
        positions.append([0.0, -1.0, 0.0])
        sizes.append(0.17)
        kinds.append('ice')

        # 7 cloud wisps
        for _ in range(7):
            cos_t = rng.uniform(-0.65, 0.65)
            phi   = rng.uniform(0, 2 * np.pi)
            sin_t = np.sqrt(max(0.0, 1.0 - cos_t ** 2))
            positions.append([sin_t * np.cos(phi), cos_t, sin_t * np.sin(phi)])
            sizes.append(rng.uniform(0.04, 0.11))
            kinds.append('cloud')

        self._pos   = np.array(positions, dtype=np.float32)   # (N, 3) unit vectors
        self._sizes = np.array(sizes,     dtype=np.float32)   # (N,)
        self._kinds = kinds                                     # list[str]

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _basis(cam):
        cy, sy = np.cos(cam.yaw),   np.sin(cam.yaw)
        cp, sp = np.cos(cam.pitch), np.sin(cam.pitch)
        right   = np.array([ cy,    0.0, -sy   ], np.float32)
        forward = np.array([ sy*cp, sp,   cy*cp], np.float32)
        up      = np.cross(forward, right)
        return right, forward, up

    def depth(self, cam):
        """Camera-space depth of Earth's centre (for painter's-algorithm sorting)."""
        _, forward, _ = self._basis(cam)
        return float(-cam.pos.astype(np.float32) @ forward)

    # ── main draw ─────────────────────────────────────────────────────────────
    def draw(self, rock_surf, mask_surf, cam, fov, sw, sh):
        """
        Returns (cx, cy, proj_r) — screen position and pixel radius — so the
        caller can add the atmosphere halo to glow_surf.  Returns None if Earth
        is outside the view frustum.
        """
        right, forward, up = self._basis(cam)
        cam_pos = cam.pos.astype(np.float32)

        # Earth is at world origin
        rel     = -cam_pos                     # origin - cam_pos
        cam_z   = float(rel @ forward)
        if cam_z < 10.0 or cam_z > MAX_DEPTH:
            return None

        cam_x = float(rel @ right)
        cam_y = float(rel @ up)
        sw2, sh2 = sw * 0.5, sh * 0.5

        cx  = int(cam_x / cam_z * fov + sw2)
        cy_ = int(-cam_y / cam_z * fov + sh2)

        proj_r = min(int(PLANET_RADIUS / cam_z * fov), sw * 3)
        if proj_r < 2:
            return None

        # ── ocean base ───────────────────────────────────────────────────────
        pygame.draw.circle(rock_surf, COL_EARTH_OCEAN, (cx, cy_), proj_r)

        # ── surface features (vectorised hemisphere + projection) ─────────────
        # Feature world positions (on sphere surface)
        feat_world = self._pos * PLANET_RADIUS          # (N, 3)
        feat_rel   = feat_world - cam_pos               # (N, 3)
        feat_cz    = feat_rel @ forward                 # (N,)
        feat_cx    = feat_rel @ right
        feat_cy    = feat_rel @ up

        # Front-hemisphere check: dot(unit_pos, cam_pos) > 0 (Earth at origin)
        front = np.einsum('ij,j->i', self._pos, cam_pos)   # (N,)
        # Foreshortening: cos of angle between feature normal and Earth→cam axis
        cam_dir  = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
        dot_f    = np.clip(np.einsum('ij,j->i', self._pos, cam_dir), 0.0, 1.0)

        visible  = (front > 0) & (feat_cz > 1.0)
        vis_idx  = np.where(visible)[0]

        for i in vis_idx:
            fx  = int(feat_cx[i] / feat_cz[i] * fov + sw2)
            fy  = int(-feat_cy[i] / feat_cz[i] * fov + sh2)
            fr  = max(2, int(self._sizes[i] * proj_r * float(dot_f[i])))
            col = self._COLORS[self._kinds[i]]
            # Simple diffuse shading: brighter if facing the notional sun
            shade = 0.7 + 0.3 * float(dot_f[i])
            pygame.draw.circle(rock_surf,
                               tuple(int(c * shade) for c in col),
                               (fx, fy), fr)

        # ── silhouette mask ───────────────────────────────────────────────────
        pygame.draw.circle(mask_surf, (0, 0, 0), (cx, cy_), proj_r)

        return cx, cy_, proj_r


# ── legacy particle planet (unused by main) ───────────────────────────────────

def create_planet():
    phi       = np.random.uniform(0, 2 * np.pi, PLANET_PARTICLES)
    cos_theta = np.random.uniform(-1.0, 1.0, PLANET_PARTICLES)
    theta     = np.arccos(cos_theta)
    radius    = np.random.normal(PLANET_RADIUS, PLANET_RADIUS * 0.055, PLANET_PARTICLES)
    radius    = np.clip(radius, PLANET_RADIUS * 0.85, PLANET_RADIUS * 1.15)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    positions = np.column_stack((x, y, z)).astype(np.float32)
    palette   = np.array([COL_OUTER, COL_MID, COL_CORE], dtype=np.float32)
    weights   = np.random.choice([0, 1, 2], size=PLANET_PARTICLES, p=[0.35, 0.45, 0.20])
    colors    = palette[weights]
    eq_factor = 1.0 - np.clip(np.abs(y) / (PLANET_RADIUS * 1.1), 0.0, 1.0)
    brightness = 0.70 + eq_factor * 0.30
    colors    = np.clip(colors * brightness[:, None], 0, 255).astype(np.uint8)
    base_sizes = np.random.uniform(2.0, 4.0, PLANET_PARTICLES).astype(np.float32)
    return positions, colors, base_sizes
