import numpy as np
import pygame
from config import ASTEROID_COUNT, MAX_DEPTH, PLANET_RADIUS

RING_INNER   = PLANET_RADIUS * 2.0    # inner edge of asteroid belt
RING_OUTER   = PLANET_RADIUS * 3.0    # outer edge
RING_Y_SCALE = 0.28                   # vertical spread as fraction of ring radius
ASTEROID_MIN_R = 4
ASTEROID_MAX_R = 10

# Directional sun — sets light/shadow on rock faces
_SUN = np.array([0.45, 0.70, 0.25], dtype=np.float32)
_SUN /= np.linalg.norm(_SUN)

_ROCK_PALETTES = [
    (115, 92, 70),   # clay brown
    (62,  62, 68),   # dark basalt
    (148, 78, 55),   # iron-red
    (172, 148, 98),  # sandy limestone
    (78,  74, 84),   # slate gray
    (98,  82, 55),   # mudstone
]


def _build_icosahedron():
    """Unit icosahedron: 12 verts, 20 triangular faces."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([
        [0,11, 5], [0, 5, 1], [0, 1, 7], [0, 7,10], [0,10,11],
        [1, 5, 9], [5,11, 4], [11,10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4,11], [6, 2,10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)
    return v, f


_ICO_VERTS, _ICO_FACES = _build_icosahedron()


class Rock:
    """
    A single solid asteroid rendered as a shaded low-poly 3D mesh.
    Shape is an icosahedron with per-axis and per-vertex scale perturbation
    so every rock looks different.
    """

    def __init__(self, center, radius, color):
        self.center   = np.array(center, dtype=np.float32)
        self.radius   = float(radius)
        self.faces    = _ICO_FACES
        self.velocity = np.zeros(3, dtype=np.float64)

        # Irregular shape: stretch axes randomly, then jitter each vertex
        axis_s = np.random.uniform(0.55, 1.45, 3)
        vert_s = np.random.uniform(0.50, 1.50, len(_ICO_VERTS))
        self.local_verts = (_ICO_VERTS * axis_s[None, :] * vert_s[:, None] * radius
                            ).astype(np.float32)   # (12, 3) offset from center

        self._bake_lighting(color)

    def set_orbital_velocity(self, earth_orbit, fraction=1.0, plane_normal=None):
        """Give this rock a circular orbital velocity around Earth (origin).
        plane_normal defines the orbital plane (default: Y axis = equatorial).
        fraction < 1.0 makes orbits sub-circular."""
        pos = self.center.astype(np.float64)
        r   = np.linalg.norm(pos)
        if r < 1.0:
            return
        pos_dir = pos / r
        n = np.array(plane_normal, dtype=np.float64) if plane_normal is not None \
            else np.array([0.0, 1.0, 0.0])
        vel_dir = np.cross(n, pos_dir)
        vl = np.linalg.norm(vel_dir)
        if vl < 1e-6:                          # n parallel to pos — pick a fallback
            n = np.array([1.0, 0.0, 0.0])
            vel_dir = np.cross(n, pos_dir)
            vl = np.linalg.norm(vel_dir)
        self.velocity = (vel_dir / vl) * earth_orbit.speed(r) * fraction

    def update_orbit(self, earth_orbit):
        """Advance one physics step: apply Earth gravity, move."""
        pos = self.center.astype(np.float64)
        r   = np.linalg.norm(pos)
        if r > 1.0:
            self.velocity += (-pos / r) * (earth_orbit.gm / r ** 2)
        self.center = (pos + self.velocity).astype(np.float32)

    def _bake_lighting(self, base_color):
        """Pre-compute per-face colour under the fixed sun direction."""
        v = self.local_verts
        v0 = v[self.faces[:, 0]]
        v1 = v[self.faces[:, 1]]
        v2 = v[self.faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms   = np.linalg.norm(normals, axis=1, keepdims=True)
        norms   = np.where(norms < 1e-8, 1.0, norms)
        normals /= norms

        diffuse = np.clip(normals @ _SUN, 0.12, 1.0)   # (20,)
        bc = np.array(base_color, dtype=np.float32)
        self.face_colors  = np.clip(bc[None, :] * diffuse[:, None], 0, 255).astype(np.uint8)
        # Slightly darker colour for the 1-px edge lines
        self.edge_colors  = np.clip(bc[None, :] * (diffuse * 0.55)[:, None], 0, 255).astype(np.uint8)
        # Face centers in local space (for depth sort and backface test)
        self.face_centers = ((v0 + v1 + v2) / 3.0).astype(np.float32)  # (20, 3)
        self.face_normals = normals.astype(np.float32)                   # (20, 3)

    # ── rendering ──────────────────────────────────────────────────────────────
    def _project(self, cam, fov, sw, sh):
        """
        Returns (order, vx, vy, vz, sw2, sh2) — projected vertex coords and
        sorted visible face indices — or None if the rock is fully culled.
        """
        cy, sy = np.cos(cam.yaw),   np.sin(cam.yaw)
        cp, sp = np.cos(cam.pitch), np.sin(cam.pitch)
        right   = np.array([ cy,     0.0, -sy    ], np.float32)
        forward = np.array([ sy*cp,  sp,   cy*cp ], np.float32)
        up      = np.cross(forward, right)

        rel_c = self.center - cam.pos.astype(np.float32)
        cz_c  = float(rel_c @ forward)
        if cz_c < -self.radius or cz_c > MAX_DEPTH:
            return None

        rel = self.local_verts + self.center - cam.pos.astype(np.float32)
        vx  =  rel @ right
        vy  =  rel @ up
        vz  =  rel @ forward

        fc_rel = self.face_centers + self.center - cam.pos.astype(np.float32)
        fc_z   = fc_rel @ forward
        vis    = (fc_z > 1.0) & (np.einsum('ij,ij->i', self.face_normals, fc_rel) < 0)
        idx    = np.where(vis)[0]
        if idx.size == 0:
            return None

        order = idx[np.argsort(fc_z[idx])[::-1]]   # back → front
        return order, vx, vy, vz, sw * 0.5, sh * 0.5

    def draw(self, rock_surf, mask_surf, cam, fov, sw, sh):
        """
        Draw shaded faces onto rock_surf and black silhouette onto mask_surf.
        Drawing both in one pass lets the caller multiply the mask into the
        glow layer to hide stars/Earth behind solid rock geometry.
        """
        proj = self._project(cam, fov, sw, sh)
        if proj is None:
            return
        order, vx, vy, vz, sw2, sh2 = proj

        _BLACK = (0, 0, 0)
        for fi in order:
            vi  = self.faces[fi]
            fvz = vz[vi]
            if np.any(fvz < 1.0):
                continue
            sx  = ( vx[vi] / fvz * fov + sw2).astype(int)
            sy_ = (-vy[vi] / fvz * fov + sh2).astype(int)
            pts = [(sx[0], sy_[0]), (sx[1], sy_[1]), (sx[2], sy_[2])]
            pygame.draw.polygon(rock_surf,  tuple(self.face_colors[fi].tolist()), pts)
            pygame.draw.polygon(rock_surf,  tuple(self.edge_colors[fi].tolist()), pts, 1)
            pygame.draw.polygon(mask_surf,  _BLACK, pts)


# ── scene generation ───────────────────────────────────────────────────────────

def create_rocks(count=None, min_r=None, max_r=None, ring_inner=None, ring_outer=None):
    """
    Build an asteroid belt in a torus around Earth (which sits at the world origin).
    All size/ring parameters override the module-level defaults when provided.
    Returns (rocks, centers_array, radii_array).
    """
    count      = ASTEROID_COUNT if count      is None else count
    min_r      = ASTEROID_MIN_R if min_r      is None else min_r
    max_r      = ASTEROID_MAX_R if max_r      is None else max_r
    ring_inner = RING_INNER     if ring_inner is None else ring_inner
    ring_outer = RING_OUTER     if ring_outer is None else ring_outer

    rng    = np.random.default_rng()
    rocks  = []
    target = count

    while len(rocks) < target:
        roll = rng.random()

        phi    = rng.uniform(0, 2 * np.pi)
        ring_r = rng.uniform(ring_inner, ring_outer)
        cx     = ring_r * np.cos(phi)
        cz     = ring_r * np.sin(phi)
        cy     = rng.uniform(-ring_r * RING_Y_SCALE, ring_r * RING_Y_SCALE)

        col = _ROCK_PALETTES[int(rng.integers(len(_ROCK_PALETTES)))]

        if roll < 0.25 and len(rocks) + 3 <= target + 2:
            # 3-rock cluster
            r0 = rng.uniform(min_r, max_r)
            rocks.append(Rock([cx, cy, cz], r0, col))
            for _ in range(2):
                off = r0 * rng.uniform(2.0, 3.5)
                c2  = _ROCK_PALETTES[int(rng.integers(len(_ROCK_PALETTES)))]
                rocks.append(Rock(
                    [cx + rng.uniform(-off, off),
                     cy + rng.uniform(-off * 0.4, off * 0.4),
                     cz + rng.uniform(-off, off)],
                    rng.uniform(min_r * 0.5, r0 * 0.8), c2))

        elif roll < 0.50 and len(rocks) + 2 <= target + 1:
            # Binary pair
            r0  = rng.uniform(min_r, max_r)
            r1  = rng.uniform(min_r * 0.5, r0 * 0.85)
            sep = (r0 + r1) * rng.uniform(1.5, 2.5)
            rocks.append(Rock([cx, cy, cz], r0, col))
            c2  = _ROCK_PALETTES[int(rng.integers(len(_ROCK_PALETTES)))]
            rocks.append(Rock(
                [cx + rng.uniform(-sep, sep),
                 cy + rng.uniform(-sep * 0.3, sep * 0.3),
                 cz + rng.uniform(-sep, sep)],
                r1, c2))

        else:
            # Solo
            r = rng.uniform(min_r, max_r)
            rocks.append(Rock([cx, cy, cz], r, col))

    centers = np.array([r.center for r in rocks], dtype=np.float32)
    radii   = np.array([r.radius for r in rocks], dtype=np.float32)
    return rocks, centers, radii


def check_collisions(cam_pos, centers, radii):
    dists = np.linalg.norm(centers - cam_pos.astype(np.float32), axis=1)
    hits  = np.where(dists < radii * 1.05)[0]
    return int(hits[0]) if hits.size else -1
