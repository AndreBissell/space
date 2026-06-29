import numpy as np
from config import STAR_COUNT, MAX_DEPTH, COL_STAR


def create_starfield(cam_pos):
    """Scatter stars uniformly in all directions around the camera start position."""
    r          = np.random.uniform(200, MAX_DEPTH * 0.95, STAR_COUNT)
    cos_theta  = np.random.uniform(-1.0, 1.0, STAR_COUNT)
    phi        = np.random.uniform(0, 2 * np.pi, STAR_COUNT)
    sin_theta  = np.sqrt(1.0 - cos_theta ** 2)

    x = cam_pos[0] + r * sin_theta * np.cos(phi)
    y = cam_pos[1] + r * cos_theta
    z = cam_pos[2] + r * sin_theta * np.sin(phi)

    brightness = np.random.uniform(0.5, 1.0, STAR_COUNT)
    positions  = np.column_stack((x, y, z)).astype(np.float32)
    sizes      = np.full(STAR_COUNT, 1.0, dtype=np.float32)
    return positions, brightness, sizes


def get_star_colors(brightness):
    base = np.array(COL_STAR, dtype=np.float32)
    return np.clip(base[None, :] * brightness[:, None], 0, 255).astype(np.uint8)


def update_starfield(positions, cam):
    """
    Respawn stars that have drifted beyond the visible sphere.
    Uses world-space distance so wrapping is direction-agnostic —
    stars repopulate correctly no matter which way the player turns.
    """
    cam_pos = cam.pos.astype(np.float32)
    dist    = np.linalg.norm(positions - cam_pos, axis=1)
    too_far = dist > MAX_DEPTH * 0.9
    n       = int(np.sum(too_far))
    if n == 0:
        return

    r         = np.random.uniform(MAX_DEPTH * 0.5, MAX_DEPTH * 0.85, n)
    cos_theta = np.random.uniform(-1.0, 1.0, n)
    phi       = np.random.uniform(0, 2 * np.pi, n)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)

    positions[too_far, 0] = cam.pos[0] + r * sin_theta * np.cos(phi)
    positions[too_far, 1] = cam.pos[1] + r * cos_theta
    positions[too_far, 2] = cam.pos[2] + r * sin_theta * np.sin(phi)
