import numpy as np
from config import SCREEN_W, SCREEN_H, FOV, MAX_DEPTH


def project_points(points, cam, base_size):
    """
    Project 3D world positions into screen space using the camera's orientation.

    Builds a view matrix from cam.yaw and cam.pitch, transforms world points
    into camera-local space, then applies standard perspective projection.

    Returns (screen_xy, sizes, alphas, visible_indices).
    visible_indices lets callers slice colour/data arrays to match the output.
    """
    cy, sy = np.cos(cam.yaw), np.sin(cam.yaw)
    cp, sp = np.cos(cam.pitch), np.sin(cam.pitch)

    # Camera basis vectors expressed in world space
    right   = np.array([ cy,       0.0,  -sy      ], dtype=np.float32)
    forward = np.array([ sy * cp,  sp,    cy * cp  ], dtype=np.float32)
    # up = forward × right  (gives +Y world-up when pitch=0)
    up = np.cross(forward, right)

    rel = np.asarray(points, dtype=np.float32) - cam.pos.astype(np.float32)

    # Transform into camera space
    cam_x =  rel @ right    # horizontal axis
    cam_y =  rel @ up       # vertical axis (world-up when level)
    cam_z =  rel @ forward  # depth

    visible = (cam_z > 5.0) & (cam_z < MAX_DEPTH)
    idx = np.where(visible)[0]
    if idx.size == 0:
        empty_f = np.empty((0,), dtype=np.float32)
        return np.empty((0, 2), dtype=np.float32), empty_f, empty_f.astype(np.uint8), idx

    d = cam_z[idx]
    screen_xy = np.column_stack([
         cam_x[idx] / d * FOV + SCREEN_W * 0.5,
        -cam_y[idx] / d * FOV + SCREEN_H * 0.5,  # flip Y: world-up → screen-up
    ])

    bs = np.asarray(base_size, dtype=np.float32)
    sizes  = np.clip(bs[idx] / d * FOV, 1.0, 8.0)
    alphas = np.clip(255.0 * (1.0 - d / MAX_DEPTH), 0, 255).astype(np.uint8)

    return screen_xy, sizes, alphas, idx
