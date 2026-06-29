import sys
import pygame
import numpy as np
from config import (SCREEN_W, SCREEN_H, FPS, BG_COLOR, FOV,
                    PLANET_RADIUS, HIGH_SPEED_CRASH, COL_EARTH_ATMO)
from camera import Camera, INITIAL_POS
from stars import create_starfield, get_star_colors, update_starfield
from planet import EarthVisual
from asteroids import create_rocks, check_collisions
from explosion import Explosion
from renderer import project_points

PLAYING = 0
CRASHED = 1
DEAD    = 2


def draw_particles(surface, screen_xy, sizes, colors, alphas):
    if len(screen_xy) == 0:
        return
    pts   = screen_xy.astype(np.int32)
    scale = alphas.astype(np.float32) / 255.0
    for i in range(len(pts)):
        x, y = int(pts[i, 0]), int(pts[i, 1])
        if not (-20 <= x <= SCREEN_W + 20 and -20 <= y <= SCREEN_H + 20):
            continue
        a   = float(scale[i])
        c   = colors[i]
        col = (int(c[0] * a), int(c[1] * a), int(c[2] * a))
        pygame.draw.circle(surface, col, (x, y), max(1, int(sizes[i])))


def build_gravity_sources(rocks):
    """Return list of (center, mass) for Earth and all asteroids.
    Mass is proportional to radius^3 (same-density assumption)."""
    sources = [(np.zeros(3, dtype=np.float64), float(PLANET_RADIUS ** 3))]
    for r in rocks:
        sources.append((r.center.astype(np.float64), float(r.radius ** 3)))
    return sources


def build_scene(cam):
    star_pos, star_bright, star_sizes = create_starfield(cam.pos)
    star_colors  = get_star_colors(star_bright)
    earth_visual = EarthVisual()
    rocks, centers, radii = create_rocks()
    gravity_sources = build_gravity_sources(rocks)
    return star_pos, star_bright, star_sizes, star_colors, earth_visual, rocks, centers, radii, gravity_sources


def main():
    pygame.init()
    screen   = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Space Simulator")
    clock    = pygame.time.Clock()
    font_hud = pygame.font.SysFont(None, 22)
    font_big = pygame.font.SysFont(None, 64)
    font_sub = pygame.font.SysFont(None, 32)

    cam = Camera()
    (star_pos, star_bright, star_sizes, star_colors,
     earth_visual, rocks, ast_centers, ast_radii, gravity_sources) = build_scene(cam)

    explosions: list[Explosion] = []
    state        = PLAYING
    flash_frames = 0

    # Surfaces:
    #   glow_surf  — stars + atmosphere halo, multiplied by inv_mask before blit
    #   rock_surf  — solid Earth + solid rocks; colorkey(0,0,0) = transparent
    #   inv_mask   — white everywhere, black silhouette punched for every solid object
    #   exp_surf   — explosion particles drawn last (no masking, appear above rocks)
    glow_surf = pygame.Surface((SCREEN_W, SCREEN_H))
    rock_surf = pygame.Surface((SCREEN_W, SCREEN_H))
    inv_mask  = pygame.Surface((SCREEN_W, SCREEN_H))
    exp_surf  = pygame.Surface((SCREEN_W, SCREEN_H))
    rock_surf.set_colorkey((0, 0, 0))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and state != PLAYING:
                    cam.reset()
                    (star_pos, star_bright, star_sizes, star_colors,
                     earth_visual, rocks, ast_centers, ast_radii, gravity_sources) = build_scene(cam)
                    explosions.clear()
                    state        = PLAYING
                    flash_frames = 0

        keys = pygame.key.get_pressed()

        # ── physics ────────────────────────────────────────────────────────────
        if state == PLAYING:
            cam.update(keys, gravity_sources)
            update_starfield(star_pos, cam)

            # Earth collision (Earth at world origin, collision = inside surface)
            if np.linalg.norm(cam.pos) < PLANET_RADIUS:
                impact       = cam.speed
                cam.speed    = 0.0
                explosions.append(Explosion(cam.pos.copy(), impact))
                state        = DEAD if impact >= HIGH_SPEED_CRASH else CRASHED
                flash_frames = 14 if state == DEAD else 7

            # Asteroid collision
            elif ast_centers is not None:
                hit = check_collisions(cam.pos, ast_centers, ast_radii)
                if hit >= 0:
                    impact       = cam.speed
                    cam.speed    = 0.0
                    explosions.append(Explosion(cam.pos.copy(), impact))
                    state        = DEAD if impact >= HIGH_SPEED_CRASH else CRASHED
                    flash_frames = 14 if state == DEAD else 7

        for exp in explosions:
            exp.update()
        explosions = [e for e in explosions if e.active]

        # ── render ─────────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Layer 1a — solid objects (Earth + rocks) drawn back-to-front onto rock_surf;
        # silhouettes simultaneously punched into inv_mask.
        rock_surf.fill((0, 0, 0))
        inv_mask.fill((255, 255, 255))

        # Build a unified back-to-front draw order for Earth and all rocks.
        earth_depth  = earth_visual.depth(cam)
        rock_dists   = np.linalg.norm(
            np.array([r.center for r in rocks], dtype=np.float32)
            - cam.pos.astype(np.float32), axis=1)

        # Combine: negative index = Earth, non-negative = rock index
        all_depths = np.append(rock_dists, earth_depth)
        order      = np.argsort(all_depths)[::-1]   # far first (painter's)

        earth_screen = None   # (cx, cy, proj_r) filled in when Earth is drawn
        for oi in order:
            if oi == len(rocks):              # Earth sentinel
                earth_screen = earth_visual.draw(rock_surf, inv_mask, cam, FOV, SCREEN_W, SCREEN_H)
            else:
                rocks[oi].draw(rock_surf, inv_mask, cam, FOV, SCREEN_W, SCREEN_H)

        # Layer 1b — stars into glow_surf, then atmosphere halo, then mask.
        glow_surf.fill((0, 0, 0))

        s_xy, s_sz, s_av, s_idx = project_points(star_pos, cam, star_sizes)
        draw_particles(glow_surf, s_xy, s_sz, star_colors[s_idx], s_av)

        # Atmosphere: a glowing ring just outside the planet disk.
        # After inv_mask multiplication, the disk interior is zeroed → only the
        # outer ring survives, creating a natural limb-glow effect.
        if earth_screen is not None:
            e_cx, e_cy, e_pr = earth_screen
            atmo_w = max(5, e_pr // 10)
            atmo_r = e_pr + atmo_w
            pygame.draw.circle(glow_surf, COL_EARTH_ATMO, (e_cx, e_cy), atmo_r, atmo_w * 2)

        # Zero out glow pixels that fall behind any solid object
        glow_surf.blit(inv_mask, (0, 0), special_flags=pygame.BLEND_MULT)

        # Composite: masked glow (additive) then solid rock pixels on top
        screen.blit(glow_surf, (0, 0), special_flags=pygame.BLEND_ADD)
        screen.blit(rock_surf, (0, 0))

        # Layer 2 — explosion particles on top of everything (no depth masking)
        if explosions:
            exp_surf.fill((0, 0, 0))
            for exp in explosions:
                data = exp.get_render_data()
                if data is None:
                    continue
                e_pos, e_col, e_sz = data
                e_xy, e_psz, _, e_idx = project_points(e_pos, cam, e_sz)
                if e_idx.size:
                    draw_particles(exp_surf, e_xy, e_psz, e_col[e_idx],
                                   np.full(e_idx.size, 255, dtype=np.uint8))
            screen.blit(exp_surf, (0, 0), special_flags=pygame.BLEND_ADD)

        # Flash overlay on big crash
        if flash_frames > 0:
            intensity = int(220 * flash_frames / 14)
            flash = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            flash.fill((intensity, 15, 15, min(intensity, 190)))
            screen.blit(flash, (0, 0))
            flash_frames -= 1

        # ── HUD ────────────────────────────────────────────────────────────────
        yaw_deg   = np.degrees(cam.yaw)
        pitch_deg = np.degrees(cam.pitch)
        hud = font_hud.render(
            f"SPACE: boost   ARROWS: aim   ESC: quit   R: restart   "
            f"spd {cam.speed:.1f}   yaw {yaw_deg:+.0f}°  pitch {pitch_deg:+.0f}°   "
            f"pos ({cam.pos[0]:.0f},{cam.pos[1]:.0f},{cam.pos[2]:.0f})   "
            f"{clock.get_fps():.0f} fps",
            True, (70, 70, 70))
        screen.blit(hud, (10, 10))

        if state == DEAD:
            msg = font_big.render("DESTROYED", True, (255, 60, 30))
            sub = font_sub.render("press R to try again", True, (200, 100, 80))
            screen.blit(msg, msg.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 30)))
            screen.blit(sub, sub.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 40)))
        elif state == CRASHED:
            msg = font_big.render("CRASHED", True, (255, 140, 30))
            sub = font_sub.render("press R to try again", True, (200, 150, 80))
            screen.blit(msg, msg.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 30)))
            screen.blit(sub, sub.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 40)))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
