import sys
import pygame
import numpy as np
from config import (SCREEN_W, SCREEN_H, FPS, BG_COLOR, FOV,
                    PLANET_RADIUS, HIGH_SPEED_CRASH)
from camera import Camera, INITIAL_POS
from stars import create_starfield, get_star_colors, update_starfield
from planet import EarthVisual
from asteroids import create_rocks, check_collisions
from explosion import Explosion
from renderer import project_points
from orbit import Orbit

PLAYING = 0
CRASHED = 1
DEAD    = 2
WON     = 3

MAX_LEVEL = 3

_EARTH_ORBIT = Orbit(PLANET_RADIUS)

# Per-level config: start_z multiplier, rock params (None = level has no rocks)
_LEVEL_CFG = {
    1: dict(start_z=4.5,  rocks=None),
    2: dict(start_z=4.5,  equatorial=False, rocks=dict(count=25,  min_r=4,  max_r=10,
                                                       ring_inner=PLANET_RADIUS*2.0,
                                                       ring_outer=PLANET_RADIUS*3.0)),
    3: dict(start_z=18.0, equatorial=True,  rocks=dict(count=100, min_r=4,  max_r=40,
                                                       ring_inner=PLANET_RADIUS*6.0,
                                                       ring_outer=PLANET_RADIUS*14.0)),
}


def _level_start(level):
    z = PLANET_RADIUS * _LEVEL_CFG[level]['start_z']
    return np.array([0.0, 0.0, z], dtype=np.float64)


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
    sources = [(np.zeros(3, dtype=np.float64), _EARTH_ORBIT.gm)]
    for r in rocks:
        sources.append((r.center.astype(np.float64), float(r.radius ** 3)))
    return sources


def build_scene(cam, level):
    star_pos, star_bright, star_sizes = create_starfield(cam.pos)
    star_colors  = get_star_colors(star_bright)
    earth_visual = EarthVisual()

    rock_cfg   = _LEVEL_CFG[level]['rocks']
    equatorial = _LEVEL_CFG[level].get('equatorial', False)
    if rock_cfg is not None:
        rocks, ast_centers, ast_radii = create_rocks(**rock_cfg)
        rng = np.random.default_rng()
        _EQ_NORMAL = np.array([0.0, 1.0, 0.0])
        for rock in rocks:
            if equatorial:
                n = _EQ_NORMAL
            else:
                n = rng.normal(0.0, 1.0, 3)
                n /= np.linalg.norm(n)
            rock.set_orbital_velocity(_EARTH_ORBIT, plane_normal=n)
        ast_centers = np.array([r.center for r in rocks], dtype=np.float32)
    else:
        rocks, ast_centers, ast_radii = [], None, None

    gravity_sources = build_gravity_sources(rocks)
    return star_pos, star_bright, star_sizes, star_colors, earth_visual, rocks, ast_centers, ast_radii, gravity_sources


def main():
    pygame.init()
    screen   = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Space Simulator")
    clock       = pygame.time.Clock()
    font_hud    = pygame.font.SysFont(None, 22)
    font_big    = pygame.font.SysFont(None, 64)
    font_sub    = pygame.font.SysFont(None, 32)
    font_lvl    = pygame.font.SysFont(None, 48)
    font_target = pygame.font.SysFont("consolas", 15)

    level       = 1
    frame_count = 0
    cam         = Camera()
    (star_pos, star_bright, star_sizes, star_colors,
     earth_visual, rocks, ast_centers, ast_radii, gravity_sources) = build_scene(cam, level)

    explosions: list[Explosion] = []
    state        = PLAYING
    flash_frames = 0

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
                    if state == WON:
                        level = level % MAX_LEVEL + 1
                    cam.reset(_level_start(level))
                    (star_pos, star_bright, star_sizes, star_colors,
                     earth_visual, rocks, ast_centers, ast_radii, gravity_sources) = build_scene(cam, level)
                    explosions.clear()
                    state        = PLAYING
                    flash_frames = 0

        keys = pygame.key.get_pressed()

        # ── physics ────────────────────────────────────────────────────────────
        if state == PLAYING:
            # Update orbiting asteroids (level 2)
            if level == 2 and rocks:
                for i, rock in enumerate(rocks):
                    rock.update_orbit(_EARTH_ORBIT)
                    ast_centers[i] = rock.center
                gravity_sources = build_gravity_sources(rocks)

            cam.update(keys, gravity_sources)
            update_starfield(star_pos, cam)

            # Reached Earth → win the level
            if np.linalg.norm(cam.pos) < PLANET_RADIUS:
                cam.velocity[:] = 0.0
                cam.speed = 0.0
                state = WON

            # Asteroid collision
            elif ast_centers is not None:
                hit = check_collisions(cam.pos, ast_centers, ast_radii)
                if hit >= 0:
                    impact       = cam.speed
                    cam.velocity[:] = 0.0
                    cam.speed    = 0.0
                    explosions.append(Explosion(cam.pos.copy(), impact))
                    state        = DEAD if impact >= HIGH_SPEED_CRASH else CRASHED
                    flash_frames = 14 if state == DEAD else 7

        for exp in explosions:
            exp.update()
        explosions = [e for e in explosions if e.active]

        # ── render ─────────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        rock_surf.fill((0, 0, 0))
        inv_mask.fill((255, 255, 255))

        # Camera-space Z depth — consistent metric for Earth and rocks
        cy_, sy_ = np.cos(cam.yaw), np.sin(cam.yaw)
        cp_, sp_ = np.cos(cam.pitch), np.sin(cam.pitch)
        fwd = np.array([sy_*cp_, sp_, cy_*cp_], dtype=np.float32)

        earth_depth = earth_visual.depth(cam)
        if rocks:
            rock_centers = np.array([r.center for r in rocks], dtype=np.float32)
            rock_depths  = (rock_centers - cam.pos.astype(np.float32)) @ fwd
            all_depths = np.append(rock_depths, earth_depth)
        else:
            all_depths = np.array([earth_depth])
        order = np.argsort(all_depths)[::-1]

        earth_screen = None
        for oi in order:
            if oi == len(rocks):
                earth_screen = earth_visual.draw(rock_surf, inv_mask, cam, FOV, SCREEN_W, SCREEN_H)
            else:
                rocks[oi].draw(rock_surf, inv_mask, cam, FOV, SCREEN_W, SCREEN_H)

        glow_surf.fill((0, 0, 0))
        s_xy, s_sz, s_av, s_idx = project_points(star_pos, cam, star_sizes)
        draw_particles(glow_surf, s_xy, s_sz, star_colors[s_idx], s_av)

        glow_surf.blit(inv_mask, (0, 0), special_flags=pygame.BLEND_MULT)
        screen.blit(glow_surf, (0, 0), special_flags=pygame.BLEND_ADD)
        screen.blit(rock_surf, (0, 0))

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

        if flash_frames > 0:
            intensity = int(220 * flash_frames / 14)
            flash = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            flash.fill((intensity, 15, 15, min(intensity, 190)))
            screen.blit(flash, (0, 0))
            flash_frames -= 1

        # ── Target reticle ─────────────────────────────────────────────────────
        if earth_screen is not None and state == PLAYING:
            ecx, ecy, epr = earth_screen
            tr  = epr + max(10, epr // 4)   # ring radius, scaled to Earth size
            p   = int(14 * abs(np.sin(frame_count * 0.05)))  # 0-14 pulse

            r_dim  = (40  + p,  0,  0)
            r_mid  = (110 + p, 10, 10)
            r_main = (205 + p, 30, 30)

            pygame.draw.circle(screen, r_dim,  (ecx, ecy), tr + 9, 1)
            pygame.draw.circle(screen, r_mid,  (ecx, ecy), tr + 4, 1)
            pygame.draw.circle(screen, r_main, (ecx, ecy), tr,     2)

            for angle in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
                dx, dy = np.cos(angle), -np.sin(angle)
                pygame.draw.line(screen, r_main,
                                 (int(ecx + dx * (tr + 5)),  int(ecy + dy * (tr + 5))),
                                 (int(ecx + dx * (tr + 13)), int(ecy + dy * (tr + 13))), 2)

            lbl = font_target.render("[ TARGET ]", True, r_main)
            screen.blit(lbl, (ecx - lbl.get_width() // 2,
                               ecy - tr - lbl.get_height() - 5))

        frame_count += 1

        # ── HUD ────────────────────────────────────────────────────────────────
        dist_earth = np.linalg.norm(cam.pos)
        yaw_deg    = np.degrees(cam.yaw)
        pitch_deg  = np.degrees(cam.pitch)
        hud = font_hud.render(
            f"SPACE: boost   ARROWS: aim   ESC: quit   R: restart   "
            f"spd {cam.speed:.1f}   dist to Earth {dist_earth:.0f}   "
            f"yaw {yaw_deg:+.0f}°  pitch {pitch_deg:+.0f}°   "
            f"{clock.get_fps():.0f} fps",
            True, (70, 70, 70))
        screen.blit(hud, (10, 10))

        lvl_label = font_lvl.render(f"LEVEL {level}", True, (60, 60, 80))
        screen.blit(lvl_label, (10, SCREEN_H - lvl_label.get_height() - 10))

        # ── State overlays ─────────────────────────────────────────────────────
        if state == WON:
            if level < MAX_LEVEL:
                msg = font_big.render("EARTH REACHED!", True, (80, 200, 120))
                sub = font_sub.render(f"press R for Level {level + 1}", True, (100, 200, 150))
            else:
                msg = font_big.render("YOU WIN!", True, (80, 200, 120))
                sub = font_sub.render("press R to play again", True, (100, 200, 150))
            screen.blit(msg, msg.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 30)))
            screen.blit(sub, sub.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 40)))

        elif state == DEAD:
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
