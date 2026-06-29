SCREEN_W = 1280
SCREEN_H  = 720
FPS       = 60
FOV       = 600
MAX_DEPTH = 3500
CAM_SPEED = 8

G         = 0.0003   # gravitational constant — tune to taste

STAR_COUNT       = 150
PLANET_PARTICLES = 2000   # kept for reference; EarthVisual no longer uses particles
PLANET_RADIUS    = 450
STARFIELD_SPREAD = 2200

COL_CORE  = (255, 220,  80)
COL_MID   = (255, 100,  30)
COL_OUTER = (200,  40, 120)
COL_STAR  = (180, 190, 255)
BG_COLOR  = (0, 0, 0)

# Earth surface colours
COL_EARTH_OCEAN = ( 20,  55, 175)
COL_EARTH_LAND  = ( 45, 130,  48)
COL_EARTH_CLOUD = (185, 200, 230)
COL_EARTH_ICE   = (225, 238, 255)
COL_EARTH_ATMO  = ( 30,  80, 210)   # atmosphere limb glow

# Asteroid
COL_ASTEROID  = (120, 100, 80)
ASTEROID_COUNT = 25          # rocks in the ring

# Explosion colours  (hot → cool)
COL_EXPLO_HOT   = (255, 230,  80)
COL_EXPLO_FIRE  = (255,  70,  20)
COL_EXPLO_EMBER = (200,  20,  10)

# Collision / game
HIGH_SPEED_CRASH = 7.0
