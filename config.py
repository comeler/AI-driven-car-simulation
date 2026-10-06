"""
config.py - every tunable parameter for the whole project, in one place.
Change values here; no other file should need editing for a normal
experiment (car handling, track shape, network size, GA hyperparameters,
training length, where a run starts from).

Every other file does `import config as cfg` and reads cfg.SOMETHING - so a
change here is picked up everywhere automatically.
"""

# ===========================================================================
# WINDOW / SIMULATION
# ===========================================================================
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600
ENV_DT = 1 / 40            # seconds simulated per step (60 steps/sec)
ENV_MAX_STEPS = 600       # episode length cap (1800 steps = 30s at 60 steps/sec)

# ===========================================================================
# CAR PHYSICS
# ===========================================================================
CAR_ACCELERATION = 500.0         # px/s^2 while accelerating
CAR_BRAKE_STRENGTH = 1200.0       # px/s^2 while braking
CAR_FRICTION = 120.0              # px/s^2 natural slowdown when coasting
CAR_MAX_SPEED = 1000.0
CAR_MAX_REVERSE_SPEED = -100.0
CAR_TURN_RATE = 18              # radians/s at full steering, scaled by speed
CAR_LENGTH = 40
CAR_WIDTH = 20

# ===========================================================================
# TRACK SHAPE - pick ONE: "oval", "wavy", "rounded_rect", "pinched"
# Only the parameters relevant to the chosen TRACK_SHAPE are actually used;
# the others are simply ignored, so you can leave every value set and just
# flip TRACK_SHAPE to try a different one.
# ===========================================================================
TRACK_SHAPE = "rounded_rect"

TRACK_CENTER_X = SCREEN_WIDTH / 2
TRACK_CENTER_Y = SCREEN_HEIGHT / 2
TRACK_NUM_POINTS = 100            # how many points make up the track's centerline
TRACK_CLOCKWISE = True            # False = car drives the loop in the opposite direction

TRACK_RX = 100                   # used by "oval", "wavy", "pinched"
TRACK_RY = 100                    # used by "oval", "wavy", "pinched"

TRACK_WOBBLE = 30               # used by "wavy" - keep well under TRACK_CORRIDOR_WIDTH
TRACK_WOBBLE_FREQUENCY = 7        # used by "wavy" - number of wobbles per lap

TRACK_RECT_WIDTH = 750            # used by "rounded_rect"
TRACK_RECT_HEIGHT = 200           # used by "rounded_rect"
TRACK_CORNER_RADIUS = 70          # used by "rounded_rect"

TRACK_NUM_PINCHES = 3             # used by "pinched"
TRACK_PINCH_STRENGTH = 0.25       # used by "pinched" - keep modest, see track.py's caution note

TRACK_CORRIDOR_WIDTH = 50     # distance between inner and outer walls, all shapes
TRACK_NUM_CHECKPOINTS = 24
TRACK_CHECKPOINT_RADIUS = 50      # how close the car must get to "hit" a checkpoint

# ===========================================================================
# SENSORS
# ===========================================================================
SENSOR_ANGLES_DEG = (-70, -30, 0, 30, 70)
SENSOR_RANGE = 250

# ===========================================================================
# REWARD SHAPING
# ===========================================================================
REWARD_PROGRESS_SCALE = 0.2        # reward per pixel closer to the next checkpoint, this frame
REWARD_PER_FRAME_PENALTY = -0.01   # constant per-frame cost, discourages idling
REWARD_CHECKPOINT_BONUS = 10.0
REWARD_OFF_TRACK_PENALTY = -70.0

# ===========================================================================
# NEURAL NETWORK
# ===========================================================================
HIDDEN_SIZE = 12

# ===========================================================================
# GENETIC ALGORITHM
# ===========================================================================
POP_SIZE = 60
NUM_GENERATIONS = 1500        # how many generations THIS run performs
MUTATION_RATE = 0.15         # probability each individual weight gets mutated
MUTATION_STRENGTH = 0.3      # size of the random nudge when it does
ELITE_FRACTION = 0.1         # fraction of the parent pool cloned unchanged each generation
HOF_SIZE = 20                # how many all-time-best networks are kept and bred from

# ===========================================================================
# WHERE A RUN STARTS FROM
# Replaces the old --resume command-line flag, since Spyder doesn't make
# passing CLI arguments convenient - just edit this value and press Run.
#
#   "fresh"        -> ignore any existing Hall of Fame file, start a brand
#                      new random population. Once this run finds new
#                      bests, it OVERWRITES the Hall of Fame file.
#   "hall_of_fame" -> load HOF_PATH (if it exists) and keep evolving from
#                      the best networks ever found, continuing the
#                      generation count instead of restarting it at 0.
# ===========================================================================
START_MODE = "hall_of_fame"

# ===========================================================================
# EVALUATION / PLOTTING
# ===========================================================================
EVAL_NUM_EPISODES = 50
PLOT_EPISODES_PER_POLICY = 3

# ===========================================================================
# FILE PATHS (created in the current working directory)
# ===========================================================================
BEST_WEIGHTS_PATH = "best_weights.npy"   # single best-ever network (watch_best.py, evaluate.py, ...)
HOF_PATH = "hall_of_fame.npz"            # top HOF_SIZE networks ever found, reloadable, updated every generation
HOF_SUMMARY_PATH = "hall_of_fame.txt"    # same top HOF_SIZE networks, as a plain-text doc for YOU to read
