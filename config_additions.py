# ===========================================================================
# POLICY GRADIENT (REINFORCE) - used by train_gradient.py
# Paste this block at the END of your config.py (the genetic-algorithm
# settings above it stay as they are and are simply ignored by this script).
# ===========================================================================
PG_NUM_UPDATES = 300            # gradient updates THIS run performs
PG_EPISODES_PER_UPDATE = 8      # episodes driven (then averaged) per gradient step
PG_LEARNING_RATE = 0.01         # Adam step size - the most sensitive setting; try 0.003 - 0.03
PG_DISCOUNT = 0.99              # how much later rewards count vs immediate ones (0-1)
PG_ENTROPY_COEF = 0.01          # exploration bonus; higher = keeps trying random actions longer
PG_GRAD_CLIP = 1.0              # largest allowed gradient size per update (0 = no clipping)
PG_EVAL_EVERY = 1               # test the greedy policy every N updates (saves weights if it's a record)
PG_INIT_OUTPUT_SCALE = 0.1      # shrinks the output layer at a fresh start -> near-uniform first policy
PG_SEED = None                  # None = different every run; an integer = reproducible run

# "fresh"        -> start from random weights
# "best_weights" -> start from the weights saved in PG_BEST_WEIGHTS_PATH (e.g. to keep training)
PG_START_MODE = "fresh"

# WARNING: this is the same file watch_best.py / evaluate.py / plot_trajectories.py read, and
# train_gradient.py OVERWRITES it as soon as it starts. To keep a genetic-algorithm result,
# copy/rename that file first, or point this to a different filename.
PG_BEST_WEIGHTS_PATH = "best_weights.npy"
