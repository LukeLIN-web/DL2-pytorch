# experiment use
EXPERIMENT_NAME = None

# random seed
RANDOMNESS = False
np_seed = 9973  # seed for numpy
tf_seed = 53  # seed for tf
trace_seed = 103  # seed for trace, not used

# configuration
LOG_MODE = "INFO"
if LOG_MODE == "DEBUG":
	NUM_AGENTS = 1
else:
	NUM_AGENTS = 1  # at most 28 for tesla p100 and 40 for gtx 1080ti
