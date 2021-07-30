# experiment use
EXPERIMENT_NAME = None

# random seed
RANDOMNESS = False
np_seed = 9973  # seed for numpy
tr_seed = 53  # seed for torch
trace_seed = 103  # seed for trace, not used

# configuration
LOG_MODE = "INFO"
if LOG_MODE == "DEBUG":
	NUM_AGENTS = 1
else:
	NUM_AGENTS = 1  # at most 28 for tesla p100 and 40 for gtx 1080ti
	

MODEL_DIR = "Models/"  # checkpoint dir
SUMMARY_DIR = "TensorBoard/"  #  tensorboard logging dir , we could use 







