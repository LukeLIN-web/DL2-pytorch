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
	
TRAINING_MODE = "RL"  # or "SL"
if TRAINING_MODE == "RL":
	VALUE_NET = True  # disable/enable critic network
else:
	VALUE_NET = False
POLICY_NN_MODEL = None #"Models/policy_sl_ps_worker_100.ckpt"  # path of the checkpointed model, or None
VALUE_NN_MODEL = None  # "Models/value_rl_ps_worker_1000.ckpt"  # path of value network model
SAVE_VALUE_MODEL = True
if TRAINING_MODE == "SL" or VALUE_NN_MODEL is not None:
	SAVE_VALUE_MODEL = False
if TRAINING_MODE == "SL":
	POLICY_NN_MODEL = None
	VALUE_NN_MODEL = None # why 
MODEL_DIR = "Models/"  # checkpoint dir
SUMMARY_DIR = "TensorBoard/"  #  tensorboard logging dir , we could use 















# hyperparameters
OPTIMIZER = "Adam"  # RMSProp
if TRAINING_MODE == "SL":
	LEARNING_RATE = 0.005
else:
	LEARNING_RATE = 0.0001



