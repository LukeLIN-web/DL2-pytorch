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
POLICY_NN_MODEL = None  # "Models/policy_sl_ps_worker_100.ckpt"  # path of the checkpointed model, or None
VALUE_NN_MODEL = None  # "Models/value_rl_ps_worker_1000.ckpt"  # path of value network model
SAVE_VALUE_MODEL = True
if TRAINING_MODE == "SL" or VALUE_NN_MODEL is not None:
    SAVE_VALUE_MODEL = False
if TRAINING_MODE == "SL":
    POLICY_NN_MODEL = None
    VALUE_NN_MODEL = None  # why
MODEL_DIR = "Models/"  # checkpoint dir
SUMMARY_DIR = "TensorBoard/"  # tensorboard logging dir , we could use

# hyperparameters
OPTIMIZER = "Adam"  # RMSProp
if TRAINING_MODE == "SL":
    LEARNING_RATE = 0.005
else:
    LEARNING_RATE = 0.0001

VAL_ON_MASTER = True  # validation on agent uses CPU instead of GPU, and may cause use up all memory, do not know why, so far it must be set true
NUM_UNCOMPLETED_JOB_REWARD = False  # set the reward to be the number of uncompleted jobs
FINE_GRAIN_JCT = True

# cluster
TESTBED = True
LARGE_SCALE = False
NUM_RESR_TYPES = 2  # number of resource types, e.g., cpu,gpu
NUM_RESR_SLOTS = 8  # number of available resource slots on each machine
if TESTBED:
    CLUSTER_NUM_NODES = 6
elif LARGE_SCALE:
    CLUSTER_NUM_NODES = 500

# dataset
JOB_EPOCH_EST_ERROR = 0
TRAIN_SPEED_ERROR = 0
FIX_JOB_LEN = True
JOB_LEN_PATTERN = "Normal"  # Ali_Trace, Normal
JOB_ARRIVAL_PATTERN = "Uniform"  # Ali_Trace, Uniform, Google_Trace, Poisson
MAX_ARRVS_PER_TS = 3  # max number of jobs arrived in one time slot
MAX_NUM_EPOCHS = 30000  # maximum duration of jobs, epochs. default 200
MAX_NUM_WORKERS = 16
TS_DURATION = 1200.0
if LARGE_SCALE:
    TOT_NUM_JOBS = 200  # number of jobs in one trace
    MAX_ARRVS_PER_TS = 10  # max number of jobs arrived in one time slot
if TESTBED:
    TOT_NUM_JOBS = 10  # nunmber of jobs in one trace
    MAX_NUM_EPOCHS = 1000
    MAX_ARRVS_PER_TS = 5  # max number of jobs arrived in one time slot
    TS_DURATION = 300.0
    SCHED_WINDOW_SIZE = 4
VAL_DATASET = 10  # number of traces for validation in each agent
# neural network
PS_WORKER = True  # whether consider ps and worker tasks separately or not
