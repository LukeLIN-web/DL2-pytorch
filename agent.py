import parameters as pm
import log
import torch as tr
import numpy as np
import os


def log_config():
	# log all configurations in parameters and backup py
	global LOG_DIR
	if pm.EXPERIMENT_NAME is None:
		LOG_DIR = "./backup/"
	else:
		LOG_DIR = "./" + pm.EXPERIMENT_NAME + "/"

	os.system("mkdir -p " + LOG_DIR + "; cp *.py *.txt " + LOG_DIR)

	pm_md = globals().get('pm', None) # Return the dictionary containing the current scope's global variables.
	train_config = dict()
	if pm_md:
		train_config = {key: value for key, value in pm_md.__dict__.iteritems() if not (key.startswith('__') or key.startswith('_'))}
	train_config_str = ""
	for key, value in train_config.iteritems():
		train_config_str += "{:<30}{:<100}".format(key, value) + "\n\n"

	# tb_logger.add_text(tag="Config", value=train_config_str, step=0)
	# tb_logger.flush()
    # the above is original codes, we do not write metadata and summary in trainconfig. there is a problems

	if pm.TRAINING_MODE == "SL":
		f = open(pm.MODEL_DIR + "sl_model.config", "w")
	else:
		f = open(pm.MODEL_DIR + "rl_model.config", "w")
	f.write(train_config_str)
	f.close()

	f = open(LOG_DIR + "config.md", 'w')
	f.write(train_config_str)
	f.close()


def central_agent(net_weights_qs, net_gradients_qs, stats_qs):
    logger = log.getLogger(name="central_agent", level=pm.LOG_MODE)
    logger.info("Start central agent...")

    if not pm.RANDOMNESS:
        np.random.seed(pm.np_seed)
        tr.manual_seed(pm.tr_seed) # specific gpu use:torch.cuda.manual_seed(seed)
        
        """need to log all configurations in parameters and backup
        py
        It uses a tf class tb_log.py"""
        
