import parameters as pm
import log
import time
import torch as tr
import numpy as np
import network
import os
import trace

def central_agent(net_weights_qs, net_gradients_qs, stats_qs):
	logger = log.getLogger(name="central_agent", level=pm.LOG_MODE)
	logger.info("Start central agent...")

	if not pm.RANDOMNESS:
		np.random.seed(pm.np_seed)
		tr.manual_seed(pm.tr_seed) # specific gpu use:torch.cuda.manual_seed(seed)
		
	"""need to log all configurations in parameters and backup
		py
		It uses a tf class tb_log.py"""
	policy_net = network.PolicyNetwork("policy_net", pm.TRAINING_MODE, logger)
	if pm.VALUE_NET:
		value_net = network.ValueNetwork("value_net", pm.TRAINING_MODE, logger)
	#logger.info("Create the policy network, with "+str(policy_net.get_num_weights())+" parameters")
	logger.info("create the policy network")
	for name,param in policy_net.net.named_parameters():
		logger.info(f"name: {name}, param: {param.shape}")
	
	if pm.POLICY_NN_MODEL is not None:
		checkpoint = tr.load(pm.POLICY_NN_MODEL)
		policy_net.net.load_state_dict(checkpoint['model'])
		policy_net.optimize.load_state_dict(checkpoint['optimizer'])
		epoch = checkpoint(['epoch']) # todo : load epoch
		logger.info("Policy model "+pm.POLICY_NN_MODEL+" is restored.")

	if pm.VALUE_NN_MODEL is not None:
		checkpoint = tr.load(pm.VALUE_NN_MODEL)
		value_net.net.load_state_dict(checkpoint['model'])
		value_net.optimize.load_state_dict(checkpoint['optimizer'])
		epoch = checkpoint(['epoch']) # todo : load epoch
		logger.info("Value model " + pm.VALUE_NN_MODEL + " is restored.")
	
	step = 1
	start_t = time.time() # there might be occur problem , maybe I need move it to train.py

	if pm.VAL_ON_MASTER:
		validation_traces = []  # validation traces
		tags_prefix = ["DRF: ", "SRTF: ", "FIFO: ", "Tetris: ", "Optimus: "]
		for i in range(pm.VAL_DATASET):
			validation_traces.append(trace.Trace(None).get_trace())
			










	



	






