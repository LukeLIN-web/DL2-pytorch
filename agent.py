import copy

import comparison
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
        tr.manual_seed(pm.tr_seed)  # specific gpu use:torch.cuda.manual_seed(seed)

    policy_net = network.PolicyNetwork("policy_net", pm.TRAINING_MODE, logger)
    logger.info("create the policy network")
    for name, param in policy_net.net.named_parameters():
        logger.info(f"name: {name}, param: {param.shape}")


    step = 1
    start_t = time.time()  # there might be occur problem , maybe I need move it to train.py

    if pm.VAL_ON_MASTER:
        validation_traces = []  # validation traces
        tags_prefix = ["DRF: ", "SRTF: ", "FIFO: ", "Tetris: ", "Optimus: "]
        for i in range(pm.VAL_DATASET):
            validation_traces.append(trace.Trace(None).get_trace())
        # deep copy to avoid changes to validation_traces
        stats = comparison.compare(copy.deepcopy(validation_traces), logger)




