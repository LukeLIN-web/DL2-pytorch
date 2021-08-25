# coding=utf-8
import copy
import torch
import comparison
import validate
import parameters as pm
import log
import time
import numpy as np
import network
import os
import trace


def test(policy_net, validation_traces, logger, step):
    LOG_DIR = "./backup/"  # stick LOGDIR .
    val_tic = time.time()
    tag_prefix = "Central "
    # except Exception as e:
    #     logger.error("Error when validation! " + str(e))
    # try: # I don't think it make sense,
    if pm.TRAINING_MODE == "SL":
        val_loss = validate.val_loss(policy_net, copy.deepcopy(validation_traces), logger, step)
    print("11")
    jct, makespan, reward = validate.val_jmr(policy_net, copy.deepcopy(validation_traces), logger, step)
    val_toc = time.time()

    logger.info("Central Agent:" + " Validation at step " + str(step) + " Time: " + '%.3f' % (val_toc - val_tic))

    # log results
    print(val_toc)
    if pm.TRAINING_MODE == "SL":
        f = open(LOG_DIR + "sl_validation.txt", 'a')
    else:
        f = open(LOG_DIR + "rl_validation.txt", 'a')
    f.write("step " + str(step) + ": " + str(jct) + " " + str(makespan) + " " + str(reward) + "\n")
    f.close()
    return jct, makespan, reward


def central_agent(net_weights_qs, net_gradients_qs, stats_qs):
    logger = log.getLogger(name="central_agent", level=pm.LOG_MODE)
    logger.info("Start central agent...")

    if not pm.RANDOMNESS:
        np.random.seed(pm.np_seed)
        torch.manual_seed(pm.tr_seed)  # specific gpu use:torch.cuda.manual_seed(seed)

    policy_net = network.PolicyNetwork("policy_net", pm.TRAINING_MODE, logger)
    if pm.VALUE_NET:
        value_net = network.ValueNetwork("value_net", pm.TRAINING_MODE, logger)

    logger.info("create the policy network")
    for name, param in policy_net.net.named_parameters():
        logger.info(f"name: {name}, param: {param.shape}")
    if pm.POLICY_NN_MODEL is not None:
        policy_net.net.load_state_dict(torch.load('policy.params'))
        logger.info("Policy model " + pm.POLICY_NN_MODEL + " is restored.")

    if pm.VALUE_NET:
        if pm.VALUE_NN_MODEL is not None:
            value_net.net.load_state_dict(torch.load('value.params'))
            logger.info("Value model " + pm.VALUE_NN_MODEL + " is restored.")

    step = 1
    start_t = time.time()  # there might be occur problem , maybe I need move it to train.py

    if pm.VAL_ON_MASTER:
        validation_traces = []  # validation traces
        tags_prefix = ["DRF: ", "SRTF: ", "FIFO: ", "Tetris: ", "Optimus: "]
        for i in range(pm.VAL_DATASET):
            validation_traces.append(trace.Trace(None).get_trace())
        # stats = comparison.compare(copy.deepcopy(validation_traces), logger)  # 'compare' method needs ten seconds
        stats = []
        # deep copy to avoid changes to validation_traces
        if not pm.SKIP_FIRST_VAL:
            stats.append(test(policy_net, copy.deepcopy(validation_traces), logger, step=0))
            tags_prefix.append("Init_NN: ")
