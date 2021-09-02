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
from torch.utils.tensorboard import SummaryWriter


def collect_stats(stats_qs, writer, step):
    policy_entropys = []
    policy_losses = []
    value_losses = []
    td_losses = []
    step_rewards = []
    jcts = []
    makespans = []
    rewards = []
    val_losses = []
    val_jcts = []
    val_makespans = []
    val_rewards = []
    for agent in range(pm.NUM_AGENTS):
        while not stats_qs[agent].empty():
            stats = stats_qs[agent].get()
            tag_prefix = "SAgent " + str(agent) + " "
            print(stats[0])
            if stats[0] == "step:sl":
                _, entropy, loss = stats
                policy_entropys.append(entropy)
                policy_losses.append(loss)
                if agent < pm.NUM_RECORD_AGENTS and pm.EXPERIMENT_NAME is None:
                    writer.add_scalar(tag=tag_prefix + "SL Loss", value=loss, step=step)
                    writer.add_scalar(tag=tag_prefix + "SL Entropy", value=entropy, step=step)
            elif stats[0] == "val":
                _, val_loss, jct, makespan, reward = stats
                val_losses.append(val_loss)
                val_jcts.append(jct)
                val_makespans.append(makespan)
                val_rewards.append(reward)
            if id < pm.NUM_RECORD_AGENTS and pm.EXPERIMENT_NAME is None:
                writer.add_scalar(tag=tag_prefix + "Val Loss", value=val_loss, step=step)
                writer.add_scalar(tag=tag_prefix + "Val JCT", value=jct, step=step)
                writer.add_scalar(tag=tag_prefix + "Val Makespan", value=makespan, step=step)
                writer.add_scalar(tag=tag_prefix + "Val Reward", value=reward, step=step)
            elif stats[0] == "step:policy":
                _, entropy, loss, td_loss, step_reward, output = stats
                policy_entropys.append(entropy)
                policy_losses.append(loss)
                td_losses.append(td_loss)
                step_rewards.append(step_reward)
                if id < pm.NUM_RECORD_AGENTS and pm.EXPERIMENT_NAME is None:
                    writer.add_scalar(tag=tag_prefix + "Policy Entropy", value=entropy, step=step)
                    writer.add_scalar(tag=tag_prefix + "Policy Loss", value=loss, step=step)
                    writer.add_scalar(tag=tag_prefix + "TD Loss", value=td_loss, step=step)
                    writer.add_scalar(tag=tag_prefix + "Step Reward", value=step_reward, step=step)
                    writer.add_histogram(tag=tag_prefix + "Output", value=output, step=step)
            elif stats[0] == "step:policy+value":
                _, entropy, policy_loss, value_loss, td_loss, step_reward, output = stats
                policy_entropys.append(entropy)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                td_losses.append(td_loss)
                step_rewards.append(step_reward)
                if id < pm.NUM_RECORD_AGENTS and pm.EXPERIMENT_NAME is None:
                    writer.add_scalar(tag=tag_prefix + "Policy Entropy", value=entropy, step=step)
                    writer.add_scalar(tag=tag_prefix + "Policy Loss", value=policy_loss, step=step)
                    writer.add_scalar(tag=tag_prefix + "Value Loss", value=value_loss, step=step)
                    writer.add_scalar(tag=tag_prefix + "TD Loss", value=td_loss, step=step)
                    writer.add_scalar(tag=tag_prefix + "Step Reward", value=step_reward, step=step)
                    writer.add_histogram(tag=tag_prefix + "Output", value=output, step=step)
            elif stats[0] == "trace:sched_result":
                _, jct, makespan, reward = stats
                jcts.append(jct)
                makespans.append(makespan)
                rewards.append(reward)
                if id < pm.NUM_RECORD_AGENTS and pm.EXPERIMENT_NAME is None:
                    writer.add_scalar(tag=tag_prefix + "Avg JCT", value=jct, step=step)
                    writer.add_scalar(tag=tag_prefix + "Makespan", value=makespan, step=step)
                    writer.add_scalar(tag=tag_prefix + "Reward", value=reward, step=step)
            elif stats[0] == "trace:job_stats":
                _, episode, jobstats = stats
                if id < pm.NUM_RECORD_AGENTS and pm.EXPERIMENT_NAME is None:
                    job_stats_tag_prefix = tag_prefix + "Trace " + str(episode) + " Step " + str(step) + " "
                    for i in range(len(jobstats["arrival"])):
                        writer.add_scalar(tag=job_stats_tag_prefix + "Arrival", value=jobstats["arrival"][i], step=i)
                    for i in range(len(jobstats["ts_completed"])):
                        writer.add_scalar(tag=job_stats_tag_prefix + "Ts_completed",
                                             value=jobstats["ts_completed"][i], step=i)
                    for i in range(len(jobstats["tot_completed"])):
                        writer.add_scalar(tag=job_stats_tag_prefix + "Tot_completed",
                                             value=jobstats["tot_completed"][i], step=i)
                    for i in range(len(jobstats["uncompleted"])):
                        writer.add_scalar(tag=job_stats_tag_prefix + "Uncompleted", value=jobstats["uncompleted"][i],
                                             step=i)
                    for i in range(len(jobstats["running"])):
                        writer.add_scalar(tag=job_stats_tag_prefix + "Running", value=jobstats["running"][i], step=i)
                    for i in range(len(jobstats["total"])):
                        writer.add_scalar(tag=job_stats_tag_prefix + "Total jobs", value=jobstats["total"][i],
                                             step=i)
                    for i in range(len(jobstats["backlog"])):
                        writer.add_scalar(tag=job_stats_tag_prefix + "Backlog", value=jobstats["backlog"][i], step=i)
                    for i in range(len(jobstats["cpu_util"])):
                        writer.add_scalar(tag=job_stats_tag_prefix + "CPU_Util", value=jobstats["cpu_util"][i],
                                             step=i)
                    for i in range(len(jobstats["gpu_util"])):
                        writer.add_scalar(tag=job_stats_tag_prefix + "GPU_Util", value=jobstats["gpu_util"][i],
                                             step=i)
                    writer.add_histogram(tag=job_stats_tag_prefix + "JCT", value=jobstats["duration"], step=step)

    tag_prefix = "Central "
    if len(policy_entropys) > 0:
        writer.add_scalar(tag=tag_prefix + "Policy Entropy", value=sum(policy_entropys) / len(policy_entropys),
                             step=step)
    if len(policy_losses) > 0:
        writer.add_scalar(tag=tag_prefix + "Policy Loss", value=sum(policy_losses) / len(policy_losses), step=step)
    if len(value_losses) > 0:
        writer.add_scalar(tag=tag_prefix + "Value Loss", value=sum(value_losses) / len(value_losses), step=step)
    if len(td_losses) > 0:
        writer.add_scalar(tag=tag_prefix + "TD Loss / Advantage", value=sum(td_losses) / len(td_losses), step=step)
    if len(step_rewards) > 0:
        writer.add_scalar(tag=tag_prefix + "Batch Reward", value=sum(step_rewards) / len(step_rewards), step=step)
    if len(jcts) > 0:
        writer.add_scalar(tag=tag_prefix + "JCT", value=sum(jcts) / len(jcts), step=step)
        # log results
        if pm.TRAINING_MODE == "SL":
            f = open(LOG_DIR + "sl_train_jct.txt", 'a')
        else:
            f = open(LOG_DIR + "rl_train_jct.txt", 'a')
        f.write("step " + str(step) + ": " + str(sum(jcts) / len(jcts)) + "\n")
        f.close()
    if len(makespans) > 0:
        writer.add_scalar(tag=tag_prefix + "Makespan", value=sum(makespans) / len(makespans), step=step)
        # log results
        if pm.TRAINING_MODE == "SL":
            f = open(LOG_DIR + "sl_train_makespan.txt", 'a')
        else:
            f = open(LOG_DIR + "rl_train_makespan.txt", 'a')
        f.write("step " + str(step) + ": " + str(sum(makespans) / len(makespans)) + "\n")
        f.close()
    if len(rewards) > 0:
        writer.add_scalar(tag=tag_prefix + "Reward", value=sum(rewards) / len(rewards), step=step)
    if len(val_losses) > 0:
        writer.add_scalar(tag=tag_prefix + "Val Loss", value=sum(val_losses) / len(val_losses), step=step)
    if len(val_jcts) > 0:
        writer.add_scalar(tag=tag_prefix + "Val JCT", value=sum(val_jcts) / len(val_jcts), step=step)
    if len(val_makespans) > 0:
        writer.add_scalar(tag=tag_prefix + "Val Makespan", value=sum(val_makespans) / len(val_makespans), step=step)
    if len(val_rewards) > 0:
        writer.add_scalar(tag=tag_prefix + "Val Reward", value=sum(val_rewards) / len(val_rewards), step=step)
    writer.flush()


def test(policy_net, validation_traces, logger, step):
    LOG_DIR = "./backup/"  # stick LOGDIR .
    val_tic = time.time()
    tag_prefix = "Central "
    # except Exception as e:
    #     logger.error("Error when validation! " + str(e))
    # try: # I don't think it make sense,
    if pm.TRAINING_MODE == "SL":
        val_loss = validate.val_loss(policy_net, copy.deepcopy(validation_traces), logger, step)
    jct, makespan, reward = validate.val_jmr(policy_net, copy.deepcopy(validation_traces), logger, step)  # val_jmr有一些
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
    global LOG_DIR
    LOG_DIR = "./backup/"
    logger = log.getLogger(name="central_agent", level=pm.LOG_MODE)
    logger.info("Start central agent...")

    if not pm.RANDOMNESS:
        np.random.seed(pm.np_seed)
        torch.manual_seed(pm.tr_seed)  # specific gpu use:torch.cuda.manual_seed(seed)

    writer = SummaryWriter(pm.SUMMARY_DIR)

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
        logger.info("start validation for heuristics and initialized NN.")  # It could begin working.
        validation_traces = []  # validation traces
        tags_prefix = ["DRF: ", "SRTF: ", "FIFO: ", "Tetris: ", "Optimus: "]
        for i in range(pm.VAL_DATASET):
            validation_traces.append(trace.Trace(None).get_trace())
        # stats = comparison.compare(copy.deepcopy(validation_traces), logger)  # 'compare' method needs ten seconds
        stats = []
        # deep copy to avoid changes to validation_traces
        if not pm.SKIP_FIRST_VAL:
            stats.append(test(policy_net, copy.deepcopy(validation_traces), logger, step=0))  # still have some problem
            tags_prefix.append("Init_NN: ")

        f = open(LOG_DIR + "baselines.txt", 'w')
        for i in range(len(stats)):
            jct, makespan, reward = stats[i]
            value = tags_prefix[i] + " JCT: " + str(jct) + " Makespan: " + str(makespan) + " Reward: " + str(
                reward) + "\n"
            f.write(value)
        f.close()
        logger.info("Finish validation for heuristics and initialized NN.")

        while step <= pm.TOT_NUM_STEPS:
            logger.info("step:" + str(step))
            # send updated parameters to agents
            policy_weights = policy_net.get_weights()
            if pm.VALUE_NET:
                value_weights = value_net.get_weights()
                for i in range(pm.NUM_AGENTS):
                    net_weights_qs[i].put((policy_weights, value_weights))
            else:
                for i in range(pm.NUM_AGENTS):
                    net_weights_qs[i].put(policy_weights)

            # display speed
            if step % pm.DISP_INTERVAL == 0:
                elaps_t = time.time() - start_t
                speed = step / elaps_t
                logger.info("Central agent: Step " + str(
                    step) + " Speed " + '%.3f' % speed + " batches/sec" + " Time " + '%.3f' % elaps_t + " seconds")

            # statistics

            collect_stats(stats_qs, writer, step)
            writer.close()
            exit()
            if not pm.FIX_LEARNING_RATE:
                if step in pm.ADJUST_LR_STEPS:
                    policy_net.lr /= 2
                    if pm.VALUE_NET:
                        value_net.lr /= 2
                    logger.info("Learning rate is decreased to " + str(policy_net.lr) + " at step " + str(step))
            if step < pm.STEP_TRAIN_CRITIC_NET:  # set policy net lr to 0 to train critic net only
                policy_net.lr = 0.0

