# coding=utf-8
import copy
import torch
import comparison
import drf_env
import fifo_env
import memory
import optimus_env
import srtf_env
import tetris_env
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


def test(policy_net, validation_traces, logger, step, writer):
    LOG_DIR = "./backup/"  # stick LOGDIR .
    val_tic = time.time()
    tag_prefix = "Central "
    # except Exception as e:
    #     logger.error("Error when validation! " + str(e))
    # try: # I don't think it make sense, collecting exception makes debug difficult
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

    # writer = SummaryWriter(pm.SUMMARY_DIR)

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
            # collect_stats(stats_qs, writer, step)
            # writer.close()

            if not pm.FIX_LEARNING_RATE:
                if step in pm.ADJUST_LR_STEPS:
                    policy_net.lr /= 2
                    if pm.VALUE_NET:
                        value_net.lr /= 2
                    logger.info("Learning rate is decreased to " + str(policy_net.lr) + " at step " + str(step))
            if step < pm.STEP_TRAIN_CRITIC_NET:  # set policy net lr to 0 to train critic net only
                policy_net.lr = 0.0

            # if step % pm.DISP_INTERVAL == 0:
            # writer.add_scalar(tag="Learning rate", scalar_value=policy_net.lr, global_step=step)

            # save model
            if step % pm.CHECKPOINT_INTERVAL == 0:
                name_prefix = ""
                if pm.TRAINING_MODE == "SL":
                    name_prefix += "sl_"
                else:
                    name_prefix += "rl_"
                if pm.PS_WORKER:
                    name_prefix += "ps_worker_"
                else:
                    name_prefix += "worker_"

                model_name = pm.MODEL_DIR + "policy_" + name_prefix + str(step) + ".ckpt"
                torch.save({'state_dict': policy_net.net.state_dict(),
                            'optimizer': policy_net.net.optimizer.state_dict()},
                           model_name)
                logger.info("Policy model saved: " + model_name)
                if pm.VALUE_NET and pm.SAVE_VALUE_MODEL:
                    model_name = pm.MODEL_DIR + "value_" + name_prefix + str(step) + ".ckpt"
                    torch.save({'state_dict': policy_net.net.state_dict(),
                                'optimizer': policy_net.net.optimizer.state_dict()},
                               model_name)
                    logger.info("Value model saved: " + model_name)

            # validation
            # if pm.VAL_ON_MASTER and step % pm.VAL_INTERVAL == 0:
            #     test(policy_net, copy.deepcopy(validation_traces), logger, step, writer)

            # poll and update parameters
            poll_ids = set([i for i in range(pm.NUM_AGENTS)])
            avg_policy_grads = []
            avg_value_grads = []
            while True:
                for i in poll_ids.copy():
                    try:
                        if pm.VALUE_NET:
                            policy_gradients, value_gradients = net_gradients_qs[i].get(False)
                        else:
                            policy_gradients = net_gradients_qs[i].get(False)  # why this is "False"?
                        poll_ids.remove(i)
                        if len(avg_policy_grads) == 0:
                            avg_policy_grads = policy_gradients
                        else:
                            for j in range(len(avg_policy_grads)):
                                avg_policy_grads[j] += policy_gradients[j]
                        if pm.VALUE_NET:
                            if len(avg_value_grads) == 0:
                                avg_value_grads = value_gradients
                            else:
                                for j in range(len(avg_value_grads)):
                                    avg_value_grads[j] += value_gradients[j]
                    except:
                        continue
                if len(poll_ids) == 0:
                    break
            for i in range(0, len(avg_policy_grads)):
                avg_policy_grads[i] = avg_policy_grads[i] / pm.NUM_AGENTS
            policy_net.apply_gradients(avg_policy_grads)

            if pm.VALUE_NET:
                for i in range(0, len(avg_value_grads)):
                    avg_value_grads[i] = avg_value_grads[i] / pm.NUM_AGENTS
                value_net.apply_gradients(avg_value_grads)

            # visualize gradients and weights
            if step % pm.VISUAL_GW_INTERVAL == 0 and pm.EXPERIMENT_NAME is None:
                assert len(policy_weights) == len(avg_policy_grads)
                for i in range(0, len(policy_weights), 10):
                    tb_logger.add_histogram(tag="Policy weights " + str(i), value=policy_weights[i], step=step)
                    tb_logger.add_histogram(tag="Policy gradients " + str(i), value=avg_policy_grads[i], step=step)
                if pm.VALUE_NET:
                    assert len(value_weights) == len(avg_value_grads)
                    for i in range(0, len(value_weights), 10):
                        tb_logger.add_histogram(tag="Value weights " + str(i), value=value_weights[i], step=step)
                        tb_logger.add_histogram(tag="Value gradients " + str(i), value=avg_value_grads[i], step=step)
            step += 1

        logger.info("Training ends...")
        if pm.VALUE_NET:
            for i in range(pm.NUM_AGENTS):
                net_weights_qs[i].put(("exit", "exit"))
        else:
            for i in range(pm.NUM_AGENTS):
                net_weights_qs[i].put("exit")
        # os.system("sudo pkill -9 python")
        exit(0)


# supervised learning
def sl_agent(net_weights_q, net_gradients_q, stats_q, id):
    logger = log.getLogger(name="agent_" + str(id), level=pm.LOG_MODE)
    logger.info("Start supervised learning, agent " + str(id) + " ...")

    if not pm.RANDOMNESS:
        np.random.seed(pm.np_seed + id + 1)

    # invoke network
    policy_net = network.PolicyNetwork("policy_net", pm.TRAINING_MODE, logger)

    global_step = 1
    avg_jct = []
    avg_makespan = []
    avg_reward = []
    if not pm.VAL_ON_MASTER:
        validation_traces = []  # validation traces
        for i in range(pm.VAL_DATASET):
            validation_traces.append(trace.Trace(None).get_trace())
    # generate training traces
    traces = []
    for episode in range(pm.TRAIN_EPOCH_SIZE):
        job_trace = trace.Trace(None).get_trace()
        traces.append(job_trace)
    mem_store = memory.Memory(maxlen=pm.REPLAY_MEMORY_SIZE)
    logger.info("Filling experience buffer...")
    for epoch in range(pm.TOT_TRAIN_EPOCHS):
        logger.info("epoch:" + str(epoch))
        for episode in range(pm.TRAIN_EPOCH_SIZE):
            tic = time.time()
            job_trace = copy.deepcopy(traces[episode])
            if pm.HEURISTIC == "DRF":
                env = drf_env.DRF_Env("DRF", job_trace, logger)
            elif pm.HEURISTIC == "FIFO":
                env = fifo_env.FIFO_Env("FIFO", job_trace, logger)
            elif pm.HEURISTIC == "SRTF":
                env = srtf_env.SRTF_Env("SRTF", job_trace, logger)
            elif pm.HEURISTIC == "Tetris":
                env = tetris_env.Tetris_Env("Tetris", job_trace, logger)
            elif pm.HEURISTIC == "Optimus":
                env = optimus_env.Optimus_Env("Optimus", job_trace, logger)

            while not env.end:
                if pm.LOG_MODE == "DEBUG":
                    time.sleep(0.01)
                data = env.step()
                logger.info("ts length:" + str(len(data)))
                logger.info("len(self.completed_jobs)" + str(len(env.completed_jobs)))

                for (input, label) in data:
                    mem_store.store(input, 0, label, 0)
                # store in mem,  use random SGD, take sample from mem_store, then begin superversed learning to calculate gradients
                logger.info("len(self.memory)" + str(len(mem_store.memory)))
                if mem_store.full():
                    # prepare a training batch
                    _, trajectories, _ = mem_store.sample(pm.MINI_BATCH_SIZE)
                    input_batch = [traj.state for traj in trajectories]
                    label_batch = [traj.action for traj in trajectories]
                    logger.info("prepared a training batch")

                    # pull latest weights before training
                    # weights = net_weights_q.get()
                    # if isinstance(weights, str) and weights == "exit":
                    #     logger.info("Agent " + str(id) + " exits.")
                    #     exit(0)
                    # policy_net.set_weights(weights)

                    # supervised learning to calculate gradients
                    logger.info("supervised learning to calculate gradients")
                    entropy, loss, policy_grads = policy_net.get_sl_gradients(np.stack(input_batch),
                                                                              np.vstack(label_batch))
                    logger.info("len(env.completed_jobs):" + str(len(env.completed_jobs)) + "loss :" + str(loss))
                    for i in range(len(policy_grads)):
                        assert np.any(np.isnan(policy_grads[i])) is False

                    # send gradients to the central agent
                    # net_gradients_q.put(policy_grads)

                    # validation
                    if not pm.VAL_ON_MASTER and global_step % pm.VAL_INTERVAL == 0:
                        val_tic = time.time()
                        val_loss = validate.val_loss(policy_net, validation_traces, logger, global_step)
                        jct, makespan, reward = validate.val_jmr(policy_net, validation_traces, logger, global_step)
                        stats_q.put(("val", val_loss, jct, makespan, reward))
                        val_toc = time.time()
                        logger.info(
                            "Agent " + str(id) + " Validation at step " + str(global_step) + " Time: " + '%.3f' % (
                                    val_toc - val_tic))
                    stats_q.put(("step:sl", entropy, loss))

                    global_step += 1

            num_jobs, jct, makespan, reward = env.get_results()
            avg_jct.append(jct)
            avg_makespan.append(makespan)
            avg_reward.append(reward)
            if global_step % pm.DISP_INTERVAL == 0:
                logger.info("Agent\t AVG JCT\t Makespan\t Reward")
                logger.info(str(id) + " \t \t " + '%.3f' % (sum(avg_jct) / len(avg_jct)) + " \t\t" + " " + '%.3f' % (
                        1.0 * sum(avg_makespan) / len(avg_makespan)) \
                            + " \t" + " " + '%.3f' % (sum(avg_reward) / len(avg_reward)))
