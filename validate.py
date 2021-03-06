# coding=utf-8
import numpy as np
import time
import parameters as pm
import drf_env
import fifo_env
import tetris_env
import srtf_env
import optimus_env
import rl_env
import torch


def val_loss(net, val_traces, logger, global_step) -> float:
    avg_loss = 0
    step = 0
    data = []
    for episode in range(len(val_traces)):
        job_trace = val_traces[episode]
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

        ts = 0
        while not env.end:
            data += env.step()
            ts += 1
            if len(data) >= pm.MINI_BATCH_SIZE:
                # prepare a validation batch
                indexes = np.random.choice(len(data), size=pm.MINI_BATCH_SIZE, replace=False)
                inputs = []
                labels = []
                for index in indexes:
                    input, label = data[index]
                    inputs.append(input)
                    labels.append(label)
                # supervised learning to calculate gradients
                output, loss = net.get_sl_loss(torch.from_numpy(np.stack(inputs)),
                                               torch.from_numpy(np.vstack(labels)))
                avg_loss += loss
                # if step % 50 == 0:
                    # # type, # of time slots in the system so far, normalized remaining epoch, dom resource
                    # print(str(episode) + "_" + str(ts) + "input:" + " type: " + str(input[0]) + " stay_ts: " + str(
                    #     input[1]) + " rt: " + str(input[2]) + " resr:" + str(input[3]) +
                    #       "\n" + " label: " + str(label) + "\n" + " output: " + str(output[-1]))
                step += 1
                data = []

    return avg_loss / step


# return jct, makespan, reward
def val_jmr(net, val_traces, logger, global_step) -> (float, float, float):
    avg_jct = []
    avg_makespan = []
    avg_reward = []
    step = 0.0
    tic = time.time()
    stats = dict()
    stats["step"] = global_step
    stats["jcts"] = []
    states_dict = dict()
    states_dict["step"] = global_step
    states_dict["states"] = []
    for episode in range(len(val_traces)):
        job_trace = val_traces[episode]
        env = rl_env.RL_Env("RL", job_trace, logger, False)
        ts = 0
        while not env.end:
            inputs = env.observe()
            output = net.predict(torch.from_numpy(np.reshape(inputs, (1, pm.STATE_DIM[0], pm.STATE_DIM[1]))))
            print(output.shape)
            masked_output, action, reward, move_on, valid_state = env.step(output)
            if episode == 0 and move_on:  # record the first trace
                states = env.get_sched_states()
                states_dict["states"].append(states)
                '''
                job id: type: num_workers:
                '''
                string = "ts: " + str(ts) + " "
                for id, type, num_workers, num_ps in states:
                    if pm.PS_WORKER:
                        string += "(id: " + str(id) + " type: " + str(type) + " num_workers: " + str(
                            num_workers) + " num_ps: " + str(num_ps) + ") \n"
                    else:
                        string += "(id: " + str(id) + " type: " + str(type) + " num_workers: " + str(
                            num_workers) + ") \n"
                ts += 1

            if episode == 0:
                if step % 50 == 0:
                    i = 0
                    value = "input:"
                    for (key, enabled) in pm.INPUTS_GATE:
                        if enabled:
                            # [("TYPE",True), ("STAY",False), ("PROGRESS",False), ("DOM_RESR",True), ("WORKERS",False)]
                            if key == "TYPE":
                                value += " type: " + str(inputs[i]) + "\n\n"
                            elif key == "STAY":
                                value += " stay_ts: " + str(inputs[i]) + "\n\n"
                            elif key == "PROGRESS":
                                value += " rt: " + str(inputs[i]) + "\n\n"
                            elif key == "DOM_RESR":
                                value += " resr: " + str(inputs[i]) + "\n\n"
                            elif key == "WORKERS":
                                value += " workers: " + str(inputs[i]) + "\n\n"
                            elif key == "PS":
                                value += " ps: " + str(inputs[i]) + "\n\n"
                            i += 1
                    value += " output: " + str(output) + "\n\n" + " masked_output: " + str(
                        masked_output) + "\n\n" + " action: " + str(action)

            step += 1
        num_jobs, jct, makespan, reward = env.get_results()
        stats["jcts"].append(env.get_job_jcts().values())
        avg_jct.append(jct)
        avg_makespan.append(makespan)
        avg_reward.append(reward)
    elapsed_t = time.time() - tic
    logger.info("time for making one decision: " + str(elapsed_t / step) + " seconds")
    with open("DL2_JCTs.txt", 'a') as f:
        f.write(str(stats) + '\n')
    with open("DL2_states.txt", 'a') as f:
        f.write(str(states_dict) + "\n")

    return 1.0 * sum(avg_jct) / len(avg_jct), 1.0 * sum(avg_makespan) / len(avg_makespan), \
           sum(avg_reward) / len(avg_reward)
