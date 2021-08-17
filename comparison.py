import multiprocessing
import time
import trace
from typing import List, Tuple
import drf_env
import srtf_env
import fifo_env

def drf(job_trace=None):
    if job_trace is None:
        job_trace = trace.Trace(None).get_trace()
    env = drf_env.DRF_Env("DRF", job_trace, None)
    while not env.end:
        env.step()
    return [env.get_results(), env.get_job_jcts().values()]


def srtf(job_trace=None):
    if job_trace is None:
        job_trace = trace.Trace(None).get_trace()
    env = srtf_env.SRTF_Env("SRTF", job_trace, None)
    while not env.end:
        env.step()
    return [env.get_results(), env.get_job_jcts().values()]


def fifo(job_trace=None):
    if job_trace is None:
        job_trace = trace.Trace(None).get_trace()
    env = fifo_env.FIFO_Env("FIFO", job_trace, None)
    while not env.end:
        env.step()
    return [env.get_results(), env.get_job_jcts().values()]



def compare(traces, logger, debug=False) -> List[Tuple]:
    if debug:
        drf(traces[0])
        srtf(traces[0])
        fifo(traces[0])
    f = open("DRF_JCTs.txt", 'w')
    f.close()

    num_schedulers = 3
    thread_list = [[] for i in range(num_schedulers)]  # a two dimension matrix
    tic = time.time()
    pool = multiprocessing.Pool(processes=40)
    for i in range(len(traces)):  # one example takes about 10s
        thread_list[0].append(pool.apply_async(drf, args=(traces[i],)))
        thread_list[1].append(pool.apply_async(srtf, args=(traces[i],)))
        thread_list[2].append(pool.apply_async(fifo, args=(traces[i],)))
    pool.close()
    pool.join()





def main():
    logger = log.getLogger(name="comparison", level="INFO")
    num_traces = 10
    traces = []
    for i in range(num_traces):
        job_trace = trace.Trace(None).get_trace()
        traces.append(job_trace)
    compare(traces, logger, False)


if __name__ == '__main__':
    main()

'''
comparison.py:74 INFO: Average      JCT: DRF 5.900 SRTF 8.132 FIFO 8.203 Tetris 9.606
comparison.py:78 INFO: Average Makespan: DRF 29.207 SRTF 36.991 FIFO 37.221 Tetris 36.204
comparison.py:82 INFO: Average   Reward: DRF 2.063 SRTF 1.633 FIFO 1.623 Tetris 1.668
'''
