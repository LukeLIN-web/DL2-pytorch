import queue as Queue
import time


class DRF_Env(Scheduler):
    # overwrite the scheduling algorithm in Scheduler
    def _schedule(self):
        tic = time.time()
        drf_queue = Queue.PriorityQueue()
        for job in self.uncompleted_jobs:
            drf_queue.put((0, job.arrv_time, job))  # enqueue jobs into drf queue



def test():
    import log, trace
    np.random.seed(9973)
    logger = log.getLogger(name="test.log", level="DEBUG")
    job_trace = trace.Trace(logger).get_trace()
    env = DRF_Env("DRF", job_trace, logger)
    while not env.end:
        env.step()
    print(env.observe())
    print(env.data)
    input()
    print(env.get_results())
    print(env.get_job_jcts())
    for i in range(len(env.trace)):
        if i in env.trace:
            for job in env.trace[i]:
                print(i + 1, job.id, job.type, job.model)


if __name__ == '__main__':
    test()
