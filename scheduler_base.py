import log
import parameters as pm


class Scheduler(object):
    def __init__(self, name, trace, logger):
        self.cluster = Cluster(self.logger)
        self.name = name  # e.g., 'DRF'
        self.trace = trace
        if logger is None:
            assert name
            self.logger = log.getLogger(name=name, fh=False)
        else:
            self.logger = logger

        self.curr_ts = 0
        self.end = False

    def step(self):
        # step by one timeslot
        assert not self.end
        self._prepare()
        self._schedule()
        self._progress()
        if len(self.completed_jobs) == pm.TOT_NUM_JOBS:
            self.end = True
        self.curr_ts += 1
        return self.data

    def _prepare(self):
        self.cluster.clear()
        self.data = []
        self.running_jobs.clear()
        if self.curr_ts in self.trace:
            for job in self.trace[self.curr_ts]:
                job.reset()  # must reset since it is trained for multiple epochs
                self.uncompleted_jobs.add(job)
                self.logger.debug(job.info())
        for job in self.uncompleted_jobs:
            job.num_workers = 0
            job.curr_worker_placement = []
            if pm.PS_WORKER:
                job.num_ps = 0
                job.curr_ps_placement = []



def test():
    env = fifo_env.FIFO_Env("FIFO", job_trace, None)
    while not env.end:
        env.step()





if __name__ == '__main__':
    test()
