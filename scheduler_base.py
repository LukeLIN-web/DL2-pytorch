import trace
import log
import parameters as pm
from cluster import Cluster


class Scheduler(object):
    def __init__(self, name, trace, logger):
        self.name = name  # e.g., 'DRF'
        self.trace = trace

        if logger is None:
            assert name
            self.logger = log.getLogger(name=name, fh=False)
        else:
            self.logger = logger
        self.cluster = Cluster(self.logger)

        self.running_jobs = set()
        self.uncompleted_jobs = set()
        self.completed_jobs = set()

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
                print(job.arrv_time)
                self.logger.debug(job.info())
        for job in self.uncompleted_jobs:
            job.num_workers = 0
            job.curr_worker_placement = []
            if pm.PS_WORKER:
                job.num_ps = 0
                job.curr_ps_placement = []

    def _schedule(self):
        self.logger.info("This method is to be implemented on child class!")

    def _progress(self):
        reward = 0
        for job in self.running_jobs.copy():
            epoch = job.step()
            reward += epoch / job.num_epochs
            if job.progress >= job.real_num_epochs:
                if pm.FINE_GRAIN_JCT:
                    job.end_time = self.curr_ts - 1 + job.get_run_time_in_ts()
                else:
                    job.end_time = self.curr_ts
                # self.running_jobs.remove(job)
                self.uncompleted_jobs.remove(job)
                self.completed_jobs.add(job)
            if pm.NUM_UNCOMPLETED_JOB_REWARD:
                reward = len(self.uncompleted_jobs)
            self.rewards.append(reward)


    def get_results(self):
        # get final results, including avg jct, makespan and avg reward
        jct_list = [(job.end_time - job.arrv_time + 1.0) for job in self.completed_jobs]
        makespan = max([job.end_time + 1.0 for job in self.completed_jobs])
        assert jct_list
        return (
            len(self.completed_jobs), 1.0 * sum(jct_list) / len(jct_list), makespan,
            sum(self.rewards) / len(self.rewards))

