import queue as Queue
import time

import numpy as np

import parameters as pm
from scheduler_base import Scheduler


class FIFO_Env(Scheduler):
    def _schedule(self):
        tic = time.time()

        fifo_queue = Queue.PriorityQueue()
        count = 0
        for job in self.uncompleted_jobs:
            count += 1
            fifo_queue.put((job.arrv_time, count, job))  # enqueue jobs into fifo queue

        flag = False
        while not fifo_queue.empty():
            (_, _, job) = fifo_queue.get()
            # allocate maximal number of workers
            # bundle one ps and one worker together by default
            for i in range(pm.MAX_NUM_WORKERS):
                _, node = self.node_used_resr_queue.get()
                if pm.PS_WORKER:
                    resr_reqs = job.resr_worker + job.resr_ps  # bind a worker with a ps
                else:
                    resr_reqs = job.resr_worker
                (succ, node_used_resrs) = self.cluster.alloc(resr_reqs, node)
                self.node_used_resr_queue.put((np.sum(node_used_resrs), node))
                if succ:
                    if False and pm.PS_WORKER and pm.BUNDLE_ACTION:
                        self._state(job.id, "bundle")
                        job.num_workers += 1
                        job.curr_worker_placement.append(node)
                        job.num_ps += 1
                        job.curr_ps_placement.append(node)
                        job.dom_share = np.max(1.0 * (job.num_workers * job.resr_worker + job.num_ps * job.resr_ps)
                                               / self.cluster.CLUSTER_RESR_CAPS)
                    else:
                        self._state(job.id, "worker")
                        job.num_workers += 1
                        job.curr_worker_placement.append(node)
                        job.dom_share = np.max(1.0 * (job.num_workers * job.resr_worker + job.num_ps * job.resr_ps)
                                               / self.cluster.CLUSTER_RESR_CAPS)

                        if pm.PS_WORKER:
                            self._state(job.id, "ps")
                            job.num_ps += 1
                            job.curr_ps_placement.append(node)
                            job.dom_share = np.max(1.0 * (job.num_workers * job.resr_worker + job.num_ps * job.resr_ps)
                                                   / self.cluster.CLUSTER_RESR_CAPS)

                    self.running_jobs.add(job)
                else:  # fail to alloc resources, continue to try other job
                    flag = True
                    break
            if flag:
                break

        toc = time.time()
        self.logger.debug(self.name + ":: " + "scheduling time: " + "%.3f" % (toc - tic) + " seconds.")
        for job in self.uncompleted_jobs:
            self.logger.debug(self.name + ":: scheduling results" + " num_worker: " + str(job.num_workers))


def test():
    import log, trace
    logger = log.getLogger(name="test.log", level="DEBUG")
    job_trace = trace.Trace(logger).get_trace()
    env = FIFO_Env("FIFO", job_trace, logger)
    while not env.end:
        env.step()
    print("   completed job num  ,  average jct ,  makespan  ,  average reward ")
    print(env.get_results())


if __name__ == '__main__':
    test()
