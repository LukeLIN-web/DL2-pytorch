

class Job:
    def __init__(self, id, type, logger=None):
        self.id = id
        self.type = type
        self.logger = logger

        self.num_epochs = None
        self.real_num_epochs = None
        self.progress = 0.0

        self.arrv_time = None
        self.start_time = None  # not tracked
        self.end_time = None

        self.num_workers = 0
        self.num_ps = 0
        self.resr_worker = None
        self.resr_ps = None

        self.model = None
        self.epoch_size = None
        self.local_comp_time = None
        self.model_size = None
        self.inter_bw = None
        self.intra_bw = None

        self.speed_func = None



def main():
    import numpy as np
    id = 1
    type = 1
    job = Job(id, type, None)  # type start from 1
    job.arrv_time = 0
    job.epoch_size = 115
    job.model_size = 102.2
    job.local_comp_time = 0.449
    job.intra_bw = 306.5
    job.inter_bw = 91.875
    job.resr_ps = np.array([3, 0])
    job.resr_worker = np.array([2, 4])
    job.num_epochs = 120
    job.real_num_epochs = 118


if __name__ == '__main__':
    main()
