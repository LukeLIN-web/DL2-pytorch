

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
