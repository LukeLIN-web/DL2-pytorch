# cluster ini
import numpy as np
import parameters as pm

class Cluster:
    def __init__(self, logger):
        # 0 means available
        self.logger = logger
        # capacity?
        # in one mechine , node == device,

        self.CLUSTER_RESR_CAPS = np.array([pm.CLUSTER_NUM_NODES * pm.NUM_RESR_SLOTS for i in range(pm.NUM_RESR_TYPES)])
        self.NODE_RESR_CAPS = np.array([pm.NUM_RESR_SLOTS for i in range(pm.NUM_RESR_TYPES)])
        self.cluster_state = np.zeros(shape=(pm.NUM_RESR_TYPES, pm.CLUSTER_NUM_NODES * pm.NUM_RESR_SLOTS))
        self.nodes_used_resrs = np.zeros(shape=(pm.CLUSTER_NUM_NODES, pm.NUM_RESR_TYPES))

    def clear(self):
        self.cluster_state.fill(0)
        self.nodes_used_resrs.fill(0)


