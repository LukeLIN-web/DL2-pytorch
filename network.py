import numpy as np
import parameters as pm 
import torch.nn as nn
import torch.nn.functional as F  # 激励函数都在这

class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)

class PolicyNetwork:
    def __init__(self, scope, mode, logger):
        self.scope = scope
        self.mode = mode
        self.logger = logger
        self.net = PNet()
        




class ValueNetwork:
    def __init__(self, scope, mode, logger):
        self.scope = scope
        self.mode = mode
        self.logger = logger
