import numpy as np
import parameters as pm 
import torch.nn as nn
import torch.nn.functional as F  # 激励函数都在这
import torch.optim as optim

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
        self.lr = pm.LEARNING_RATE
        # todo : this is demo network , need carefully designed.
        if pm.OPTIMIZER == "Adam":
            self.optimize =  optim.Adam( self.net.parameters(), lr = 0.0001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        elif pm.OPTIMIZER == "RMSProp":
            self.optimize = optim.RMSprop( self.net.parameters(),lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)



class ValueNetwork:
    def __init__(self, scope, mode, logger):
        self.scope = scope
        self.mode = mode
        self.logger = logger
