import numpy as np
import torch.nn

import parameters as pm
import torch.nn as nn
import torch.nn.functional as F  # 激励函数都在这
import torch.optim as optim


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        # type, arrival, progress, resource
        dense_net_1 = nn.Linear(pm.STATE_DIM[1], pm.NUM_NEURONS_PER_FCN)
        self.linears = nn.ModuleList()
        self.linears.append(dense_net_1)
        self.linears.append(nn.ReLU())
        for i in range(1, pm.NUM_FCN_LAYERS):
            self.linears.append(nn.Linear(pm.NUM_NEURONS_PER_FCN, pm.NUM_NEURONS_PER_FCN))
            self.linears.append(nn.ReLU())
        output = nn.Linear(pm.NUM_NEURONS_PER_FCN, pm.ACTION_DIM)
        self.linears.append(output)

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = l(x)
        x = F.softmax(x)
        return x


class PolicyNetwork:
    def __init__(self, scope, mode, logger):
        self.scope = scope
        self.mode = mode
        self.logger = logger
        self.net = PNet()
        self.lr = pm.LEARNING_RATE
        # todo : this is demo network , need carefully designed.
        if pm.OPTIMIZER == "Adam":
            self.optimize = optim.Adam(self.net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                       amsgrad=False)
        elif pm.OPTIMIZER == "RMSProp":
            self.optimize = optim.RMSprop(self.net.parameters(), lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0,
                                          momentum=0, centered=False)
        if self.mode == "SL":
            if pm.SL_LOSS_FUNCTION == "Mean_Square":
                self.criterion = nn.MSELoss()
            elif pm.SL_LOSS_FUNCTION == "Cross_Entropy":
                self.criterion = nn.CrossEntropyLoss()
            elif pm.SL_LOSS_FUNCTION == "Absolute_Difference":
                self.criterion = nn.L1Loss()

    def predict(self, inputs):
        output = self.net(inputs)
        return output

    def get_sl_loss(self, inputs: torch.Tensor, label):
        assert self.mode == "SL"
        return self.net(inputs.float()), self.criterion(self.net(inputs.float()), label.long())


class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)


class ValueNetwork:
    def __init__(self, scope, mode, logger):
        self.scope = scope
        self.mode = mode
        self.logger = logger
        self.net = PNet()
        self.lr = pm.LEARNING_RATE
        # todo : this is demo network , need carefully designed.
        if pm.OPTIMIZER == "Adam":
            self.optimize = optim.Adam(self.net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                       amsgrad=False)
        elif pm.OPTIMIZER == "RMSProp":
            self.optimize = optim.RMSprop(self.net.parameters(), lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0,
                                          momentum=0, centered=False)
