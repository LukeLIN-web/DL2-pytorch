import numpy as np
import parameters as pm
import torch.nn as nn
import torch.nn.functional as F  # 激励函数都在这
import torch.optim as optim


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        # type, arrival, progress, resource
        input = tflearn.input_data(shape=[None, self.state_dim[0], self.state_dim[1]],
                                   name="input")  # row is info type, column is job
        if pm.JOB_CENTRAL_REPRESENTATION or pm.ATTRIBUTE_CENTRAL_REPRESENTATION:
            if pm.JOB_CENTRAL_REPRESENTATION:
                fc_list = []
                for i in range(self.state_dim[1]):
                    if pm.FIRST_LAYER_TANH:
                        self.fc1 = nn.Linear(input[:, :, i], self.state_dim[0]) # , activation="tanh",  name="job_" + str(i))
                    else:
                        self.fc1 = nn.Linear(input[:, :, i], self.state_dim[0])
                        # , activation="relu",name="job_" + str(i))
                    if pm.BATCH_NORMALIZATION:
                        bn = nn.BatchNorm1d(256)
                    fc_list.append(bn)
            else:
                j = 0
                fc_list = []
                for (key,enable) in pm.INPUTS_GATE:  # INPUTS_GATE=[("TYPE",True), ("STAY",False), ("PROGRESS",False), ("DOM_RESR",False), ("WORKERS",True)]
                    if enable:
                        if pm.FIRST_LAYER_TANH:
                            self.fc1 = nn.Linear(input[:, j], pm.SCHED_WINDOW_SIZE) # activation="tanh",name=key)
                        else:
                            self.fc1 = nn.Linear(input[:, j], pm.SCHED_WINDOW_SIZE)# activation="relu",name=key)
                        if pm.BATCH_NORMALIZATION:
                            bn =  nn.BatchNorm1d(256)
                        fc_list.append(fc1)
                        j += 1
            if len(fc_list) == 1:
                merge_net = fc_list[0]
                if pm.BATCH_NORMALIZATION:
                    merge_net =  nn.BatchNorm1d(256)
             else:
                    merge_net = tflearn.merge(fc_list, 'concat', name="merge_net_1")
                        if pm.BATCH_NORMALIZATION:
                            merge_net = nn.BatchNorm1d(256)
                        dense_net_1 = tflearn.fully_connected(merge_net,
                                                              # pm.NUM_NEURONS_PER_FCN, activation='relu',name='dense_net_1')
                        else:
                        dense_net_1 = tflearn.fully_connected(input, pm.NUM_NEURONS_PER_FCN, activation='relu',
                                                              name='dense_net_1')
                        if pm.BATCH_NORMALIZATION:
                            dense_net_1 = tflearn.batch_normalization(dense_net_1, name='dense_net_1_bn')

                        for i in range(1, pm.NUM_FCN_LAYERS):
                            dense_net_1 = tflearn.fully_connected(dense_net_1, pm.NUM_NEURONS_PER_FCN,
                                                                  activation='relu',
                                                                  name='dense_net_' + str(i + 1))
                        if pm.BATCH_NORMALIZATION:
                            dense_net_1 = tflearn.batch_normalization(dense_net_1,
                                                                      name='dense_net_' + str(i + 1) + 'bn')

                        if pm.JOB_CENTRAL_REPRESENTATION and pm.NN_SHORTCUT_CONN:  # add shortcut the last layer
                            fc2_list = []
                        for fc in fc_list:
                            merge_net_2 = tflearn.merge([fc, dense_net_1], 'concat')
                        if pm.PS_WORKER:
                            if pm.BUNDLE_ACTION:
                        fc2 = tflearn.fully_connected(merge_net_2, 3, activation='linear')
                        else:
                        fc2 = tflearn.fully_connected(merge_net_2, 2, activation='linear')
                        else:
                        fc2 = tflearn.fully_connected(merge_net_2, 1, activation='linear')
                        fc2_list.append(fc2)

                        if pm.SKIP_TS:
                            fc2 = tflearn.fully_connected(dense_net_1, 1, activation='linear')
                        fc2_list.append(fc2)
                        merge_net_3 = tflearn.merge(fc2_list, 'concat')
                        # output = tflearn.activation(merge_net_3, activation="softmax", name="policy_output")
                    nn.Softmax()
                        else:

                        output = tflearn.fully_connected(dense_net_1, self.action_dim, activation="softmax",
                                                         name="policy_output")
        return input, output


    def forward(self, x):
        x = F.relu(self.fc1(x))

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

    def get_sl_loss(self, inputs, label):
        assert self.mode == "SL"
        return self.criterion(self.net(inputs), label)


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
