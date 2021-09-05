import numpy as np
import collections

'''
implement prioritized experience replay later
sample transitions with probability relative to the last encountered absolute TD error
'''


class Memory:
    def __init__(self, maxlen):
        self.maxLen = maxlen
        self.memory = collections.deque(maxlen=maxlen)
        self.Transition = collections.namedtuple('Transition', ('state', 'output', 'action', 'reward'))

    def store(self, state, output, action, reward):
        self.memory.append(self.Transition(state, output, action, reward))

    def sample(self, batch_size):
        indexes = np.random.choice(self.maxLen, size=batch_size, replace=False)
        experience_batch = [self.memory[index] for index in indexes]
        is_weights = [1 for _ in range(batch_size)]
        return indexes, experience_batch, is_weights

    def full(self):
        if len(self.memory) == self.maxLen:
            return True
        else:
            return False

    def avg_reward(self):
        assert len(self.memory) > 0
        sum_reward = 0
        for i in range(len(self.memory)):
            transition = self.memory[i]
            sum_reward += transition.reward
        return sum_reward / len(self.memory)
