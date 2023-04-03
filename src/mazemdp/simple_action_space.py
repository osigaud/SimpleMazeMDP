import numpy as np
from toolbox import sample_categorical


class SimpleActionSpace:  # class describing the action space of the markov decision process
    def __init__(self, action_list=None, nactions=0):
        if action_list is None or len(action_list) == 0:
            self.actions = np.arange(nactions)
        else:
            self.actions = action_list

        self.size = len(self.actions)

    def sample(self, prob_list=None):
        # returns an action drawn according to the prob_list distribution,
        # if the param is not set, then it is drawn from a uniform distribution
        if prob_list is None:
            prob_list = np.ones(self.size) / self.size

        index = sample_categorical(prob_list)
        return self.actions[index]
