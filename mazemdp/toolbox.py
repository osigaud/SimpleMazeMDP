from random import randrange

import numpy as np

# ============ action constants ================
N = 0
S = 1
E = 2
W = 3
NOOP = 4


def discreteProb(p):
    # Draw a random number using probability table p (column vector)
    # Suppose probabilities p=[p(1) ... p(n)] for the values [1:n] are given, sum(p)=1
    # and the components p(j) are nonnegative.
    # To generate a random sample of size m from this distribution,
    # imagine that the interval (0,1) is divided into intervals with the lengths p(1),...,p(n).
    # Generate a uniform number rand, if this number falls in the jth interval given the discrete distribution,
    # return the value j. Repeat m times.
    r = np.random.random()
    cumprob = np.hstack((np.zeros(1), p.cumsum()))
    sample = -1
    for j in range(p.size):
        if (r > cumprob[j]) & (r <= cumprob[j + 1]):
            sample = j
            break
    return sample


def softmax(q, x, tau):
    # Returns a soft-max probability distribution over actions
    # Inputs :
    # - Q : a Q-function represented as a nX times nU matrix
    # - x : the state for which we want the soft-max distribution
    # - tau : temperature parameter of the soft-max distribution
    # Note that tau can be set to 0 because numpy can deal with the division by 0
    # Output :
    # - p : probability of each action according to the soft-max distribution

    p = np.zeros((len(q[x])))
    sump = 0
    for i in range(len(p)):
        p[i] = np.exp((q[x, i] / tau).round(5))
        sump += p[i]

    p = p / sump
    return p


def egreedy(q, x, epsilon):
    # Returns an action following the epsilon-greedy exploration policy
    # Inputs :
    # - Q: a Q-function represented as a nX times nU matrix
    # - x: the state for which we want the action
    # - epsilon: rate of random actions
    # Output :
    # - u : chosen action
    r = np.random
    if r.random() < epsilon:
        return randrange(len(q[x]))
    else:
        # choose randomly about the actions with a max value
        return np.random.choice(np.where(q[x, :] == q[x, :].max())[0])


def egreedy_loc(action, nb, epsilon):
    # Returns an action following the epsilon-greedy exploration policy
    # Inputs :
    # - epsilon: rate of random actions
    # Output:
    # - action: the chosen action
    r = np.random
    if r.random() < epsilon:
        return randrange(nb)
    else:
        return action


def compare(v, q, pol):
    # compares the state value V with the state-action value Q following policy pol
    epsilon = 0.01  # precision of the comparison
    sumval = np.zeros(v.size)
    for i in range(v.size):  # compute the difference between V and Q for each state
        sumval[i] = abs(v[i] - q[i, pol[i]])

    if np.max(sumval) < epsilon:
        return True
    else:
        return False


def random_policy(mdp):
    # Returns a random policy given an mdp
    # Inputs :
    # - mdp : the mdp
    # Output :
    # - pol : the policy

    rand = np.random
    pol = np.zeros(mdp.nb_states, dtype=np.int16)
    for x in range(mdp.nb_states):
        pol[x] = rand.choice(mdp.action_space.actions)
    return pol
