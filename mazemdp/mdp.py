'''
Author: Olivier Sigaud
'''

import numpy as np
from mazemdp.toolbox import discreteProb


class SimpleActionSpace:  # class describing the action space of the markov decision process
    def __init__(self, action_list=[], nactions=0):
        if len(action_list) == 0:
            self.actions = np.array([a for a in range(nactions)])
        else:
            self.actions = action_list
            
        self.size = len(self.actions)

    def sample(self, prob_list=None):
        # returns an action drawn according to the prob_list distribution,
        # if the param is not set, then it is drawn from a uniform distribution
        if prob_list is None:
            prob_list = np.ones(self.size)/self.size
            
        index = discreteProb(prob_list) 
        return self.actions[index]

    
class Mdp:  # defines a Markov Decision Process

    def __init__(self, nb_states, action_space, start_distribution, transition_matrix,
                  reward_matrix, plotter, gamma=0.9, terminal_states=[], timeout=50):
        assert (timeout > 10), 'timeout too short:' + timeout
        self.nb_states = nb_states
        self.terminal_states = terminal_states
        self.action_space = action_space
        self.current_state = -1  # current position of the agent in the maze, it is set by the reset() method
        self.timeout = timeout  # maximum length of an episode
        self.timestep = 0 
        self.P0 = start_distribution  # distribution used to draw the first state of the agent, used in method reset()
        self.P = transition_matrix
        self.r = reward_matrix
        self.plotter = plotter  # used to plot the maze
        self.gamma = gamma  # discount factor
        self.last_action_achieved = False  # used to tell whether the last state has been reached or not (see done())

    def reset(self, uniform=False):  # initializes an episode and returns the state of the agent
        # if uniform is set to False, the first state is drawn according to the P0 distribution,
        # else it is drawn from a uniform distribution over all the states except for walls
        
        if uniform:
            prob = np.ones(self.nb_states)/self.nb_states
            self.current_state = discreteProb(prob)
        else:
            self.current_state = discreteProb(self.P0)
            
        self.timestep = 0
        self.last_action_achieved = False
        return self.current_state

    def done(self):  # returns True if the episode is over
        if self.last_action_achieved:
            return True
        if self.current_state in self.terminal_states:  # done when a terminal state is reached
            # the terminal states are actually a set of states from which any action leads to an added imaginary state,
            # the "well", with a reward of 1. To know if the episode is over, we have to check
            # whether the agent is on one of these last states and performed the action that gives it its last reward
            self.last_action_achieved = True
        return self.timestep == self.timeout  # done when timeout reached

    def step(self, u, deviation=0):  # performs a step forward in the environment,
        # if you want to add some noise to the reward, give a value to the deviation param 
        # which represents the mean μ of the normal distribution used to draw the noise 
        
        noise = deviation*np.random.randn()  # generate noise, useful for RTDP

        # r is the reward of the transition, you can add some noise to it
        reward = self.r[self.current_state, u] + noise
        
        # the state reached when performing action u from state x is sampled 
        # according to the discrete distribution self.P[x,u,:]
        state = discreteProb(self.P[self.current_state, u, :])

        self.timestep += 1 

        info = {"State transition probabilities": self.P[self.current_state, u, :],
                "reward's noise value": noise}  # can be used when debugging

        self.current_state = state
        done = self.done()  # checks if the episode is over
        
        return [state, reward, done, info]
    
    def new_render(self, title):  # initializes a new environment rendering (a plot defined by a figure, an axis...)
        self.plotter.new_render(title)
    
    def render(self, v=[], policy=[], agent_pos=-1, title='No Title'):  # outputs the agent in the environment with values V (or Q)
        if agent_pos > -1:
            self.plotter.render(agent_state=agent_pos, v=v, title='No Title')
        elif self.current_state > -1:  # and not self.last_action_achieved:
            self.plotter.render(agent_state=self.current_state, v=v, policy=policy, title='No Title')
        else:
            self.plotter.render(v=v, title='No Title')
        
    def save_fig(self, title):  # saves the current output into the disk
        self.plotter.save_fig(title)
