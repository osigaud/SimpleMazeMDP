"""
Author: Olivier Sigaud
"""

from functools import cached_property
import numpy as np
import random

from mazemdp.toolbox import sample_categorical


class Mdp:
    """
    defines a Markov Decision Process
    """

    def __init__(
        self,
        nb_states,
        nb_actions,
        start_distribution,
        transition_matrix,
        reward_matrix,
        plotter,
        gamma=0.9,
        terminal_states=None,
        timeout=50,
        has_state=True,
    ):
        assert timeout > 10, "timeout too short:" + timeout
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        if terminal_states is None:
            terminal_states = []
        self.terminal_states = terminal_states
        self.has_state = has_state
        self.timeout = timeout  # maximum length of an episode
        self.timestep = 0
        self.P0 = start_distribution  # distribution used to draw the first state of the agent, used in method reset()
        self.P = transition_matrix
        self.r = reward_matrix
        self.plotter = plotter  # used to plot the maze
        self.gamma = gamma  # discount factor
        self.current_state = None

    @cached_property
    def action_space(self):
        """Legacy method to get the action space"""
        import gym

        return gym.spaces.Discrete(self.nb_actions)

    def reset(
        self, uniform=False, **kwargs
    ):  # initializes an episode and returns the state of the agent
        # if uniform is set to False, the first state is drawn according to the P0 distribution,
        # else it is drawn from a uniform distribution over all the states except for walls

        if uniform:
            prob = np.ones(self.nb_states) / (self.nb_states)
            self.current_state = sample_categorical(prob)
        else:
            self.current_state = sample_categorical(self.P0)

        self.timestep = 0
        self.last_action_achieved = False
        return self.current_state

    def terminated(self):  # returns True if a terminal state was reached
        return self.current_state in self.terminal_states

    def truncated(self):
        return self.timestep == self.timeout  # truncated when timeout reached

    def step(self, u, deviation=0):  # performs a step forward in the environment,
        # if you want to add some noise to the reward, give a value to the deviation param
        # which represents the mean μ of the normal distribution used to draw the noise

        noise = deviation * np.random.randn()  # generate noise, useful for RTDP

        # r is the reward of the transition, you can add some noise to it
        reward = self.r[self.current_state, u] + noise

        # the state reached when performing action u from state x is sampled
        # according to the discrete distribution self.P[x,u,:]
        next_state = sample_categorical(self.P[self.current_state, u, :])

        self.timestep += 1

        info = {
            "State transition probabilities": self.P[self.current_state, u, :],
            "reward's noise value": noise,
        }  # can be used when debugging

        terminated = self.terminated()  # checks if the episode is over
        truncated = self.truncated()  # checks if timit limit was reached
        self.current_state = next_state

        return next_state, reward, terminated, truncated, info

    def sample_transition(self):
        state = random.randint(0, self.nb_states - 1)
        action = random.randint(0, self.nb_actions - 1)
        next_state = sample_categorical(self.P[state, action, :])
        return state, action, next_state

    def new_render(
        self, title, mode="human"
    ):  # initializes a new environment rendering (a plot defined by a figure, an axis...)
        return self.plotter.new_render(title, mode=mode)

    def render(
        self, v=None, policy=None, agent_pos=None, title="No Title", mode="legacy"
    ):  # outputs the agent in the environment with values V (or Q)
        if v is None:
            v = np.array([])

        if policy is None:
            policy = np.array([])

        if not self.has_state:
            return self.plotter.render(v=v, agent_state=None, title=title, mode=mode)
        elif agent_pos is not None:
            return self.plotter.render(
                v=v, agent_state=agent_pos, title=title, mode=mode
            )
        elif self.current_state is not None:
            return self.plotter.render(
                v=v,
                agent_state=self.current_state,
                policy=policy,
                title=title,
                mode=mode,
            )
        else:
            return self.plotter.render(v=v, title=title, mode=mode)

        assert False, "Should not happen"

    def save_fig(self, title):  # saves the current output into the disk
        self.plotter.save_fig(title)

    def check_navigability(self):
        v = np.zeros(self.nb_states)  # initial state values are set to 0
        stop = False

        while not stop:
            v_old = v.copy()

            for x in range(self.nb_states):  # for each state x
                # Compute the value of state x for each action u of the MDP action space
                if x in self.terminal_states:
                    v[x] = np.max(self.r[x, :])
                else:
                    v_temp = []
                    for u in range(self.nb_actions):
                        # Process sum of the values of the neighbouring states
                        summ = 0
                        for y in range(self.nb_states):
                            summ = summ + self.P[x, u, y] * v_old[y]
                        v_temp.append(self.r[x, u] + summ)

                        # Select the highest state value among those computed
                        v[x] = np.max(v_temp)

                        # Test if convergence has been reached
            if (np.linalg.norm(v - v_old)) < 0.01:
                stop = True

        # We should reach terminal states from any starting point
        reachable = self.nb_states - np.count_nonzero(v) == 0
        return reachable
