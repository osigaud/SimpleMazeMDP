'''
Author: Olivier Sigaud
'''

import numpy as np
from mazemdp.toolbox import N, S, E, W
from mazemdp.maze_plotter import MazePlotter  # used to plot the maze
from mazemdp.mdp import SimpleActionSpace, Mdp


def check_navigability(mdp):
    v = np.zeros(mdp.nb_states)  # initial state values are set to 0
    stop = False

    while not stop:
        v_old = v.copy()
 
        for x in range(mdp.nb_states):  # for each state x
            # Compute the value of the state x for each action u of the MDP action space
            v_temp = []
            for u in mdp.action_space.actions:
                if x not in mdp.terminal_states:
                    # Process sum of the values of the neighbouring states
                    summ = 0
                    for y in range(mdp.nb_states):
                        summ = summ + mdp.P[x, u, y] * v_old[y]
                    v_temp.append(mdp.r[x, u] + summ)
                else:  # if the state is final, then we only take the reward into account
                    v_temp.append(mdp.r[x, u])

                    # Select the highest state value among those computed
            v[x] = np.max(v_temp)

        # Test if convergence has been reached
        if (np.linalg.norm(v - v_old)) < 0.01:
            stop = True
    return np.all(v)


def build_maze(width, height, walls, hit=False):
    ts = height * width - 1 - len(walls)
    maze = Maze(width, height, hit, walls=walls, terminal_states=[ts])  # Markov Decision Process definition
    return maze.mdp


def create_random_maze(width, height, ratio):
    size = width * height
    n_walls = round(ratio * size)

    stop = False
    mdp = None
    # the loop below is used to check that the maze has a solution
    # if one of the values after check_navigability is null, then another maze should be produced
    while not stop:
        walls = random.sample(range(size), int(n_walls))

        mdp = build_maze(width, height, walls)
        stop = check_navigability(mdp)
    return mdp


class Maze:  # describes a maze-like environment
    def __init__(self, width, height, hit=False, walls=[], action_list=[], nb_actions=4,
                 gamma=0.9, timeout=50, start_states=[0], terminal_states=[]):
        """
        :param width: Int number defining the maze width
        :param height: Int number defining the maze height
        :param walls: List of the states that represent walls in our maze environment
        :param action_list: List of possible actions
        :param nb_actions: used when action_list is empty, by default there are 4 of them (go north, south, eat or west)
        :param gamma: Discount factor of the mdp
        :param timeout: Defines the length of an episode (max timestep) --see done() function
        :param start_states: List defining the states where the agent can be at the beginning of an episode
        :param terminal_states: List defining the states corresponding to the end of an episode
                        (agent reaches a terminal state) --cf. done() function
        """
        self.width = width
        self.height = height
        self.cells = np.zeros((width, height), int)
        self.walls = walls
        self.size = width*height
        state = 0
        cell = 0

        self.terminal_states = terminal_states
        self.state_width = []
        self.state_height = []
        # ##################### State Space ######################
        for i in range(width):
            for j in range(height):
                if cell not in walls:  # or self.cells[i][j] in self.terminal_states):
                    self.cells[i][j] = state
                    state = state + 1
                    self.state_width.append(i)
                    self.state_height.append(j)
                else:
                    self.cells[i][j] = -1
                cell = cell + 1

        self.nb_states = state

        # ##################### Action Space ######################
        self.action_space = SimpleActionSpace(action_list=action_list, nactions=nb_actions)

        # ##################### Distribution Over Initial States ######################
        
        start_distribution = np.zeros(self.nb_states)  # distribution over initial states

        # supposed to be uniform
        for state in start_states:
            start_distribution[state] = 1.0/len(start_states)

        # ##################### Transition Matrix ######################

        # a "well" state is added that only the terminal states can get into
        transition_matrix = np.empty((self.nb_states+1, self.action_space.size, self.nb_states+1))

        # Init the transition matrix
        transition_matrix[:, N, :] = np.zeros((self.nb_states+1, self.nb_states+1))
        transition_matrix[:, S, :] = np.zeros((self.nb_states+1, self.nb_states+1))
        transition_matrix[:, E, :] = np.zeros((self.nb_states+1, self.nb_states+1))
        transition_matrix[:, W, :] = np.zeros((self.nb_states+1, self.nb_states+1))
        
        for i in range(self.width):
            for j in range(self.height):
                state = self.cells[i][j]
                if not state == -1:

                    # Transition Matrix when going north (no state change if highest cells or cells under a wall)
                    if j == 0 or self.cells[i][j-1] == -1:
                        transition_matrix[state][N][state] = 1.0
                    else:  # it goes up
                        transition_matrix[state][N][self.cells[i][j-1]] = 1.0
        
                    # Transition Matrix when going south (no state change if lowest cells or cells above a wall)
                    if j == self.height-1 or self.cells[i][j+1] == -1:
                        transition_matrix[state][S][state] = 1.0
                    else:  # it goes down
                        transition_matrix[state][S][self.cells[i][j+1]] = 1.0

                    # Transition Matrix when going east (no state change if left cells or on the left side of a wall)
                    if i == self.width-1 or self.cells[i+1][j] == -1:
                        transition_matrix[state][E][state] = 1.0
                    else:  # it goes left
                        transition_matrix[state][E][self.cells[i+1][j]] = 1.0

                    # Transition Matrix when going west (no state change if right cells or on the right side of a wall)
                    if i == 0 or self.cells[i-1][j] == -1:
                        transition_matrix[state][W][state] = 1.0
                    else:  # it goes right
                        transition_matrix[state][W][self.cells[i-1][j]] = 1.0
                
        # Transition Matrix of terminal states 
        well = self.nb_states  # all the final states' transitions go there
        for s in self.terminal_states:
            transition_matrix[s, :, :] = 0
            transition_matrix[s, :, well] = 1

        if hit:
            reward_matrix = self.reward_hit_walls()
        else:
            reward_matrix = self.simple_reward()

        plotter = MazePlotter(self)  # renders the environment

        self.mdp = Mdp(self.nb_states, self.action_space, start_distribution, transition_matrix, reward_matrix,
                       plotter, gamma=gamma, terminal_states=terminal_states, timeout=timeout)

        # self.mdp = MyMdp(self.nb_states, self.action_space, start_distribution, transition_matrix, reward_matrix,
        #                plotter, proba_action=0.5, gamma=gamma, terminal_states=terminal_states, timeout=timeout)

    # --------------------------------- Reward Matrix ---------------------------------
    def simple_reward(self):
        reward_matrix = np.zeros((self.nb_states, self.action_space.size))
        for s in self.terminal_states:
            reward_matrix[s, :] = 1  # leaving a final state gets the agent a reward of 1
        return reward_matrix

    # --------------------------------- Reward Matrix ---------------------------------
    def reward_hit_walls(self):
        reward_matrix = np.zeros((self.nb_states, self.action_space.size))
        for s in self.terminal_states:
            reward_matrix[s, :] = 1  # leaving a final state gets the agent a reward of 1
            for i in range(self.width):
                for j in range(self.height):
                    state = self.cells[i][j]
                    if not state == -1:

                        # Reward Matrix when going north
                        if j == 0 or self.cells[i][j-1] == -1:  # highest cells + cells under a wall
                            reward_matrix[state, N] = -0.5

                        # Reward Matrix when going south
                        if j == self.height-1 or self.cells[i][j+1] == -1:  # lowest cells + cells above a wall
                            reward_matrix[state, S] = -0.5

                        # Reward Matrix when going east
                        if i == self.width-1 or self.cells[i+1][j] == -1:  # cells on the left + left side of a wall
                            reward_matrix[state, E] = -0.5

                        # Reward Matrix when going west
                        if i == 0 or self.cells[i-1][j] == -1:  # cells on the right + right side of a wall
                            reward_matrix[state, W] = -0.5
            return reward_matrix
