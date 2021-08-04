'''
Author: Olivier Sigaud
'''

import numpy as np
from mazemdp.toolbox import N, S, E, W
from mazemdp.maze_plotter import MazePlotter  # used to plot the maze
from mazemdp.mdp import SimpleActionSpace, Mdp


def build_maze(width, height, walls, hit=False):
    ts = height * width - 1 - len(walls)
    maze = Maze(width, height, hit, walls=walls, terminal_states=[ts])  # Markov Decision Process definition
    return maze.mdp

class Maze:  # describes a maze-like environment
    def __init__(self, width, height, hit=False, walls=[], action_list=[], nb_actions=4,
                 gamma=0.9, timeout=50, start_states=[0], terminal_states=[]):
        # width, height : int numbers defining the maze attributes
        # walls : list of the states that represent walls in our maze environment
        # action_list : list of possible actions
        # nb_actions : used when action_list is empty, by default there are 4 of them (go north, south, eat or west)
        # gamma : the discount factor of our mdp
        # timeout : defines the length of an episode (max timestep) --see done() function
        # start_states : list that defines the states where the agent can be at the beginning of an episode
        # terminal_states : list that defines the states corresponding to the end of an episode
        #                  (agent reaches a terminal state) --cf. done() function
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
