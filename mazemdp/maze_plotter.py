'''
Author: Olivier Sigaud
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.table import Table
from mazemdp.toolbox import N, S, E, W

# ------------------- plot functions for a maze like environment ----------------#


def coords(width, height, i, j):
    # processes the starting position of the arrows
    # i is the width
    # j is the height
    x = 0.15 - 0.01 * width + i / width
    y = 0.8 + 0.02 * height - j / height
    return x, y


def arrow_params(width, height, i, j, action):
    # processes the starting position of the arrows
    x, y = coords(width, height, i, j)

    if action == N:
        return [x, y + 0.08 / height, 0.0, 0.16 / height]
    elif action == S:
        return [x, y - 0.08 / height, 0.0, -0.16 / height]
    elif action == E:
        return [x + 0.12 / width, y, 0.08 / width, 0.0]
    elif action == W:
        return [x - 0.12 / width, y, -0.08 / width, 0.0]
    else:
        return [x, y, 0.0, 0.0]


def qvalue_params(height, width, i, j, action):
    x, y = coords(height, width, i, j)

    if action == N:
        return [x - 0.04 / width, y + 0.32 / height]
    elif action == S:
        return [x - 0.04 / width, y - 0.4 / height]
    elif action == E:
        return [x + 0.32 / width, y]
    elif action == W:
        return [x - 0.24 / width, y]
    else:
        return [x, y]


def qarrow_params(width, height, i, j, action):
    # processes the starting position of the arrows
    x, y = coords(width, height, i, j)

    if action == N:
        return [x, y + 0.12 / height, 0.0, 0.08 / height]
    elif action == S:
        return [x, y - 0.12 / height, 0.0, -0.08 / height]
    elif action == E:
        return [x + 0.12 / width, y, 0.02 / width, 0.0]
    elif action == W:
        return [x - 0.12 / width, y, -0.02 / width, 0.0]
    else:
        return [x, y, 0.0, 0.0]

# -------------------------------------------------------------------------------#
# maze_mdp plot, used to plot the agent in its environment while processing the V/Q function and policy
# it can also create videos given a list of V/Q values and a list of policies


class MazePlotter:
    def __init__(self, maze):  # maze defined in the mdp notebook
        self.maze_attr = maze
        self.terminal_states = maze.terminal_states
        plt.ion()
        self.figW = self.maze_attr.width
        self.figH = self.maze_attr.height
        self.figure_history = []
        self.axes_history = []
        self.table_history = []
        self.agent_patch_history = []
        
    def init_table(self):  # the states of the mdp are drawn in a matplotlib table, this function creates this table
        
        width = 0.1
        height = 0.2
        
        for i in range(self.maze_attr.width):
            for j in range(self.maze_attr.height):
                color = np.zeros(3)
                if self.maze_attr.cells[i][j] == -1:
                    color[0] = color[1] = color[2] = 0
                else:
                    color[0] = color[1] = color[2] = 1
                self.table_history[-1].add_cell(j, i, width, height, facecolor=color, text='', loc='center')
        
        self.axes_history[-1].add_table(self.table_history[-1])
    
    def new_render(self, title):  # initializes the plot by creating its basic components (figure, axis, agent patch and table)
        # a trace of these components is stored so that the old outputs will last on the notebook
        # when a new rendering is performed
        plt.title(title)
        self.figure_history.append(plt.figure(figsize=(self.figW, self.figH)))
        self.axes_history.append(self.figure_history[-1].add_subplot(111))
        self.table_history.append(Table(self.axes_history[-1], bbox=[0, 0, 1, 1]))
        self.agent_patch_history.append(mpatches.Ellipse((-1, -1), 0.06, 0.085, ec="none", fc="dodgerblue", alpha=0.6))
        self.axes_history[-1].add_patch(self.agent_patch_history[-1])
        self.init_table()
    
    def render(self, agent_state=-1, v=[], policy=[], stochastic=False, title='No Title'):  # updates the values of the table
        # and the agent position and current policy 
        # some of these components may not show depending on the parameters given when calling this function
        if len(self.figure_history) == 0:  # new plot
            self.new_render(title)
        
        self.axes_history[-1].clear()
        self.axes_history[-1].add_table(self.table_history[-1])
        
        # Table values and policy update
        for i in range(self.maze_attr.width):
            for j in range(self.maze_attr.height):
                state = self.maze_attr.cells[i][j]
                if len(v) > 0:  # working with state values
                    if len(v.shape) == 1:
                        self.cell_render_v(v, i, j, state)
                    else:  # working with state-action values
                        self.cell_render_q(v, i, j, state)
                if len(policy) > 0:
                    if stochastic:
                        self.render_stochastic_policy(v, policy, i, j, state)
                    else:
                        self.render_policy(policy, i, j, state)
        
        if agent_state >= 0:
            x, y = coords(self.maze_attr.width, self.maze_attr.height,
                          self.maze_attr.state_width[agent_state],
                          self.maze_attr.state_height[agent_state])
            self.agent_patch_history[-1].center = x, y
            self.axes_history[-1].add_patch(self.agent_patch_history[-1])
        
        plt.subplots_adjust(left=0.2, bottom=0.2)
        plt.xticks([])
        plt.yticks([])
        self.figure_history[-1].canvas.draw()
        self.figure_history[-1].canvas.flush_events()

    def cell_render_v(self, v, i, j, state):
        color = np.zeros(3)
        if state == -1:
            color[0] = color[1] = color[2] = 0
        else:
            color[0] = color[1] = color[2] = np.min([1 - v[state] / (np.max(v) + 1), 1])

        self.table_history[-1]._cells[(j, i)].set_facecolor(color)
        self.table_history[-1]._cells[(j, i)]._text.set_text(np.round(v[state], 2))

    def cell_render_q(self, q, i, j, state):
        color = np.zeros(3)
        if state == -1:
            color[0] = color[1] = color[2] = 0
        else:
            color[0] = color[1] = color[2] = np.min([1 - np.max(q[state]) / (np.max(q) + 1), 1])

        self.table_history[-1]._cells[(j, i)].set_facecolor(color)
        self.table_history[-1]._cells[(j, i)]._text.set_text(np.round(np.max(q[state]), 2))

        if not (state == -1 or state in self.terminal_states):
            qmin = np.min(q[state])
            if qmin < 0:
                qmin *= -1  # TODO: should deal better with negative Q values
            pos_q = q[state] + qmin
            qmax = np.max(pos_q)
            norm_q = pos_q / (np.sum(pos_q) - (list(pos_q).count(qmax) * qmax) + 0.1)

            for action in range(len(q[state])):

                x0, y0, x, y = qarrow_params(self.maze_attr.width, self.maze_attr.height, i, j, action)

                arw_color = "green"
                alpha = 0.9
                qmax = np.max(q[state])

                if not q[state][action] == qmax:
                    arw_color = "red"
                    alpha = norm_q[action]

                if x == 0 and y == 0:
                    circle = mpatches.Circle((x0, y0), 0.08 / self.maze_attr.width, ec=arw_color, fc=arw_color, alpha=alpha)
                    self.axes_history[-1].add_patch(circle)
                else:
                    self.axes_history[-1].arrow(x0, y0, x, y, alpha=alpha,
                                                head_width=0.12 / self.maze_attr.width,
                                                head_length=0.12 / self.maze_attr.height,
                                                fc=arw_color, ec=arw_color)

    def render_pi(self, policy):
        for i in range(self.maze_attr.width):
            for j in range(self.maze_attr.height):
                state = self.maze_attr.cells[i][j]
                self.render_policy(policy, i, j, state)
        self.figure_history[-1].canvas.draw()
        self.figure_history[-1].canvas.flush_events()

    def render_policy(self, policy, i, j, state):
        if not (state == -1 or state in self.terminal_states):
            x0, y0, x, y, = arrow_params(self.maze_attr.width, self.maze_attr.height,
                                         i, j, policy[state])
            arw_color = "green"
            alpha = 0.6
            self.axes_history[-1].arrow(x0, y0, x, y, alpha=alpha,
                                        head_width=0.12 / self.maze_attr.width,
                                        head_length=0.12 / self.maze_attr.height,
                                        fc=arw_color,
                                        ec=arw_color)

    def render_stochastic_policy(self, q, policy, i, j, state):
        color = np.zeros(3)
        if state in self.maze_attr.walls:
            color[0] = color[1] = color[2] = 0
        else:
            color[0] = color[1] = color[2] = np.min([1 - np.max(q[state]) / (np.max(q) + 1), 1])

        self.table_history[-1]._cells[(j, i)].set_facecolor(color)
        self.table_history[-1]._cells[(j, i)]._text.set_text(np.round(np.max(q[state]), 2))

        if not (state == -1 or state in self.terminal_states):
            qmin = np.min(q[state])
            if qmin < 0:
                qmin *= -1  # TODO: should deal better with negative Q values
            pos__q = q[state] + qmin
            qmax = np.max(pos__q)
            norm_q = pos__q / (np.sum(pos__q) - (list(pos__q).count(qmax) * qmax) + 0.1)

            for action in range(len(q[state])):
                x0, y0, x, y = qarrow_params(self.maze_attr.width,
                                             self.maze_attr.height, i, j, action)

                q_x, q_y = qvalue_params(self.maze_attr.width,
                                         self.maze_attr.height, i, j, action)
                arw_color = "green"
                alpha = 0.9
                qmax = np.max(q[state])

                proba = policy[state][action]
                print(proba)
                plt.text(q_x, q_y, "{:.2f}".format(proba))

                if not q[state][action] == qmax:
                    arw_color = "red"
                    alpha = norm_q[action]

                if x == 0 and y == 0:
                    circle = mpatches.Circle((x0, y0), 0.02, ec=arw_color, fc=arw_color, alpha=alpha)
                    self.axes_history[-1].add_patch(circle)
                else:
                    self.axes_history[-1].arrow(x0, y0, x, y, alpha=alpha,
                                                head_width=0.03, head_length=0.03, fc=arw_color, ec=arw_color)

    def save_fig(self, title):
        self.figure_history[-1].savefig(title)
