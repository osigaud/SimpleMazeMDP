# SimpleMazeMDP: 

This repository contains code to provide a simple Maze environment used as example MDP for tabular dynamic programming and reinforcement learning labs.

## Documentation ##


### MDPs and mazes ###

Some code is provided to create mazes, transform them into MDPs and visualize them together with policies or value functions.
It is contained into three files: \url{maze.py}, \url{mdp.py} and \url{maze_plotter.py}.
The following sections give an overview of this code.

#### Content of the maze.py file ####

A maze is represented as an object of the \url{Maze} class. It is defined as a grid of $width \times height$ cells, and some of these cells contain a wall.

The \url{build_maze(width, height, walls, hit=False)} function is used to create a Maze, where \url{walls} is a list of the number of the cells which contain a wall.
The \url{hit} parameter has an impact on the MDP reward function: if \url{hit} is \url{true}, the agent is penalized each time it tries to move to a wall cell. Otherwise,
the agent is just rewarded when it reaches terminal states. In the provided function, the list of terminal states is a singleton corresponding to the last cell that the agent can visit.

Apart from representing the two reward functions described above, the \url{Maze} class contains a constructor whose only role is to create the MDP corresponding to the maze and the maze plotter used to display simulations. A key point is that only cells where there is no wall are considered as states of the underlying MDP. To facilitate the correspondence between mazes and MDPs, each free cell (i.e. with no wall) knows the number of its corresponding MDP state.

The maze constructors also builds the action space, the initial state distribution, the transition function and the reward function of the MDP. Once all these data structures have been created, the resulting MDP is built.

#### Content of the mdp.py file ####

The \url{mdp.py} file contains the \url{SimpleActionSpace} class and the \url{Mdp} class.

The \url{SimpleActionSpace} class contains the list of actions and a method to sample from this list. In our maze environment, the possible actions for the agent are going north, south, east or west (resp. [0, 1, 2, 3]).

The \url{Mdp} class is designed to be compatible with the OpenAI gym interface\footnote{\url{https://gym.openai.com/}}. The main methods are \url{reset(self, uniform=False)}, which resets the MDP into an initial state drawn from the initial state distribution, and \url{step(self, u, deviation=0)} which is used to let the agent perform a step in the environment, sending and action and receiving the next state, the reward, and a signal telling whether a terminal state was reached.

#### Content of the maze_plotter.py file ####

The code to display the effect of the algorithms in these environments is in \url{maze_plotter.py}, in the \url{MazePlotter} class.
In order to visualize the environment, you use the \url{new_render()} function to initialize the rendering, then \url{render(V, policy, agent_pos)} to refresh the maze with either the newly calculated state values and the policy, or the state-action values, and eventually the current position of the agent. There is also a \url{render_pi(policy)} function which only displays the policy (useful for {\sc policy iteration}). The function \url{save_fig(title)} is used to save the last render into a file.

%Given a list of the different $V$ or $Q$ values, a list of policies, and the number of your frames, you can generate a video (animation) of your results using the function \url{create_animation()}. It is particularly useful in the RL functions, where the number of episodes is high and outputting the results during the process makes it last longer. 

You can see examples of calls to these different visualizations in the functions defined in \url{dynamic_programming.py} and \url{reinforcement_learning.py}.
