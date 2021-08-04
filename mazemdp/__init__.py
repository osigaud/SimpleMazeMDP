import random
import numpy as np
import matplotlib.pyplot as plt

from mazemdp.maze import build_maze, create_random_maze
from mazemdp.toolbox import random_policy, softmax, egreedy_loc, egreedy
