import numpy as np

from mazemdp import create_random_maze
from mazemdp.chrono import Chrono

def test_maze():
    chrono = Chrono()
    mdp = create_random_maze(10, 10, 0.2, hit=True)
    random_policy = np.random.randint(len(mdp.action_space.actions), size=(mdp.nb_states,))
    random_value = np.random.random(size=(mdp.nb_states,))
    chrono.stop()


def test_maze_visu():
    mdp = create_random_maze(10, 10, 0.2)
    mdp.new_render("Test visu value")
    for _ in range(3):
        random_value = np.random.random(size=(mdp.nb_states,))
        mdp.render(random_value, title="Test visu")

    random_policy = np.random.randint(len(mdp.action_space.actions), size=(mdp.nb_states,))
    # Note: apparently not showing anything...
    mdp.render(random_value, random_policy, title="Test visu")
    mdp.plotter.render_pi(random_policy)

    mdp.new_render("Test visu q-value")
    for _ in range(3):
        random_q_value = np.random.random(size=(mdp.nb_states, mdp.action_space.size))
        mdp.render(random_q_value, title="Test visu q-value")

    # import matplotlib.pyplot as plt
    # plt.ioff()
    # plt.show()
