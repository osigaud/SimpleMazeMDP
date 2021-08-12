import numpy as np

from mazemdp import create_random_maze


def test_maze():
    mdp = create_random_maze(10, 10, 0.2)
    random_policy = np.random.randint(len(mdp.action_space.actions), size=(mdp.nb_states,))
    random_value = np.random.random(size=(mdp.nb_states,))


def test_maze_visu():
    mdp = create_random_maze(10, 10, 0.2)
    mdp.new_render("Test visu")
    for _ in range(10):
        random_value = np.random.random(size=(mdp.nb_states,))
        mdp.render(random_value, title="Test visu")

    random_policy = np.random.randint(len(mdp.action_space.actions), size=(mdp.nb_states,))
    mdp.render(random_value, random_policy, title="Test visu")
