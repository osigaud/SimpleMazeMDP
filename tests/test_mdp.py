import numpy as np

from mazemdp import create_random_maze
from mazemdp.chrono import Chrono
from mazemdp.toolbox import egreedy_loc

np.random.seed(0)


def test_create():
    chrono = Chrono()
    mdp = create_random_maze(5, 5, 0.2, hit=True)
    mdp.new_render("Test visu maze")
    chrono.stop()


def test_maze_visu():
    mdp, _ = create_random_maze(4, 5, 0.2)
    mdp.new_render("Test visu value")
    for _ in range(3):
        random_value = np.random.random(size=(mdp.nb_states,))
        mdp.render(random_value, title="Test visu")

    random_policy = np.random.randint(
        len(mdp.action_space.actions), size=(mdp.nb_states,)
    )
    # Note: apparently not showing anything...
    mdp.render(random_value, random_policy, title="Test visu")
    mdp.plotter.render_pi(random_policy)

    mdp.new_render("Test visu q-value")
    for _ in range(3):
        random_q_value = np.random.random(size=(mdp.nb_states, mdp.action_space.size))
        mdp.render(random_q_value, title="Test visu q-value")


def test_step():
    mdp, _ = create_random_maze(5, 4, 0.2)
    x = mdp.reset(uniform=True)
    done = mdp.done()
    random_policy = np.random.randint(
        len(mdp.action_space.actions), size=(mdp.nb_states,)
    )
    random_value = np.random.random(size=(mdp.nb_states,))
    mdp.new_render("Test step")

    while not done:
        # Show agent
        mdp.render(random_value, random_policy)
        x, _, done, _ = mdp.step(
            egreedy_loc(random_policy[x], mdp.action_space.size, epsilon=0.2)
        )
