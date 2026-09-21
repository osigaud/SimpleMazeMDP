"""Tests for the notebook widget.

The widget is only built here, never displayed: what matters is the state it
sends to the front-end.
"""

from mazemdp import create_random_maze
from mazemdp.nbwidget import MazeWidget


def widget():
    mdp, *_ = create_random_maze(4, 4, False, 0.2, None, seed=1)
    return MazeWidget(mdp.plotter.maze_attr, title="Test widget")


def test_the_first_step_is_shown():
    """The values of the first step reach the front-end.

    `add_step` moves `step` from 1 to 1 the first time, so the observer does
    not fire and only `show_step` keeps the maze from being drawn empty.
    """
    w = widget()
    w.add_step(value=[0.5] * w.mdp.nb_states, policy=[1] * w.mdp.nb_states)

    assert w.step == 1
    assert w.steps == 1
    assert w.values == [0.5] * w.mdp.nb_states
    assert w.policy == [1] * w.mdp.nb_states


def test_a_later_step_is_shown():
    w = widget()
    w.add_step(value=[0.5] * w.mdp.nb_states)
    w.add_step(value=[1.5] * w.mdp.nb_states, agent_state=2)

    assert w.step == 2
    assert w.values == [1.5] * w.mdp.nb_states
    assert w.agent_state == 2


def test_stepping_back_shows_the_previous_step():
    w = widget()
    w.add_step(value=[0.5] * w.mdp.nb_states)
    w.add_step(value=[1.5] * w.mdp.nb_states)
    w.history_step(-1)

    assert w.step == 1
    assert w.values == [0.5] * w.mdp.nb_states
