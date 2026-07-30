"""Tests for the maze construction and the navigability check.

Deliberately free of any rendering, so that they run anywhere.
"""

import random

from mazemdp.maze import build_maze, create_random_maze


def layout(mdp):
    """A value identifying the maze, to compare two constructions"""
    return (mdp.nb_states, tuple(mdp.terminal_states), mdp.P.tobytes())


# --- Navigability


def test_navigability_of_an_open_maze():
    mdp, *_ = build_maze(3, 3, False, [], None)
    assert mdp.check_navigability()


def test_navigability_when_the_goal_is_walled_off():
    # in a 3x3 maze the cell index is `i * height + j`, so walling the cells 5
    # and 7 isolates the cell 8 -- which is where the terminal state sits
    mdp, *_ = build_maze(3, 3, False, [5, 7], None)
    assert not mdp.check_navigability()


def test_a_random_maze_is_navigable():
    # `create_random_maze` draws until the maze has a solution
    mdp, *_ = create_random_maze(6, 6, False, 0.2, None, seed=1)
    assert mdp.check_navigability()


# --- Seeding: the three behaviours of the `seed` argument


def test_an_integer_seed_always_gives_the_same_maze():
    first = create_random_maze(6, 6, False, 0.2, None, seed=42)[0]
    # draw other mazes in between: they must not shift the result
    create_random_maze(8, 8, False, 0.3, None)
    second = create_random_maze(6, 6, False, 0.2, None, seed=42)[0]
    assert layout(first) == layout(second)


def test_different_seeds_give_different_mazes():
    first = create_random_maze(6, 6, False, 0.2, None, seed=42)[0]
    other = create_random_maze(6, 6, False, 0.2, None, seed=43)[0]
    assert layout(first) != layout(other)


def test_a_generator_gives_a_reproducible_sequence_of_different_mazes():
    def sequence(seed):
        rng = random.Random(seed)
        return [
            layout(create_random_maze(6, 6, False, 0.2, None, seed=rng)[0])
            for _ in range(3)
        ]

    first = sequence(7)
    # successive mazes of one sequence differ ...
    assert len(set(first)) > 1
    # ... but the sequence itself is reproducible
    assert sequence(7) == first
    assert sequence(8) != first
