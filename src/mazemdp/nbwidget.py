from typing import List, Optional
from pathlib import Path
import anywidget
import ipyreact
from dataclasses import dataclass
import traitlets
import importlib.resources
import numpy as np

from mazemdp.maze import Maze


def get_file(name: str, force_string: bool = False):
    with importlib.resources.path("mazemdp", name) as path:
        if isinstance(path, Path) and not force_string:
            return path
        return path.read_text()


def array_to_json(array: np.ndarray, widget: anywidget.AnyWidget):
    return array.tolist()


@dataclass
class HistoryStep:
    values: np.ndarray
    policy: Optional[np.ndarray]
    agent_state: Optional[int]


class MazeWidget(ipyreact.Widget):
    # Widget front-end JavaScript code
    _esm = get_file("nbwidget.tsx")
    _css = get_file("nbwidget.css", True)

    # Stateful property that can be accessed by JavaScript & Python
    step = traitlets.Int(1).tag(sync=True)

    # Stateful property that can be accessed by JavaScript & Python
    steps = traitlets.Int(1).tag(sync=True)

    # Stateful property that can be accessed by JavaScript & Python
    title = traitlets.Unicode("No title").tag(sync=True)

    # The maze cells
    cells = traitlets.Instance(np.ndarray).tag(sync=True, to_json=array_to_json)
    terminal_states = traitlets.List(traitlets.Int()).tag(sync=True)

    # Values (V or Q, depending on shape)
    policy = traitlets.List(None, allow_none=True).tag(sync=True)

    # Policy (deterministic or not, depending on the shape)
    values = traitlets.List(None, allow_none=True).tag(sync=True)

    # Policy (deterministic or not, depending on the shape)
    agent_state = traitlets.Integer(None, allow_none=True).tag(sync=True)

    #: List of v or q-values
    history: List[HistoryStep]

    def __init__(self, mdp: Maze, title=""):
        super().__init__(
            title="", cells=mdp.cells.T, terminal_states=mdp.terminal_states
        )
        self.mdp = mdp
        self.history = []

    def show_step(self, step: int):
        """Send the values, policy and agent position of `step` to the front-end"""
        history_step = self.history[step - 1]
        self.set_trait("values", history_step.values)
        self.set_trait("policy", history_step.policy)
        self.set_trait("agent_state", history_step.agent_state)

    @traitlets.observe("step")
    def observe_step(self, change):
        self.show_step(change.new)

    def history_step(self, delta):
        new_step = self.step + delta
        if new_step >= 0 and new_step <= len(self.history):
            self.set_trait("step", new_step)

    def add_step(self, value=None, policy=None, agent_state=None):
        self.history.append(HistoryStep(value, policy, agent_state))

        # Change the # of steps
        self.set_trait("steps", len(self.history))

        # Move to the last -- and show it: the first step does not move
        # `step` (it already is 1), so the observer would not fire and the
        # maze would be drawn empty
        self.set_trait("step", len(self.history))
        self.show_step(len(self.history))

    def set_title(self, title: str):
        self.set_trait("title", title)
