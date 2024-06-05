from typing import List
from pathlib import Path
import anywidget
import ipyreact
import traitlets
import importlib.resources
import numpy as np

from mazemdp.maze import Maze

def get_file(name: str):
    with importlib.resources.path("mazemdp", name) as path:
        if isinstance(path, Path):
            return path
        return path.read_text()
    
def array_to_json(array: np.ndarray, widget: anywidget.AnyWidget):
    return array.tolist()

class MazeWidget(ipyreact.Widget):
    # Widget front-end JavaScript code
    _esm = get_file("nbwidget.tsx")
    _css = get_file("nbwidget.css")

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
    values = traitlets.List(None).tag(sync=True)

    #: List of v or q-values
    history: List
    
    def __init__(self, mdp: Maze, title=""):
        super().__init__(title="", cells=mdp.cells, terminal_states=mdp.terminal_states)
        self.mdp = mdp
        self.history = []
        
    @traitlets.observe("step")
    def observe_step(self, change):
        self.set_trait("values", self.history[change.new-1])
        
    def add_step(self, value=None, policy=None, agent_state=None):
        self.history.append(value)
        
        # Change the # of steps
        self.set_trait("steps", len(self.history))
        
        # Move to the last
        self.set_trait("step", len(self.history))
        
    def set_title(self, title: str):
        self.set_trait("title", title)
