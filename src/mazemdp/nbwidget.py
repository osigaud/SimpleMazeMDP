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

class MazeWidget(anywidget.AnyWidget):
    # Widget front-end JavaScript code
    _esm = get_file("nbwidget.js")
    _css = get_file("nbwidget.css")

    # Stateful property that can be accessed by JavaScript & Python
    ready = traitlets.Bool(False).tag(sync=True)

    # Stateful property that can be accessed by JavaScript & Python
    step = traitlets.Int(0).tag(sync=True)

    # Stateful property that can be accessed by JavaScript & Python
    steps = traitlets.Int(0).tag(sync=True)

    # Stateful property that can be accessed by JavaScript & Python
    title = traitlets.Unicode("No title").tag(sync=True)
    
    # The maze cells
    cells = traitlets.Instance(np.ndarray).tag(sync=True, to_json=array_to_json)
    terminal_states = traitlets.List(traitlets.Int()).tag(sync=True)

    def __init__(self, mdp: Maze, title=""):
        super().__init__(title="", cells=mdp.cells, terminal_states=mdp.terminal_states)
        self.mdp = mdp

        self.observe(self._ready, "ready", "change")
        
    def _ready(self, change):
        print(change)
        self.unobserve(self._ready, "ready", "change")
        if change.new:
            print("Yeahh")
        
    def set_title(self, title: str):
        self.set_trait("title", title)