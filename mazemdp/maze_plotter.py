"""
Author: Olivier Sigaud + Antonin Raffin
"""
from dataclasses import dataclass
from typing import Any, List
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os

try:
    import cv2
except ImportError:
    cv2 = None
import matplotlib

# Force backend
if os.environ.get("PLOT_BACKEND"):
    matplotlib.use(os.environ.get("PLOT_BACKEND"))
print(f"Matplotlib backend: {matplotlib.get_backend()}")
import base64
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

from mazemdp.toolbox import E, N, S, W

try:
    shell = get_ipython().__class__.__module__  # noqa: F401
    if shell is not None and shell in ['ipykernel.zmqshell', 'google.colab._shell']:
        os.environ["WEB_NOTEBOOK"] = shell
except NameError:
    pass

# ------------------- plot functions for a maze like environment ----------------#


def show_videos(video_path: str = "", prefix: str = "") -> None:
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: Path to the folder containing videos
    :param prefix: Filter the video, showing only the only starting with this prefix
    """
    from IPython import display as ipythondisplay

    html = []

    for avi in Path(video_path).glob(f"{prefix}*.avi"):
        mp4_video = str(avi).replace("avi", "mp4")
        # Convert
        if not os.path.isfile(mp4_video) or os.path.getmtime(mp4_video) < os.path.getmtime(avi):
            print(f"Converting {avi}")
            os.system(f"ffmpeg -y -i {avi} -c:v libx264 -crf 19 {mp4_video}")

    for mp4 in Path(video_path).glob(f"{prefix}*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def coords(width, height, i, j):
    """
    processes the starting position of the arrows
    i is the width
    j is the height
    """
    x = (0.44 + i) / width
    y = 0.98 - 0.28 / height - j / height
    return x, y


def arrow_params(width, height, i, j, action):
    """
    processes the starting position of the arrows
    """
    x, y = coords(width, height, i, j)

    if action == N:
        return [x, y + 0.08 / height, 0.0, 0.16 / height]
    elif action == S:
        return [x, y - 0.08 / height, 0.0, -0.16 / height]
    elif action == E:
        return [x + 0.16 / width, y, 0.08 / width, 0.0]
    elif action == W:
        return [x - 0.16 / width, y, -0.08 / width, 0.0]
    else:
        return [x, y, 0.0, 0.0]


def qvalue_params(width, height, i, j, action):
    x, y = coords(width, height, i, j)

    if action == N:
        return [x - 0.04 / width, y + 0.32 / height]
    elif action == S:
        return [x - 0.04 / width, y - 0.4 / height]
    elif action == E:
        return [x + 0.32 / width, y]
    elif action == W:
        return [x - 0.24 / width, y]
    else:
        return [x, y]


# -------------------------------------------------------------------------------#



@dataclass
class Plot:
    canvas: FigureCanvasAgg
    fig: Figure
    axes: Axes
    table: Table
    patch: Any



class MazePlotter:
    """
    maze_mdp plot, used to plot the agent in its environment while processing the V/Q function and policy
    it can also create videos given a list of V/Q values and a list of policies
    """

    def __init__(
        self, maze, using_notebook=bool(os.environ.get("WEB_NOTEBOOK", False))
    ):  # maze defined in the mdp notebook
        self.maze_attr = maze
        self.terminal_states = maze.last_states
        self.using_notebook = using_notebook
        self.figW = self.maze_attr.width
        self.figH = self.maze_attr.height
        
        self.plot_history: List[Plot] = []

        self.image_idx = 0
        self.video_writer = None
        self.video_name = ""
        self.video_folder = "videos"

        os.makedirs(self.video_folder, exist_ok=True)

        self.widget_out = None

    def init_table(self, table: Table):
        """
        the states of the mdp are drawn in a matplotlib table, this function creates this table
        """

        for i in range(self.maze_attr.width):
            for j in range(self.maze_attr.height):
                color = np.zeros(3)
                if self.maze_attr.cells[i][j] == -1:
                    color[0] = color[1] = color[2] = 0
                else:
                    color[0] = color[1] = color[2] = 1
                table.add_cell(
                    j, i, 0.1, 0.2, facecolor=color, text="", loc="center"
                )

    def display(self, rgba, mode):
        if mode == "human" or mode == "legacy":
            if self.using_notebook:
                import IPython.display as display 
                from PIL import Image
                import io
                import ipywidgets as widgets

                # Creates a new widget if needed
                # (1) no widget
                # (2) widget in another cell
                if self.widget_out is None or self.widget_execution_count != get_ipython().execution_count:
                    self.widget_out = widgets.Output()
                    self.widget_execution_count = get_ipython().execution_count
                    display.display(self.widget_out)

                self.widget_out.clear_output(True)
                with self.widget_out:
                    image = Image.fromarray(rgba, 'RGBA')
                    output = io.BytesIO()
                    image.save(output, format='png')
                    display.display(display.Image(data=output.getvalue(), format="png"))
            else:
                fig = plt.gcf()
                fig.clear()
                ax = fig.add_subplot()
                ax.axis('off')
                ax.imshow(rgba)
                plt.show(block=False)

        elif mode == "rgb_array":
            return rgba
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented")

    def render_base(self, title):
        fig = Figure(figsize=(self.figW, self.figH))
        canvas = FigureCanvasAgg(fig)
        axes = fig.add_subplot(111)
        axes.set_title(title)
        table = Table(axes, bbox=[0, 0, 1, 1])
        patch = mpatches.Ellipse(
            (-1, -1), 0.06, 0.085, ec="none", fc="dodgerblue", alpha=0.6
        )

        self.plot_history.append(Plot(canvas, fig, axes, table, patch))
        axes.add_patch(patch)

        self.init_table(table)
        axes.add_table(table)



    def new_render(self, title, mode="human"):
        """
        initializes the plot by creating its basic components (figure, axis, agent patch and table)
        a trace of these components is stored so that the old outputs will last on the notebook
        when a new rendering is performed
        """

        self.render_base(title)

        canvas = self.plot_history[-1].canvas
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())

        if mode == "legacy":
            # Do not draw if other than legacy
            self.video_name = f"{title.replace(' ', '')}.avi"
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

        return self.display(rgba, mode)

    def render(
        self,
        v=None,
        policy=None,
        agent_state=None,
        stochastic=False,
        title="No title",
        mode="legacy"
    ):
        """
        updates the values of the table
        and the agent position and current policy
        some of these components may not show depending on the parameters given when calling this function
        the agent state is set to None if we do not want to plot the agent

        Args:
            mode: legacy to use the old display mode, or human/rgba for gym environments
        """
        # self.render_base(title)

        if v is None:
            v = np.array([])

        if policy is None:
            policy = np.array([])


        if len(self.plot_history) == 0:
            self.render_base(title)
        plot = self.plot_history[-1]
        
        plot.axes.clear()
        plot.axes.add_table(plot.table)

        # Table values and policy update
        for i in range(self.maze_attr.width):
            for j in range(self.maze_attr.height):
                state = self.maze_attr.cells[i][j]
                if len(v) > 0:  # working with state values
                    if len(v.shape) == 1:
                        self.cell_render_v(plot, v, i, j, state)
                    else:  # working with state-action values
                        self.cell_render_q(plot, v, i, j, state)
                if len(policy) > 0:
                    if stochastic:
                        self.render_stochastic_policy(plot, v, policy, i, j, state)
                    else:
                        self.render_policy(plot, policy, i, j, state)

        if agent_state is not None and len(self.maze_attr.state_width) > agent_state:
            x, y = coords(
                self.maze_attr.width,
                self.maze_attr.height,
                self.maze_attr.state_width[agent_state],
                self.maze_attr.state_height[agent_state],
            )
            plot.patch.center = x, y
            plot.axes.add_patch(plot.patch)


        plot.fig.subplots_adjust(left=0.2, bottom=0.2)
        # plot.axes.xticks([])
        # plot.axes.yticks([])
        plot.fig.tight_layout()

        plot.canvas.draw()
        plot.canvas.flush_events()
        rgba = np.asarray(plot.canvas.buffer_rgba())

        if self.using_notebook and mode == "legacy":
            # Adds an image to the video

            # Get image
            image = rgba
            # Record video
            if self.video_writer is None:
                loc_height, loc_width, _ = image.shape
                codec = cv2.VideoWriter_fourcc(*"MJPG")
                fps = int(os.environ.get("VIDEO_FPS", 3))
                self.video_writer = cv2.VideoWriter(
                    f"{self.video_folder}/{self.video_name}",
                    codec,
                    fps,
                    (loc_width, loc_height),
                )
            image = image[:, :, :3]  # remove alpha
            self.video_writer.write(image[:, :, ::-1])  # convert to BGR

        return self.display(rgba, mode)

    def cell_render_v(self, plot: Plot, v, i, j, state):
        color = np.zeros(3)
        if state == -1:
            color[0] = color[1] = color[2] = 0
        else:
            color[0] = color[1] = color[2] = np.min([1 - v[state] / (np.max(v) + 1), 1])

        plot.table._cells[(j, i)].set_facecolor(color)
        plot.table._cells[(j, i)]._text.set_text(np.round(v[state], 2))

    def cell_render_q(self, plot: Plot, q, i, j, state):
        color = np.zeros(3)
        if state == -1:
            color[0] = color[1] = color[2] = 0
        else:
            color[0] = color[1] = color[2] = np.min(
                [1 - np.max(q[state]) / (np.max(q) + 1), 1]
            )

        plot.table._cells[(j, i)].set_facecolor(color)
        plot.table._cells[(j, i)]._text.set_text(
            np.round(np.max(q[state]), 2)
        )

        if not (state == -1 or state in self.terminal_states):
            qmin = np.min(q[state])
            if qmin < 0:
                qmin *= -1  # TODO: should deal better with negative Q values
            pos_q = q[state] + qmin
            qmax = np.max(pos_q)
            norm_q = pos_q / (np.sum(pos_q) - (list(pos_q).count(qmax) * qmax) + 0.1)

            for action in range(len(q[state])):

                x0, y0, x, y = arrow_params(
                    self.maze_attr.width, self.maze_attr.height, i, j, action
                )

                arw_color = "green"
                alpha = 0.9
                qmax = np.max(q[state])

                if not q[state][action] == qmax:
                    arw_color = "red"
                    alpha = norm_q[action]

                if x == 0 and y == 0:
                    circle = mpatches.Circle(
                        (x0, y0), 0.04, ec=arw_color, fc=arw_color, alpha=alpha
                    )
                    plot.axes.add_patch(circle)
                else:
                    plot.axes.arrow(
                        x0,
                        y0,
                        x,
                        y,
                        alpha=alpha,
                        head_width=0.12 / self.maze_attr.width,
                        head_length=0.12 / self.maze_attr.height,
                        fc=arw_color,
                        ec=arw_color,
                    )

    def render_pi(self, policy):
        for i in range(self.maze_attr.width):
            for j in range(self.maze_attr.height):
                state = self.maze_attr.cells[i][j]
                self.render_policy(policy, i, j, state)

    def render_policy(self, plot: Plot, policy, i, j, state):
        if not (state == -1 or state in self.terminal_states):
            x0, y0, x, y = arrow_params(
                self.maze_attr.width, self.maze_attr.height, i, j, policy[state]
            )
            arw_color = "green"
            alpha = 0.6
            plot.axes.arrow(
                x0,
                y0,
                x,
                y,
                alpha=alpha,
                head_width=0.12 / self.maze_attr.width,
                head_length=0.12 / self.maze_attr.height,
                fc=arw_color,
                ec=arw_color,
            )

    def render_stochastic_policy(self, plot: Plot, q, policy, i, j, state):
        color = np.zeros(3)
        if state in self.maze_attr.walls:
            color[0] = color[1] = color[2] = 0
        else:
            color[0] = color[1] = color[2] = np.min(
                [1 - np.max(q[state]) / (np.max(q) + 1), 1]
            )

        plot.table._cells[(j, i)].set_facecolor(color)
        plot.table._cells[(j, i)]._text.set_text(
            np.round(np.max(q[state]), 2)
        )

        if not (state == -1 or state in self.terminal_states):
            qmin = np.min(q[state])
            if qmin < 0:
                qmin *= -1  # TODO: should deal better with negative Q values
            pos__q = q[state] + qmin
            qmax = np.max(pos__q)
            norm_q = pos__q / (np.sum(pos__q) - (list(pos__q).count(qmax) * qmax) + 0.1)

            for action in range(len(q[state])):
                x0, y0, x, y = arrow_params(
                    self.maze_attr.width, self.maze_attr.height, i, j, action
                )

                q_x, q_y = qvalue_params(
                    self.maze_attr.width, self.maze_attr.height, i, j, action
                )
                arw_color = "green"
                alpha = 0.9
                qmax = np.max(q[state])

                proba = policy[state][action]
                plot.axes.text(q_x, q_y, "{:.2f}".format(proba))

                if not q[state][action] == qmax:
                    arw_color = "red"
                    alpha = norm_q[action]

                if x == 0 and y == 0:
                    circle = mpatches.Circle(
                        (x0, y0), 0.02, ec=arw_color, fc=arw_color, alpha=alpha
                    )
                    plot.axes.add_patch(circle)
                else:
                    plot.axes.arrow(
                        x0,
                        y0,
                        x,
                        y,
                        alpha=alpha,
                        head_width=0.03,
                        head_length=0.03,
                        fc=arw_color,
                        ec=arw_color,
                    )

    def save_fig(self, title):
        self.figure_history[-1].savefig(title)
