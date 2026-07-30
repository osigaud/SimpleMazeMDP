import matplotlib

# Must happen before anything imports pyplot: the tests only check that
# rendering runs, and an interactive backend needs a display (and a Tk
# installation) that a headless machine does not have.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    """Close the figures a test leaves behind.

    The renderer calls `plt.show(block=False)`, which keeps every figure open;
    without this the interpreter can fail to exit once the tests are done.
    """
    yield
    plt.close("all")
