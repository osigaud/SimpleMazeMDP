import os
import subprocess

from setuptools import setup

with open(os.path.join("mazemdp", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

# Taken from PyTorch code to have a different version per commit
# hash = (
#     subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=".")
#     .decode("ascii")
#     .strip()
# )

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name="mazemdp",
    url="https://github.com/osigaud/SimpleMazeMDP",
    author="Olivier Sigaud",
    author_email="Olivier.Sigaud@isir.upmc.fr",
    # Needed to actually package something
    packages=["mazemdp"],
    version=__version__,
    # Needed for dependencies
    install_requires=["numpy", "matplotlib"],
    # *strongly* suggested for sharing
    # version=f"{__version__}.dev0+{hash}",
    # The license can be anything you like
    license="MIT",
    description="An simple maze to test dynamic programming and tabular reinforcement learning algorithms",
    long_description=open("README.md").read(),
)
