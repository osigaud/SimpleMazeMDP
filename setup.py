import os
import subprocess

from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name="mazemdp",
    url="https://github.com/osigaud/SimpleMazeMDP",
    author="Olivier Sigaud",
    author_email="Olivier.Sigaud@isir.upmc.fr",
    # Needed to actually package something
    packages=["mazemdp"],
    # Needed for dependencies
    install_requires=["numpy", "matplotlib"],
    # *strongly* suggested for sharing
    # version=f"{__version__}.dev0+{hash}",
    # The license can be anything you like
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    license="MIT",
    description="An simple maze to test dynamic programming and tabular reinforcement learning algorithms",
    long_description=open("README.md").read(),
)
