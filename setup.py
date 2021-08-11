import os
from setuptools import setup

with open(os.path.join("mazemdp", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='simplemazemdp',
    url='https://github.com/osigaud/SimpleMazeMDP',
    author='Olivier Sigaud',
    author_email='Olivier.Sigaud@upmc.fr',
    # Needed to actually package something
    packages=['mazemdp'],
    # Needed for dependencies
    install_requires=['numpy', 'plotly>=5.1.0'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='An simple maze to test dynamic programming and tabular reinforcement learning algorithms',
    long_description=open('README.md').read(),
)
