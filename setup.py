## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['srbl'],
    package_dir={'': 'scripts'},
    install_requires=[
        'dynamixel-sdk',
    ],
)
setup(**setup_args)