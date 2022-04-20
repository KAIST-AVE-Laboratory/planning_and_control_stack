from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['ave_control_stack', 'ave_planning_stack'],
    package_dir={'': 'src'},
)

setup(**d)