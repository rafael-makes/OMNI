import os
from glob import glob

from setuptools import setup

package_name = 'dock_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='pi@omni.local',
    description='Back-in AprilTag docking controller for OMNI (pixel-servo + rear ToF stop)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dock_node = dock_node.dock_node:main',
        ],
    },
)
