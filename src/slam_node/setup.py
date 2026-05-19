from setuptools import setup
import os
from glob import glob

package_name = 'slam_node'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='omni@local',
    description='OMNI slam_node — slam_toolbox launch and configuration',
    license='MIT',
    entry_points={
        'console_scripts': [
            'map_saver = slam_node.map_saver:main',
        ],
    },
)
