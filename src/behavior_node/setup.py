from setuptools import setup
import os
from glob import glob

package_name = 'behavior_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # *.txt as well as *.yaml — scene_prompt.txt lives here and is read from
        # the share dir at runtime, so a yaml-only glob leaves it uninstalled.
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml') + glob('config/*.txt')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='omni@local',
    description='OMNI central brain',
    license='MIT',
    entry_points={
        'console_scripts': [
            'behavior_node = behavior_node.behavior_node:main',
        ],
    },
)
