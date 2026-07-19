from setuptools import find_packages, setup

package_name = 'tof_viz_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='omni@local',
    description='ToF sensor MarkerArray visualizer for Foxglove — coloured distance arrows, no labels',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tof_viz_node = tof_viz_node.tof_viz_node:main',
        ],
    },
)
