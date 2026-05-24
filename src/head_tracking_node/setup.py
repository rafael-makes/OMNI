from setuptools import find_packages, setup

package_name = 'head_tracking_node'

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
    description='OMNI pan/tilt head tracking — P controller driven by /camera/detections',
    license='MIT',
    entry_points={
        'console_scripts': [
            'head_tracking_node = head_tracking_node.head_tracking_node:main',
        ],
    },
)
