from setuptools import find_packages, setup

package_name = 'lidar_node'

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
    description='OMNI LDROBOT LD19 LiDAR — publishes /scan (sensor_msgs/LaserScan)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lidar_node = lidar_node.lidar_node:main',
        ],
    },
)
