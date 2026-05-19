from setuptools import find_packages, setup

package_name = 'tof_node'

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
    description='OMNI VL53L0X TOF sensors — publishes 6x sensor_msgs/Range via PCA9548A mux',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tof_node = tof_node.tof_node:main',
        ],
    },
)
