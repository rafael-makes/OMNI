from setuptools import find_packages, setup

package_name = 'baro_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/baro_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='omni@local',
    description='BMP280 barometer node for floor detection',
    license='MIT',
    entry_points={
        'console_scripts': [
            'baro_node = baro_node.baro_node:main',
            'baro_calibrate = baro_node.calibrate:main',
            'baro_select_map = baro_node.select_map:main',
            'baro_floor_resolver = baro_node.floor_resolver:main',
        ],
    },
)
