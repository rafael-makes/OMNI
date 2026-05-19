from setuptools import find_packages, setup

package_name = 'chest_node'

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
    description='OMNI chest panel — ESP32 serial bridge for state display, EQ, battery, and WiFi config',
    license='MIT',
    entry_points={
        'console_scripts': [
            'chest_node = chest_node.chest_node:main',
        ],
    },
)
