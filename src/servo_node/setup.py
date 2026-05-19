from setuptools import find_packages, setup

package_name = 'servo_node'

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
    description='OMNI dual PCA9685 servo control node — arms and head pan/tilt',
    license='MIT',
    entry_points={
        'console_scripts': [
            'servo_node = servo_node.servo_node:main',
        ],
    },
)
