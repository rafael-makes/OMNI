from setuptools import find_packages, setup

package_name = 'rear_camera'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/rear_camera.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rafael',
    maintainer_email='redline6k@gmail.com',
    description="Rear 2K USB camera owner for OMNI.",
    license='MIT',
    entry_points={
        'console_scripts': [
            'rear_camera_node = rear_camera.rear_camera_node:main',
        ],
    },
)
