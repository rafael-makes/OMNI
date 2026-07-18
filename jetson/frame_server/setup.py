from setuptools import find_packages, setup

package_name = 'frame_server'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/frame_server.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rafael',
    maintainer_email='redline6k@gmail.com',
    description='JPEG frame service for OMNI scene description.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'frame_server_node = frame_server.frame_server_node:main',
        ],
    },
)
