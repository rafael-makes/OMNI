from setuptools import find_packages, setup

package_name = 'audio_node'

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
    description='OMNI audio — Gemini Live API bridge with Pi Zero W TCP audio relay',
    license='MIT',
    entry_points={
        'console_scripts': [
            'audio_node = audio_node.audio_node:main',
        ],
    },
)
