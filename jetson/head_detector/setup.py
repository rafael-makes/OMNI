from setuptools import find_packages, setup

package_name = 'head_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/head_detector.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rafael',
    maintainer_email='redline6k@gmail.com',
    description='Jetson IMX219 + YOLO26n TensorRT person detector for OMNI head tracking.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'head_detector_node = head_detector.head_detector_node:main',
        ],
    },
)
