from setuptools import find_packages, setup

package_name = 'yaw_fusion_node'

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
    description='OMNI yaw fusion — blends BNO085 yaw with wheel odometry for accurate heading',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yaw_fusion_node = yaw_fusion_node.yaw_fusion_node:main',
        ],
    },
)
