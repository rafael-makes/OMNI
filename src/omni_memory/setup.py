from glob import glob

from setuptools import setup

package_name = "omni_memory"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pi",
    maintainer_email="pi@omni.local",
    description="OMNI persistent per-person memory layer (ROS2 node wrapping the memory core)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "omni_memory_node = omni_memory.node:main",
        ],
    },
)
