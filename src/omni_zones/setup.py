from glob import glob

from setuptools import setup

package_name = "omni_zones"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pi",
    maintainer_email="pi@omni.local",
    description="OMNI named zones (rooms) as map-frame polygons — ROS-free shared library",
    license="MIT",
    # No entry points: this is a library, not a node.
)
