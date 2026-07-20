from glob import glob

from setuptools import setup

package_name = "event_generator"

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
    description=(
        "OMNI semantic presence events: person_appeared / person_left / "
        "unknown_person_detected, debounced for a face-anchored world state"
    ),
    license="MIT",
    entry_points={
        "console_scripts": [
            "event_generator_node = event_generator.node:main",
        ],
    },
)
