import os
from glob import glob
from setuptools import setup

package_name = "f1sim_ros"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "maps"), glob("maps/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="shchon11",
    maintainer_email="shchon724@gmail.com",
    description="ROS 2 bridge for the f1sim simulator",
    license="MIT",
    entry_points={"console_scripts": ["bridge = f1sim_ros.bridge_node:main", "vesc_sim = f1sim_ros.vesc_sim_node:main", "teleop = f1sim_ros.teleop_node:main", "policy = f1sim_ros.policy_node:main"]},
)
