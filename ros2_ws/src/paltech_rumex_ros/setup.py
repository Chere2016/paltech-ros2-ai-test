from setuptools import setup

package_name = "paltech_rumex_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Paltech Candidate",
    maintainer_email="devnull@example.com",
    description="ROS2 nodes for Rumex plant detection and publishing.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "image_publisher = paltech_rumex_ros.image_publisher:main",
            "plant_detector = paltech_rumex_ros.detection_node:main",
        ],
    },
)
