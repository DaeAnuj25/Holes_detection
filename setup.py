from glob import glob
from setuptools import find_packages, setup

package_name = 'pothole_detection'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'new_model.pt', 'tutorial.txt']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/web', glob('web/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Pothole Detection Maintainers',
    maintainer_email='maintainer@example.com',
    description='Grayscale pothole detector and Linux dashboard for ROS 2 Jazzy',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pothole_detector = pothole_detection.node:main',
        ],
    },
)
