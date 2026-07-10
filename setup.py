from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'nmpc_robot_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),  glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),  glob('config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'),    glob('urdf/*')),
        (os.path.join('share', package_name, 'worlds'),  glob('worlds/*')),
        (os.path.join('share', package_name, 'rviz'),    glob('rviz/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Pradeep Sivaa Aiyanar',
    maintainer_email='pradeepsivaa2003@gmail.com',
    description='NMPC with LiDAR obstacle avoidance for differential-drive robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nmpc_controller    = nmpc_robot_nav.nmpc_controller:main',
            'obstacle_detector  = nmpc_robot_nav.obstacle_detector:main',
            'global_planner     = nmpc_robot_nav.global_planner:main',
        ],
    },
)
