from setuptools import setup

package_name = 'franka_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andrew',
    description='Perception for Franka sorting',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'block_detector = franka_perception.block_detector:main',
            'block_listener = franka_perception.block_listener:main'
        ],
    },
)
