from setuptools import setup
import os
from glob import glob

package_name = 'nero_ai'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'torch',
        'transformers',
        'pillow',
        'mcp',
        'google-genai',
        'python-dotenv',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    zip_safe=True,
    maintainer='nero',
    maintainer_email='nero@example.com',
    description='AgileX Nero pick-and-place with MCP + MoonDream2 + Gemini',
    license='MIT',
    entry_points={
        'console_scripts': [
            'command_parser   = nero_ai.command_parser:main',
            'mcp_robot_server = nero_ai.mcp_robot_server:main',
            'planning_node    = nero_ai.planning_node:main',
            'perception_node  = nero_ai.perception_node:main',
        ],
    },
)
