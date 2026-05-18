from setuptools import setup

setup(
    name='nero_ai',
    version='0.0.1',
    packages=['nero_ai'],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'command_parser  = nero_ai.command_parser:main',
            'planning_node   = nero_ai.planning_node:main',
            'perception_node = nero_ai.perception_node:main',
        ],
    },
)
