from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nero_ai',
            executable='perception_node',
            name='perception_node',
            output='screen',
        ),
        Node(
            package='nero_ai',
            executable='planning_node',
            name='planning_node',
            output='screen',
        ),
        Node(
            package='nero_ai',
            executable='mcp_robot_server',
            name='mcp_robot_server',
            output='screen',
        ),
        Node(
            package='nero_ai',
            executable='command_parser',
            name='command_parser',
            output='screen',
        ),
    ])
