from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('detections_topic', default_value='/pothole/detections'),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('confidence_threshold', default_value='0.25'),
        DeclareLaunchArgument('dashboard_host', default_value='127.0.0.1'),
        DeclareLaunchArgument('dashboard_port', default_value='8080'),
        Node(
            package='pothole_detection',
            executable='pothole_detector',
            name='pothole_detector',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'detections_topic': LaunchConfiguration('detections_topic'),
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'dashboard_host': LaunchConfiguration('dashboard_host'),
                'dashboard_port': LaunchConfiguration('dashboard_port'),
            }],
        ),
    ])
