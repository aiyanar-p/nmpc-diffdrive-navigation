#!/usr/bin/env python3
"""
Brings up the full NMPC navigation stack in Gazebo: simulator + world, robot
description and ROS/Gazebo bridge, slam_toolbox (LiDAR SLAM), the obstacle
detector, A* global planner, NMPC controller, and RViz.

Usage:
  ros2 launch nmpc_robot_nav sim_nmpc.launch.py
  ros2 launch nmpc_robot_nav sim_nmpc.launch.py goal_x:=10.0 goal_y:=4.0
"""

import os
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from lifecycle_msgs.msg import Transition


def generate_launch_description():

    pkg_name       = 'nmpc_robot_nav'
    pkg_share      = FindPackageShare(pkg_name).find(pkg_name)
    pkg_ros_gz_sim = FindPackageShare('ros_gz_sim').find('ros_gz_sim')

    # File paths
    world_file    = os.path.join(pkg_share, 'worlds', 'obstacle_course.world')
    urdf_file     = os.path.join(pkg_share, 'urdf',   'diff_robot_lidar.urdf.xacro')
    bridge_config = os.path.join(pkg_share, 'config', 'ros_gz_bridge.yaml')
    params_file   = os.path.join(pkg_share, 'config', 'nmpc_params.yaml')
    slam_params   = os.path.join(pkg_share, 'config', 'slam_params.yaml')
    rviz_config   = os.path.join(pkg_share, 'rviz',   'nmpc_view.rviz')

    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use Gazebo simulated clock')
    goal_x_arg = DeclareLaunchArgument(
        'goal_x', default_value='13.0',
        description='Goal X position (m)')
    goal_y_arg = DeclareLaunchArgument(
        'goal_y', default_value='0.0',
        description='Goal Y position (m)')

    use_sim_time = LaunchConfiguration('use_sim_time')
    goal_x       = LaunchConfiguration('goal_x')
    goal_y       = LaunchConfiguration('goal_y')

    # Let Gazebo find the world and meshes
    gz_resource = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(pkg_share, 'worlds'))

    # 1. Gazebo + world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    # 2. Robot description (URDF via xacro)
    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]), value_type=str)

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
    )

    # 3. ROS <-> Gazebo bridge (cmd_vel, odom, tf, clock, scan)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        arguments=['--ros-args', '-p', f'config_file:={bridge_config}'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # 4. Spawn robot at the start pose
    spawn_robot = TimerAction(
        period=3.0,
        actions=[Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_diff_robot',
            output='screen',
            arguments=[
                '-name',  'diff_robot',
                '-topic', '/robot_description',
                '-x', '0.0', '-y', '0.0', '-z', '0.05',
                '-R', '0.0', '-P', '0.0', '-Y', '0.0',
            ],
        )],
    )

    # 4b. Alias Gazebo's scoped lidar frame to lidar_link.
    # Gazebo Harmonic ignores the sensor's <frame_id> and stamps scans with the
    # scoped name 'diff_robot/base_link/lidar', which slam_toolbox can't resolve.
    lidar_frame_alias = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_frame_alias',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0',
                   'lidar_link', 'diff_robot/base_link/lidar'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # 5. SLAM Toolbox (async online mapping).
    # Driven as a lifecycle node: configure, then activate on the transition.
    slam_node = LifecycleNode(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        namespace='',
        parameters=[slam_params, {
            'use_sim_time': use_sim_time,
            'use_lifecycle_manager': False,
        }],
    )

    slam_configure_event = TimerAction(
        period=5.0,
        actions=[EmitEvent(
            event=ChangeState(
                lifecycle_node_matcher=matches_action(slam_node),
                transition_id=Transition.TRANSITION_CONFIGURE,
            )
        )],
    )

    slam_activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_node,
            start_state='configuring',
            goal_state='inactive',
            entities=[
                LogInfo(msg='[SLAM] Configured — activating slam_toolbox'),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_node),
                    transition_id=Transition.TRANSITION_ACTIVATE,
                )),
            ],
        )
    )

    # 6. Obstacle detector — starts after SLAM so map->odom TF exists
    obstacle_detector = TimerAction(
        period=8.0,
        actions=[Node(
            package=pkg_name,
            executable='obstacle_detector',
            name='obstacle_detector',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        )],
    )

    # 7. NMPC controller
    nmpc_controller = TimerAction(
        period=8.0,
        actions=[Node(
            package=pkg_name,
            executable='nmpc_controller',
            name='nmpc_controller',
            output='screen',
            parameters=[
                params_file,
                {
                    'use_sim_time': use_sim_time,
                    'goal_x': goal_x,
                    'goal_y': goal_y,
                },
            ],
        )],
    )

    # 7b. Global planner — A* on the inflated /map, publishes /global_path
    global_planner = TimerAction(
        period=8.0,
        actions=[Node(
            package=pkg_name,
            executable='global_planner',
            name='global_planner',
            output='screen',
            parameters=[
                params_file,
                {
                    'use_sim_time': use_sim_time,
                    'goal_x': goal_x,
                    'goal_y': goal_y,
                },
            ],
        )],
    )

    # 8. RViz2
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        ros_arguments=['--log-level', 'error'],  # suppress TF_OLD_DATA spam
    )

    return LaunchDescription([
        use_sim_time_arg,
        goal_x_arg,
        goal_y_arg,
        gz_resource,
        gazebo,
        robot_state_publisher,
        bridge,
        lidar_frame_alias,
        spawn_robot,
        slam_node,
        slam_configure_event,
        slam_activate_event,
        obstacle_detector,
        nmpc_controller,
        global_planner,
        rviz,
    ])
