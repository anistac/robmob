#! /usr/bin/env python3
import os
import pathlib
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rospy
import tf
import tf2_ros
import tf.transformations as tft
from geometry_msgs.msg import Point, Pose, PoseStamped
from matplotlib.animation import FuncAnimation
from nav_msgs.msg import Path
from nav_msgs.srv import GetMap
from std_msgs.msg import Header

# syspath magic bcz ros 🥴
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.bit_star import BITStar, BitStarVisualizer
from scripts.image_world_transform import GridWorldTransform

OCCUPANCY_THRESH = 0.8
global path_publisher, tf_listener


def handle_path_finding(pose_stamped_msg):
    
    # send empty path to be sure that the robot stops
    path_msg = Path(header=Header(frame_id="/map"), poses=[])
    path_publisher.publish(path_msg)
    
    # get goal and robot position
    goal_position = pose_stamped_msg.pose.position

    rospy.wait_for_service("/dynamic_map")

    try:
        get_map = rospy.ServiceProxy("/dynamic_map", GetMap)
        map = get_map().map
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return None

    print(f"\tMap width: {map.info.width}, height: {map.info.height}")
    print(f"\tMap resolution: {map.info.resolution:.2g}")
    print(
        f"\tMap origin: {map.info.origin.position.x:.2g}, {map.info.origin.position.y:.2g}"
    )

    (trans, rot) = tf_listener.lookupTransform("/map", "/base_link", rospy.Time(0))
    transform_matrix = tf.TransformerROS().fromTranslationRotation(trans, rot)
    P_robot_ref = np.array([0.2, 0, 0, 1]).T
    P_robot = transform_matrix @ P_robot_ref

    robot_x = P_robot[0]
    robot_y = P_robot[1]
    print(f"\tRobot position: {robot_x:.2g}, {robot_y:.2g}")
    print(f"\tGoal position: {goal_position.x:.2g}, {goal_position.y:.2g}")
    img_world_transform = GridWorldTransform()
    img_world_transform.update(map.info)

    goal_pixel = img_world_transform.world_to_grid(goal_position.x, goal_position.y)
    robot_pixel = img_world_transform.world_to_grid(robot_x, robot_y)
    print(f"\tRobot pos in pixel: {robot_pixel}")

    # reshape, normalize and apply threshold
    map_data = (
        np.asarray(map.data, dtype=np.int8).reshape(map.info.height, map.info.width).T
    )
    map_data = np.where(map_data == -1, 1, map_data)

    map_data[map_data > OCCUPANCY_THRESH] = 1
    map_data[map_data <= OCCUPANCY_THRESH] = 0
    map_data = map_data.astype(np.uint8)
    
    # Apply median blur to the map to remove noise
    map_data = cv2.medianBlur(map_data, 3)
    
    # save the map to a file
    unique_file_name = f'map_{rospy.Time.now().to_nsec()}.png'
    map_file = os.path.join(pathlib.Path.home(),"debug_maps" , unique_file_name)

    # Make a dillation of the map to take into account the robot size
    robot_radius = 0.75  # meters
    kernel_size = int(robot_radius / map.info.resolution)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    map_data = cv2.dilate(map_data, kernel, iterations=1)
    
    # Small patch to the map to avoid situations where the robot start position is in an obstacle
    # Add circle around the robot position
    cv2.circle(map_data, tuple(robot_pixel), kernel_size, 0, -1)

    path_finder = BITStar(
        occupancy_grid=map_data,
        start=robot_pixel,
        goal=goal_pixel,
        batch_size=20,
        max_batch=30,
        eta=5,
        prune_threshold=0.95,
        first_batch_factor=3,
        stop_on_goal=False,
        limit_secs=3,
    )
    # bit_star_visualizer = BitStarVisualizer(path_finder)
    path_px = path_finder.plan()

    # Visualize the path
    # anim = FuncAnimation(bit_star_visualizer.fig, bit_star_visualizer.update_plt, init_func=bit_star_visualizer.init_plt, frames=path_generator, repeat=False, interval=1000)
    # plt.show()

    # Convert path from pixel to map coordinates to Path message
    path = [img_world_transform.grid_to_world(p[0], p[1]) for p in path_px]
    # Create a Path message
    header = Header(frame_id="/map")
    path_msg = Path(header=header)
    path_msg.poses = []
    for p in path:
        pose = Pose(position=Point(x=p[0], y=p[1]))
        pose_stamped = PoseStamped(pose=pose, header=header)
        path_msg.poses.append(pose_stamped)

    path_publisher.publish(path_msg)


def path_finding_server():
    rospy.init_node("path_finding_server")
    rospy.loginfo("Path finding server started")

    global tf_listener
    tf_listener = tf.TransformListener()
    # wait for transform to be available
    tf_listener.waitForTransform(
        "/map", "/base_link", rospy.Time(0), rospy.Duration(secs=5), rospy.Duration(nsecs=10e6) # 50ms
    )

    rospy.Subscriber("/move_base_simple/goal", PoseStamped, handle_path_finding)

    # Create a publisher for the path
    global path_publisher
    path_publisher = rospy.Publisher("/path", Path, queue_size=1)

    rospy.spin()


if __name__ == "__main__":
    path_finding_server()
