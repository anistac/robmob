#!/usr/bin/env python3
import sys
import os
import rospy
from std_msgs.msg import Empty, ColorRGBA, Header
from geometry_msgs.msg import Twist, Vector3, PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker
from nav_msgs.srv import GetMap
import cv2
import tf
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.image_world_transform import GridWorldTransform

OCCUPANCY_THRESH = 0.8


def pub_new_explo_target():
    ## ROUTINE
    rospy.wait_for_service("/dynamic_map")
    try:
        get_map = rospy.ServiceProxy("/dynamic_map", GetMap)
        map = get_map().map
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return None
    map_data = np.asarray(map.data).reshape(map.info.height, map.info.width).T

    map_data = np.where(map_data > OCCUPANCY_THRESH, 255, map_data)
    map_data = np.where((0 < map_data) & (map_data <= OCCUPANCY_THRESH), 0, map_data)
    map_data = np.where(map_data == -1, 128, map_data)
    map_data = map_data.astype(np.uint8)

    # cv2.imshow("before uwu", map_data)

    unexplored_mask = np.where(map_data == 128, 255, 0).astype(np.uint8)
    free_mask = np.where(map_data == 0, 255, 0).astype(np.uint8)
    obstacle_mask = np.where(map_data == 255, 255, 0).astype(np.uint8)

    # Filter masks to remove noise.
    unexplored_mask = cv2.morphologyEx(
        unexplored_mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8)
    )
    free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    obstacle_mask = cv2.morphologyEx(
        obstacle_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )
    # cv2.imshow("unexplored", unexplored_mask)
    # cv2.imshow("free", free_mask)
    # cv2.imshow("obstacle", obstacle_mask)

    # Dilating masks to expand boundary.
    kernel = np.ones((3, 3), dtype=np.uint8)
    unexplored_mask = cv2.dilate(unexplored_mask, kernel, iterations=1)
    free_mask = cv2.dilate(free_mask, kernel, iterations=1)
    obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)

    # Required points now will have both color's mask val as 255.
    frontier = np.where(
        (free_mask == 255) & (unexplored_mask == 255) & (obstacle_mask == 0), 255, 0
    ).astype(np.uint8)

    # detect lines in the resulting image
    height, width = frontier.shape
    skel = np.zeros([height, width], dtype=np.uint8)  # [height,width,3]
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while np.count_nonzero(frontier) != 0:
        eroded = cv2.erode(frontier, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(frontier, temp)
        skel = cv2.bitwise_or(skel, temp)
        frontier = eroded.copy()

    # cv2.imshow("skel", skel)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    line_image = np.copy(skel) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(
        skel, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
    )
    lines = [line[0] for line in lines]
    for x1, y1, x2, y2 in lines:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    # cv2.imshow("lines", line_image)

    # process to get the perpendicular of each line
    perpendiculars_angle = [np.arctan2(y2 - y1, x2 - x1) + np.pi / 2 for x1, y1, x2, y2 in lines]
    centers = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in lines]
    dist_to_center = 15
    # get the perpendicular points (points at a distance of dist_to_center from the center
    # and in the direction of the perpendicular)
    perpendicular_points = []
    for i, (center, perpendicular) in enumerate(zip(centers, perpendiculars_angle)):
        x, y = center
        dx = dist_to_center * np.cos(perpendicular)
        dy = dist_to_center * np.sin(perpendicular)
        perpendicular_points.append((int(x + dx), int(y + dy)))
        perpendicular_points.append((int(x - dx), int(y - dy)))

    # filter points outside image and those that are not in free space
    filtered_perpendicular_points = [
        (x, y)
        for x, y in perpendicular_points
        if (0 <= x < height and 0 <= y < width) and map_data[y, x] == 0
    ]
    for x, y in filtered_perpendicular_points:
        cv2.circle(line_image, (x, y), 1, 255, -1)
    # cv2.imshow("perpendiculars", line_image)

    # Get robot position
    copy_map_data = map_data.copy()
    ((robot_x, robot_y, _), rot) = tf_listener.lookupTransform("/map", "/base_link", rospy.Time(0))
    # Conver to pixel coordinates
    img_world_transform = GridWorldTransform()
    img_world_transform.update(map.info)
    robot_pixel = img_world_transform.world_to_grid(robot_x, robot_y)

    # compute closest point to robot
    closest_point = min(
        filtered_perpendicular_points,
        key=lambda point: np.linalg.norm(np.array(point) - np.array(robot_pixel)),
    )
    cv2.circle(copy_map_data, tuple(closest_point), 3, 255, -1)

    # convert closest point to world coordinates
    closest_point = img_world_transform.grid_to_world(*closest_point)
    return closest_point


def main():
    rospy.init_node("explo_node")
    rospy.loginfo("explo_node started")
    
    global cmd_vel_publisher
    cmd_vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    goal_publisher = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
    marker_publisher = rospy.Publisher("/visualization_marker", Marker, queue_size=1)
    
    global tf_listener
    tf_listener = tf.TransformListener()
    tf_listener.waitForTransform(
        "/map",
        "/base_link",
        rospy.Time(0),
        rospy.Duration(secs=5),
        rospy.Duration(nsecs=10e6),  # 50ms
    )
    
    rospy.sleep(2)
    rot_360_msg = Twist(linear=Vector3(x=0.0, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0,z=.3))
    stop_msg = Twist(linear=Vector3(x=0.0, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0,z=0.0))
    rospy.loginfo(f"pub mdg {rot_360_msg}")
    cmd_vel_publisher.publish(rot_360_msg)
    rospy.sleep(10)
    cmd_vel_publisher.publish(stop_msg)
    
    # Small delay to be sure the controller is ready
    rospy.sleep(5)
    

    while not rospy.is_shutdown():
        target = pub_new_explo_target()
        rospy.loginfo(f"New target: {target}")

        # send new goal to follow
        pose = Pose(position=Point(x=target[1], y=target[0]), orientation=Quaternion())
        pose_stamped = PoseStamped(pose=pose, header=rospy.Header(frame_id="map"))
        goal_publisher.publish(pose_stamped)
        # Display goal in rviz
        marker = Marker(type=Marker.ARROW, id=0)
        marker.header = Header(frame_id="map")
        marker.points = [Point(x=target[1], y=target[0], z=1), Point(x=target[1], y=target[0], z=0)]
        marker.scale = Vector3(x=0.2, y=0.4, z=0.4)
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        marker.lifetime = rospy.Duration(secs=10)
        marker_publisher.publish(marker)
        
        try:
            rospy.wait_for_message("/goal_reached", Empty)
        except rospy.ROSInterruptException:
            pass
        


if __name__ == "__main__":
    main()
