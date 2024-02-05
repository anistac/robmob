#!/usr/bin/env python3
import sys
import os
import pathlib
import math
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

    desktop_folder = os.path.join(pathlib.Path.home(), "Desktop", "explo_node_export")
    file_name = os.path.join(desktop_folder, "map_raw.png")
    cv2.imwrite(file_name, map_data)

    unexplored_mask = np.where(map_data == 128, 255, 0).astype(np.uint8)
    free_mask = np.where(map_data == 0, 255, 0).astype(np.uint8)
    obstacle_mask = np.where(map_data == 255, 255, 0).astype(np.uint8)

    # Save raw masks
    stack = cv2.hconcat([unexplored_mask, free_mask, obstacle_mask])
    file_name = os.path.join(desktop_folder, "masks_raw.png")
    cv2.imwrite(file_name, stack)

    # Filter masks to remove noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    unexplored_mask = cv2.morphologyEx(unexplored_mask, cv2.MORPH_OPEN, kernel)
    free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    obstacle_mask = cv2.morphologyEx(
        obstacle_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )
    # Save filtered masks
    stack = cv2.hconcat([unexplored_mask, free_mask, obstacle_mask])
    file_name = os.path.join(desktop_folder, "masks_filtered.png")
    cv2.imwrite(file_name, stack)
    
    # Dilating masks to expand boundary.
    unexplored_mask = cv2.dilate(unexplored_mask, kernel, iterations=1)
    free_mask = cv2.dilate(free_mask, kernel, iterations=1)
    obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)

    # save dilated masks
    stack = cv2.hconcat([unexplored_mask, free_mask, obstacle_mask])
    file_name = os.path.join(desktop_folder, "masks_dilated.png")
    cv2.imwrite(file_name, stack)
    
    # Required points now will have both color's mask val as 255.
    frontiers = np.where(
        (free_mask == 255) & (unexplored_mask == 255) & (obstacle_mask == 0), 255, 0
    ).astype(np.uint8)

    # save the frontier
    file_name = os.path.join(desktop_folder, "frontiers.png")
    cv2.imwrite(file_name, frontiers)

    # detect lines in the resulting image
    height, width = frontiers.shape
    skel = np.zeros([height, width], dtype=np.uint8)  # [height,width,3]
    while np.count_nonzero(frontiers) != 0:
        eroded = cv2.erode(frontiers, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(frontiers, temp)
        skel = cv2.bitwise_or(skel, temp)
        frontiers = eroded.copy()

    # save the skeleton
    file_name = os.path.join(desktop_folder, "skeleton.png")
    cv2.imwrite(file_name, skel)

    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 100  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 2  # maximum gap in pixels between connectable line segments
    line_image = np.copy(skel) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(
        skel, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
    )
    lines = [line[0] for line in lines]
    for x1, y1, x2, y2 in lines:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # save the lines
    file_name = os.path.join(desktop_folder, "lines.png")
    cv2.imwrite(file_name, line_image)

    # process to get the perpendicular of each line
    perpendiculars_angle = [np.arctan2(y2 - y1, x2 - x1) + np.pi / 2 for x1, y1, x2, y2 in lines]
    centers = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in lines]
    # get the perpendicular points (points at a distance of dist_to_center from the center
    # and in the direction of the perpendicular)
    robot_radius = 0.5  # meters
    robot_radius_px = math.ceil(robot_radius / map.info.resolution)
    perpendicular_points = []
    for i, (center, perpendicular) in enumerate(zip(centers, perpendiculars_angle)):
        x, y = center
        dx = robot_radius_px * 1.2 * np.cos(perpendicular)
        dy = robot_radius_px * 1.2 * np.sin(perpendicular)
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

    # save the filtered perpendicular points
    file_name = os.path.join(desktop_folder, "filtered_perpendicular_points.png")
    cv2.imwrite(file_name, line_image)

    # Get robot position
    ((robot_x, robot_y, _), rot) = tf_listener.lookupTransform("/map", "/base_link", rospy.Time(0))
    # Conver to pixel coordinates
    img_world_transform = GridWorldTransform()
    img_world_transform.update(map.info)
    robot_pixel = img_world_transform.world_to_grid(robot_x, robot_y)
    robot_pixel_reverse = (robot_pixel[1], robot_pixel[0])

    # compute closest point to robot
    sorted_points = sorted(
        filtered_perpendicular_points,
        key=lambda point: np.linalg.norm(np.array([point[1], point[0]]) - np.array(robot_pixel)),
    )
    # find the closest point that is in line of sight (if any)
    map_filtered = filtered_occ_map(map, robot_pixel)
    chosen_point = sorted_points[0]
    for closest_point in sorted_points:
        if all(map_filtered[y, x] == 0 for x, y in bresenham(robot_pixel, closest_point)):
            chosen_point = closest_point
            break

    # draw robot and closest point
    copy_map_data = cv2.cvtColor(map_data, cv2.COLOR_GRAY2RGB)
    cv2.circle(copy_map_data, tuple(robot_pixel), 3, (0, 0, 255), -1)
    cv2.circle(copy_map_data, tuple(chosen_point), 3, (0, 255, 0), -1)
    file_name = os.path.join(desktop_folder, "robot_and_closest_point.png")
    cv2.imwrite(file_name, copy_map_data)

    # convert closest point to world coordinates
    chosen_point_world = img_world_transform.grid_to_world(*chosen_point)
    return chosen_point_world


def filtered_occ_map(map, robot_pixel):
    map_data = np.asarray(map.data, dtype=np.int8).reshape(map.info.height, map.info.width).T
    unknown_mask = np.where(map_data == -1, 1, 0).astype(np.uint8)
    map_data[map_data > OCCUPANCY_THRESH] = 1
    map_data[map_data <= OCCUPANCY_THRESH] = 0
    map_data = np.where(map_data == -1, 1, map_data)
    map_data = map_data.astype(np.uint8)

    # open the unknown areas to remove artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    unknown_mask = cv2.morphologyEx(unknown_mask, cv2.MORPH_OPEN, kernel)
    # apply mask to the map
    map_data = np.where(unknown_mask == 1, 1, map_data)
    map_data_copy = cv2.cvtColor(map_data * 255, cv2.COLOR_GRAY2RGB)

    # Small patch to the map to avoid situations where the robot start position is in an obstacle
    # Add circle around the robot position
    robot_radius = 0.5  # meters
    robot_radius_px = math.ceil(robot_radius / map.info.resolution)
    cv2.circle(map_data, (robot_pixel[1], robot_pixel[0]), robot_radius_px, 0, -1)
    map_data_copy = cv2.cvtColor(map_data * 255, cv2.COLOR_GRAY2RGB)
    cv2.circle(map_data_copy, (robot_pixel[1], robot_pixel[0]), 4, (0, 0, 255), -1)

    # Make a dillation of the map to take into account the robot size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    map_data = cv2.dilate(map_data, kernel, iterations=robot_radius_px)

    return map_data


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

    rot_360_msg = Twist(linear=Vector3(x=0.0, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=np.pi / 2))
    stop_msg = Twist(linear=Vector3(x=0.0, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=0.0))

    # Small delay to be sure the controller is ready
    rospy.sleep(5)

    while not rospy.is_shutdown():
        rospy.loginfo("Rotating 360 degrees")
        cmd_vel_publisher.publish(rot_360_msg)
        rospy.sleep(4)
        cmd_vel_publisher.publish(stop_msg)
        
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


def bresenham(p1, p2):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).

    Input coordinates should be integers.

    The result will contain both the start and the end point.
    """
    (x0, y0), (x1, y1) = p1, p2
    dx, dy = (x1 - x0), (y1 - y0)

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx, dy = abs(dx), abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2 * dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x * xx + y * yx, y0 + x * xy + y * yy
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy


if __name__ == "__main__":
    main()
