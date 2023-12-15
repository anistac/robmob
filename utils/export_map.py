#! /usr/bin/env python3

import os
import sys
import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
import numpy as np
import cv2
import pathlib


"""
This script exports the current map to a png file (by calling the /dynamic_map service).
"""

def main():
    rospy.init_node('export_map', anonymous=True)

    rospy.wait_for_service('/dynamic_map')
    
    try:
        get_map = rospy.ServiceProxy('/dynamic_map', GetMap)
        map = get_map().map
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

    map_data = np.array(map.data).reshape((map.info.height, map.info.width))
    map_data = np.where(map_data == -1, 0.5, map_data)

    # Rescale from 0-1 to 0-255 and convert to uint8
    map_data = map_data * 255
    map_data = map_data.astype(np.uint8)

    # write img to file with date and time
    now_ros = rospy.get_rostime().to_sec()
    img_name = "map_" + str(now_ros) + ".png"
    parent_dir = pathlib.Path(__file__).parent.absolute()

    # Create map_images folder if it doesn't exist
    if not os.path.exists(os.path.join(parent_dir, "map_images")):
        os.makedirs(os.path.join(parent_dir, "map_images"))
    img_name = os.path.join(parent_dir, "map_images", img_name)

    # Write img to file
    cv2.imwrite(img_name, map_data)
    print(f'Img written to: {img_name}')

if __name__ == "__main__":
    main()