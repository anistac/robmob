#! /usr/bin/env python3
import sys
import numpy as np

import rospy
import subprocess
import tf
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import Joy

# LINEAR_SPEED_LIMIT = 0.3
# ANGULAR_SPEED_LIMIT = np.pi / 2
LINEAR_SPEED_LIMIT = 5
ANGULAR_SPEED_LIMIT = np.pi 


class Teleop:
    def __init__(self):
        self._twist_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self._joystick_sub = rospy.Subscriber("joy", Joy, self._callback)
        self._goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self._expl_subprocess = None
        self._robot_tf = tf.TransformListener()
        self._robot_tf.waitForTransform("/map", "/base_link", rospy.Time(0), rospy.Duration(secs=5))

    def _callback(self, joy_msg):
        twist_msg = Twist()
        twist_msg.linear.x = np.clip(joy_msg.axes[1], -LINEAR_SPEED_LIMIT, LINEAR_SPEED_LIMIT)
        twist_msg.angular.z = np.clip(joy_msg.axes[0], -ANGULAR_SPEED_LIMIT, ANGULAR_SPEED_LIMIT)
        self._twist_pub.publish(twist_msg)

        # if X is pressed return to the origin
        if joy_msg.buttons[0] == 1:
            rospy.loginfo("Returning to origin")
            pose = Pose(position=Point(x=0.0, y=0.0), orientation=Quaternion())
            pose_stamped = PoseStamped(pose=pose, header=rospy.Header(frame_id="map"))
            self._goal_pub.publish(pose_stamped)

        # if triangle is pressed run the exploration node
        if joy_msg.buttons[2] == 1:
            if self._expl_subprocess is not None:
                rospy.logwarn("Exploration node already running, cannot start another one")
            else:
                rospy.loginfo("Exploration node started")
                self._expl_subprocess = subprocess.Popen(["rosrun", "explo_node", "explo_node.py"])

        # if circle is pressed stop the exploration node
        if joy_msg.buttons[1] == 1:
            if self._expl_subprocess is None:
                rospy.logwarn("No exploration node running")
            else:
                self._expl_subprocess.terminate()
                self._expl_subprocess = None
                # stop current traj by sending goal to robot position
                trans, rot = self._robot_tf.lookupTransform("/map", "/base_link", rospy.Time(0))
                pose = Pose(position=Point(x=trans[0], y=trans[1]), orientation=Quaternion())
                pose_stamped = PoseStamped(pose=pose, header=rospy.Header(frame_id="map"))
                self._goal_pub.publish(pose_stamped)
                rospy.loginfo("Exploration node stopped")


if __name__ == "__main__":
    rospy.init_node("teleop")
    try:
        teleop = Teleop()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)
