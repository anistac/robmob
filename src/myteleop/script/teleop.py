#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import sys

class Teleop:
    def __init__(self):
        self._pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self._sub = rospy.Subscriber('joy', Joy, self._callback)

    def _callback(self, joy_msg):
        new_twist = Twist()
        new_twist.linear.x = joy_msg.axes[1]
        new_twist.angular.z = joy_msg.axes[0]
        self._pub.publish(new_twist)


    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('teleop')
    try:
        teleop = Teleop()
        teleop.spin()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)