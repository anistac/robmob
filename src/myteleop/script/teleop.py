import rospy


class Teleop:
    def __init__(self):
        self._pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self._sub = rospy.Subscriber('joy', Joy, self._callback)
        self._twist = Twist()

    def _callback(self, joy_msg):
        self._twist.linear.x = joy_msg.axes[1]
        self._twist.angular.z = joy_msg.axes[0]

    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self._pub.publish(self._twist)
            rate.sleep()
