#! /usr/bin/env python3
"""
ROS node that listen for a path in topic '/path' and follows it
"""

import threading

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA, Header, Empty
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker

MAX_LINEAR_VEL = 1  # m/s
MAX_ANGULAR_VEL = 2 * np.pi  # rad/s
P_L1, P_L2 = 0.2, 0
K1 = K2 = 1

global path_subscriber, twist_publisher, marker_publisher, goal_reached_publisher


class TrajectoryFollower:
    def __init__(self):
        self.current_traj = []
        self.new_traj = None
        self.lock = threading.Lock()
        self.thread = None
        self.dt = 0.5

    def traj_callback(self, path_msg: Path):
        ###################################
        #! Do the path interpolation here #
        ###################################

        # if the path is empty, we stop the robot
        if len(path_msg.poses) == 0:  # type: ignore
            with self.lock:
                self.dt = 0.5
                self.new_traj = []
            return

        positions = [pose.pose.position for pose in path_msg.poses]  # type: ignore
        checkpoints = [(position.x, position.y) for position in positions]

        # define some constants to do the interpolation of the path
        traj_max_speed = 0.2
        step_length = 0.01
        dt = step_length / traj_max_speed
        list_interpolated_points = []
        total_dist = 0.0

        for i in range(len(checkpoints) - 1):
            # get the two consecutive checkpoints
            checkpoint_1 = np.array(checkpoints[i])
            checkpoint_2 = np.array(checkpoints[i + 1])
            # interpolate between the two checkpoints
            dist = float(np.linalg.norm(checkpoint_1 - checkpoint_2))
            n_steps = int(dist / step_length)
            points = np.linspace(checkpoint_1, checkpoint_2, n_steps)
            # add the interpolated points to the list
            list_interpolated_points.extend(points)
            total_dist += dist

        # add the last checkpoint to the list
        list_interpolated_points.append(np.array(checkpoints[-1]))

        # update the current trajectory
        with self.lock:
            self.new_traj = list_interpolated_points
            self.dt = dt

    def _traj_control_task(self):
        # init tf wait till available
        listener = tf.TransformListener()
        listener.waitForTransform("/map", "/base_link", rospy.Time(0), rospy.Duration(secs=5))

        idx_current_point = 0
        while not rospy.is_shutdown():
            with self.lock:
                if self.new_traj is not None:
                    # Acknowledge the new trajectory
                    self.current_traj = self.new_traj
                    self.new_traj = None
                    idx_current_point = 0  # reset the index of the current point
                    rospy.logwarn(
                        f"Following new path (dt={self.dt:.2f}s), {len(self.current_traj)} points"
                    )
                    # if the new trajectory is empty, stop the robot
                    if len(self.current_traj) == 0:
                        twist_msg = Twist(
                            linear=Vector3(x=0, y=0, z=0),
                            angular=Vector3(x=0, y=0, z=0),
                        )
                        twist_publisher.publish(twist_msg)
                    continue

            if idx_current_point < len(self.current_traj) - 1:
                try:
                    (trans, rot) = listener.lookupTransform("/map", "/base_link", rospy.Time(0))
                except (
                    tf.LookupException,
                    tf.ConnectivityException,
                    tf.ExtrapolationException,
                ):
                    rospy.logwarn("Could not get the robot pose")
                    continue

                # extract trans value
                x, y, _ = trans
                theta = euler_from_quaternion(rot)[2]
                
                
                rot_theta = np.array(
                    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
                )
                x_r = self.current_traj[idx_current_point][0]
                y_r = self.current_traj[idx_current_point][1]
                
                x_r_next = self.current_traj[idx_current_point + 1][0]
                y_r_next = self.current_traj[idx_current_point + 1][1]
                
                dotx_r = (x_r_next - x_r) / self.dt
                doty_r = (y_r_next - y_r) / self.dt

                # Commande
                p_ncontraint = np.array([x, y]) + rot_theta @ np.array([P_L1, P_L2])

                x_P = p_ncontraint[0]
                y_P = p_ncontraint[1]

                v1 = dotx_r - K1 * (x_P - x_r)
                v2 = doty_r - K2 * (y_P - y_r)

                B = np.array(
                    [[np.cos(theta), -P_L1 * np.sin(theta)],
                     [np.sin(theta), P_L1 * np.cos(theta)]]
                )

                cmd_vec = np.linalg.pinv(B) @ np.array([v1, v2]).T
                u1 = cmd_vec[0]
                u2 = cmd_vec[1]

                # Limit the velocities
                u1 = np.clip(u1, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
                u2 = np.clip(u2, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

                twist_msg = Twist(linear=Vector3(x=u1, y=0, z=0), angular=Vector3(x=0, y=0, z=u2))
                twist_publisher.publish(twist_msg)

                marker_radius = 0.05
                marker_period = 2
                if idx_current_point % marker_period == 0:
                    marker_P_nc = Marker(
                        type=Marker.SPHERE,
                        id=0,
                        scale=Vector3(x=marker_radius, y=marker_radius, z=marker_radius),
                    )
                    marker_P_nc.header = Header(frame_id="map", stamp=rospy.Time.now())
                    marker_P_nc.lifetime = rospy.Duration(
                        secs=0, nsecs=int(self.dt * marker_period * 1e9)
                    )
                    marker_P_nc.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1)
                    marker_P_nc.pose.position = Point(x=x_P, y=y_P, z=0.05)
                    marker_P_nc.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)

                    marker_publisher.publish(marker_P_nc)

                if idx_current_point % marker_period == 1:
                    marker_ref_point = Marker(
                        type=Marker.SPHERE,
                        id=1,
                        scale=Vector3(x=marker_radius, y=marker_radius, z=marker_radius),
                    )
                    marker_ref_point.header = Header(frame_id="map", stamp=rospy.Time.now())
                    marker_ref_point.lifetime = rospy.Duration(
                        secs=0, nsecs=int(self.dt * marker_period * 1e9)
                    )
                    marker_ref_point.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
                    marker_ref_point.pose.position = Point(x=x_r, y=y_r, z=0.1)
                    marker_ref_point.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)

                    marker_publisher.publish(marker_ref_point)

                idx_current_point += 1

                # Make sure that the robot stops when the path is finished
                if idx_current_point == len(self.current_traj) - 1:
                    twist_msg = Twist(linear=Vector3(x=0, y=0, z=0), angular=Vector3(x=0, y=0, z=0))
                    twist_publisher.publish(twist_msg)
                    rospy.loginfo("Trajectory finished")
                    goal_reached_publisher.publish(Empty())

            try:
                rospy.sleep(self.dt)
            except rospy.ROSInterruptException:
                return

    def start(self):
        self.thread = threading.Thread(target=self._traj_control_task)
        self.thread.start()


def main():
    rospy.init_node("traj_follow_node")
    global path_subscriber, twist_publisher, marker_publisher, goal_reached_publisher
    twist_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    follower = TrajectoryFollower()
    follower.start()
    path_subscriber = rospy.Subscriber("/path", Path, follower.traj_callback, queue_size=1)
    marker_publisher = rospy.Publisher("/visualization_marker", Marker, queue_size=1)
    goal_reached_publisher = rospy.Publisher("/goal_reached", Empty, queue_size=1)

    rospy.spin()

    # wait for the thread to finish
    follower.thread.join()


if __name__ == "__main__":
    main()
