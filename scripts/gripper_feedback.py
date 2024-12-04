#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray
from srbl.dynamixel import DynamixelRobot
import numpy as np

class Robot:
    def __init__(self, MY_DXL: str= 'PRO_SERIES'):
        self._robot = DynamixelRobot(MY_DXL)
        self._robot.open()
        self._robot.enable_torque()
        rospy.sleep(1.0)
        
    def maximum_position(self) -> int:
        return self._robot.DXL_MAXIMUM_POSITION_VALUE

    def command_joints(self, pos_data: float) -> None:
        max_pos = self.maximum_position()
        pos_data_exp = int(max_pos * pos_data)
        # rospy.loginfo("Commanding the robot with value: %d", pos_data_exp)
        self._robot.set_joints(pos_data_exp)

    def close(self) -> None:
        self._robot.close()


class FeedbackLoop:
    def __init__(self):
        self.desired_force = 8  # Desired average force
        self.kp = 0.007  # Proportional gain
        self.ki = 0.002  # Integral gain
        self.kd = 0.0012  # Derivative gain

        self.integral = 0.0  # Integral term
        self.prev_error = 0.0  # Previous error
        self.grip_flag = 0  # Grip flag to indicate gripping state
        self.max_pos_data = 0.48  # Maximum allowable pos_data

        # Initialize sensor force variables
        self.force_R3 = 0.0
        self.force_L3 = 0.0

        # Subscribe to sensor_force topic
        self.force_subscriber = rospy.Subscriber('sensor_force', Float32MultiArray, self.force_callback)

    def force_callback(self, msg):
        if len(msg.data) >= 6:
            self.force_R3 = -1 * msg.data[2] # For the direction
            self.force_L3 = -1 * msg.data[5] # for the direction

    def feedback(self, pos_data: float) -> float:
        if pos_data > 0.2:  # Gripping intention detected
            self.grip_flag = 1
        elif pos_data < 0.3:  # Releasing intention detected
            self.grip_flag = 0

        if self.grip_flag == 1:
            # Calculate average force and error
            avg_force = (self.force_R3 + self.force_L3) / 2.0
            error = self.desired_force - avg_force

            # PID control
            self.integral += error
            derivative = error - self.prev_error
            self.prev_error = error

            adjustment = self.kp * error + self.ki * self.integral + self.kd * derivative
            new_pos_data = pos_data + adjustment

            # Ensure pos_data stays within bounds
            new_pos_data = max(0.0, min(new_pos_data, self.max_pos_data))
            rospy.loginfo("Feedback adjusted pos_data: %.3f, avg_force: %.3f, error: %.3f", new_pos_data, avg_force, error)
            return new_pos_data

        return pos_data  # No adjustment if not gripping


class GripperNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('gripper_srbl', anonymous=True)
        self.robot = Robot('PRO_SERIES')
        self.feedback_loop = FeedbackLoop()

        # Subscribe to the 'gello_data' topic
        self.subscriber = rospy.Subscriber("gello_replay", Float32MultiArray, self.callback)
        self.start_time = rospy.get_time()

    def callback(self, data) -> None:
        # Calculate elapsed time
        current_time = rospy.get_time()
        elapsed_time = current_time - self.start_time
        # Assuming data.data is a list and we are interested in the first element
        if len(data.data) > 0:
            pos_data = data.data[0]
            # rospy.loginfo("Received data: %f at time: %f seconds", pos_data, elapsed_time)

            new_pos_data = self.feedback_loop.feedback(pos_data)
            rospy.loginfo("Received data: %f at time: %f seconds", new_pos_data, elapsed_time)
            self.robot.command_joints(new_pos_data)

        else:
            rospy.logwarn("Received empty data")

    def listen(self) -> None:
        rospy.spin()
    def shutdown(self) -> None:
        """ Safely close the connection when shutting down. """
        self.robot.close()


if __name__ == '__main__':
    try:
        node = GripperNode()
        node.listen()
    except rospy.ROSInterruptException:
        node.shutdown()
        rospy.loginfo("Shutting down gripper node.")

