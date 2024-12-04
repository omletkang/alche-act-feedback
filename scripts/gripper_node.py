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
        rospy.loginfo("Commanding the robot with value: %d", pos_data_exp)
        self._robot.set_joints(pos_data_exp)

    def close(self) -> None:
        self._robot.close()

class GripperNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('gripper_srbl', anonymous=True)
        self.robot = Robot('PRO_SERIES')

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
            rospy.loginfo("Received data: %f at time: %f seconds", pos_data, elapsed_time)
            self.robot.command_joints(pos_data)

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

