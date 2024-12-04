#!/usr/bin/env python

import sys
import os
import rospy
from std_msgs.msg import Float32MultiArray
from srbl.dynamixel import DynamixelRobot
import numpy as np

class SRBLGelloNode:
    def __init__(self):
        rospy.init_node('gello_srbl', anonymous=True)

        self.gello_pub = rospy.Publisher('gello_data', Float32MultiArray, queue_size=10)
        self.gello_open_close = (2020, 970)
        self.gello_robot = DynamixelRobot(MY_DXL='X_SERIES')
        self.gello_robot.open()

        self.rate = rospy.Rate(100)  # 100 Hz

    def publish_data(self):
        """ Continuously read gello data and publish to a ROS topic. """
        while not rospy.is_shutdown():
            
            joint = self.gello_robot.get_joints() # Read data from the FingerSensor

            gello_msg = Float32MultiArray()
            joint_norm = (joint - self.gello_open_close[0]) / (self.gello_open_close[1] - self.gello_open_close[0])
            joint_norm = min(max(0, joint_norm), 1)
            gello_msg.data = [joint_norm,]  # gello is a list of floats

            self.gello_pub.publish(gello_msg)
            self.rate.sleep()

    def shutdown(self):
        """ Safely close the connection when shutting down. """
        self.gello_robot.close()

if __name__ == '__main__':
    try:
        node = SRBLGelloNode()
        node.publish_data()
    except rospy.ROSInterruptException:
        node.shutdown()
        rospy.loginfo("Shutting down srbl_gello node.")
