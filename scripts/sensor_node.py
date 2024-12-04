#!/usr/bin/env python

# import sys
# import os
import rospy
from std_msgs.msg import Float32MultiArray
from srbl.sensor_srbl import FingerSensor


class SRBLSensorNode:
    def __init__(self):
        rospy.init_node('sensor_srbl', anonymous=True)

        self.sensor_pub = rospy.Publisher('sensor_data', Float32MultiArray, queue_size=10)

        self.finger_sensor = FingerSensor(port='/dev/ttyUSB0') # Check the port
        self.finger_sensor.open_serial()
        self.finger_sensor.initialize_offset()

        self.rate = rospy.Rate(100)  # 100 Hz

    def publish_sensor_data(self):
        """ Continuously read sensor1 data and publish to a ROS topic. """

        rospy.loginfo("Starting srbl_sensor_node.")

        while not rospy.is_shutdown():
            
            sensor1, sensor2 = self.finger_sensor.split_read() # Read np.ndarray data from the FingerSensor

            sensor_msg = Float32MultiArray()
            combined_data = list(sensor1) + list(sensor2)  # Concatenate sensor1 and sensor2 into one list
            sensor_msg.data = combined_data

            self.sensor_pub.publish(sensor_msg)
            self.rate.sleep()

    def shutdown(self):
        """ Safely close the connection when shutting down. """
        rospy.loginfo("Shutting down srbl_sensor node...")
        self.finger_sensor.close()


if __name__ == '__main__':
    try:
        node = SRBLSensorNode()
        node.publish_sensor_data()
    except rospy.ROSInterruptException:
        node.shutdown()
        rospy.loginfo("Shutting down srbl_sensor node.")