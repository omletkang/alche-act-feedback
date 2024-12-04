#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import threading
import rospkg
from tensorflow.keras.models import load_model
import joblib

class SensorForcePublisherNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('sensor_force_publisher', anonymous=True)
        self.sub = rospy.Subscriber('sensor_data', Float32MultiArray, self.callback)
        self.pub = rospy.Publisher('sensor_force', Float32MultiArray, queue_size=10)

        self.data_lock = threading.Lock()
        self.latest_data = None  # Store the most recent data for prediction

        # Load models and scalers
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('alche_srbl')  # Replace with your package name if necessary

        model_path = f"{package_path}/sensor_calibration/mlp_model_1.h5"
        scaler_path = f"{package_path}/sensor_calibration/scaler_1.pkl"
        self.model_1 = load_model(model_path)
        self.scaler_1 = joblib.load(scaler_path)

        model_path = f"{package_path}/sensor_calibration/mlp_model_2.h5"
        scaler_path = f"{package_path}/sensor_calibration/scaler_2.pkl"
        self.model_2 = load_model(model_path)
        self.scaler_2 = joblib.load(scaler_path)

        rospy.loginfo("SensorForcePublisherNode initialized and ready.")

    def callback(self, msg):
        with self.data_lock:
            self.latest_data = np.array(msg.data)

    def predict_force(self):
        with self.data_lock:
            if self.latest_data is None:
                return None  # No data available yet
            data = self.latest_data

        # Predict forces using models
        hall_sensor_values_R = data[0:4].reshape(1, -1)  # Extract right-hand sensor values
        hall_sensor_values_R_scaled = self.scaler_1.transform(hall_sensor_values_R)
        force_vector_R = self.model_1.predict(hall_sensor_values_R_scaled)

        hall_sensor_values_L = data[4:8].reshape(1, -1)  # Extract left-hand sensor values
        hall_sensor_values_L_scaled = self.scaler_2.transform(hall_sensor_values_L)
        force_vector_L = self.model_2.predict(hall_sensor_values_L_scaled)

        # Extract individual components (ensure compatibility with ROS message)
        force_R1, force_R2, force_R3 = force_vector_R[0]
        force_L1, force_L2, force_L3 = force_vector_L[0]

        # Combine forces into a single list
        return [force_R1, force_R2, force_R3, force_L1, force_L2, force_L3]

    def spin(self):
        rate = rospy.Rate(30)  # 30 Hz publishing loop
        while not rospy.is_shutdown():
            sensor_force = self.predict_force()
            if sensor_force is not None:
                self.publish_force(sensor_force)
            rate.sleep()

    def publish_force(self, sensor_force):
        # Publish the predicted forces
        msg = Float32MultiArray()
        msg.data = sensor_force
        self.pub.publish(msg)
        # rospy.loginfo("Published sensor_force: %s", sensor_force)
        rospy.loginfo_throttle(0.1, "Published sensor_force: %s", sensor_force)

if __name__ == '__main__':
    try:
        node = SensorForcePublisherNode()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("SensorForcePublisherNode terminated.")
