#!/usr/bin/env python
import sys
import time
import threading
import rospy
import rospkg
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyvista as pv
from pyvistaqt import QtInteractor
from std_msgs.msg import Float32MultiArray

# Deep Learning
from tensorflow.keras.models import load_model
import joblib

# Class for subscribing to ROS topic
class SensorSubscriberNode:
    def __init__(self):
        rospy.init_node('sensor_3dplotter', anonymous=True)
        self.sub = rospy.Subscriber('sensor_data', Float32MultiArray, self.callback)
        self.latest_data = np.zeros(8)
        self.data_lock = threading.Lock()

        self.update_rate = 50  # Update plot at 50 Hz
        self.time_window = 10  # Display last 10 seconds
        self.num_points = self.update_rate * self.time_window  # Total points in buffer

        self.data_buffer = np.zeros((8, self.num_points)) # Create empty buffers

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

    def callback(self, msg):
        with self.data_lock:
            data = np.array(msg.data)

            # Shift data left and add the new data point
            self.data_buffer = np.roll(self.data_buffer, -1, axis=1)
            self.data_buffer[:, -1] = data

    def get_data(self):
        with self.data_lock:
            return self.data_buffer.copy()

    def get_force(self):
        with self.data_lock:
            # Extract the latest 4 Hall sensor values (indices 0 to 3)
            hall_sensor_values = self.data_buffer[0:4, -1].reshape(1, -1)
            hall_sensor_values_scaled = self.scaler_1.transform(hall_sensor_values)
            force_vector_R = self.model_1.predict(hall_sensor_values_scaled)
            # Extract the latest 4 Hall sensor values (indices 4 to 7)
            hall_sensor_values = self.data_buffer[4: , -1].reshape(1, -1)
            hall_sensor_values_scaled = self.scaler_2.transform(hall_sensor_values)
            force_vector_L = self.model_2.predict(hall_sensor_values_scaled)
            return force_vector_R, force_vector_L

# Class for managing the PyQtGraph widget
class PyQtGraphWidget(QtWidgets.QWidget):
    def __init__(self, subscriber):
        super().__init__()
        self.subscriber = subscriber
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.graph_layout = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graph_layout)

        # Create upper and lower plots
        self.plot1 = self.graph_layout.addPlot(title="Sensor Data Left finger")
        self.plot2 = self.graph_layout.addPlot(title="Sensor Data Right finger", row=1, col=0)

        # Configure plots
        for plot in [self.plot1, self.plot2]:
            plot.showGrid(x=True, y=True)
            plot.setLabel('left', 'Sensor Data')
            plot.setLabel('bottom', 'Time', units='s')
            plot.setYRange(-500, 500) # Set y-axis range
            plot.addLegend()
        
        # Load subsrciber data
        self.update_rate = subscriber.update_rate 
        self.time_window = subscriber.time_window
        self.num_points = subscriber.num_points

        colors = ['r', 'g', 'b', 'm', 'purple', 'darkgreen', 'navy', 'brown']
        labels = ['data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7']
        self.curves = []

        for i in range(4):
            self.curves.append(self.plot1.plot(pen=pg.mkPen(colors[i], width=2), name=labels[i]))
        for i in range(4,8):
            self.curves.append(self.plot2.plot(pen=pg.mkPen(colors[i], width=2), name=labels[i]))
        
        # Track start time for dynamic x-axis
        self.start_time = time.time()

        # Timer for updating the plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(1000 // self.update_rate) # Update interval in ms

    def update_plots(self):
        """Update the plot with the latest data from the buffer."""
        # Calculate elapsed time for dynamic x-axis
        current_time = time.time() - self.start_time
        x = np.linspace(current_time - self.time_window, current_time, self.num_points)

        # Update curves in each subplot
        buffer = subscriber.get_data()
        for i in range(4):
            self.curves[i].setData(x, buffer[i, :])
        for i in range(4, 8):
            self.curves[i].setData(x, buffer[i, :])


# Class for managing the PyVista widget
class PyVistaWidget(QtWidgets.QWidget):
    def __init__(self, subscriber):
        super().__init__()
        self.subscriber = subscriber
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)

        # Load subsrciber data
        # self.update_rate = subscriber.update_rate 
        self.update_rate = 30

        # Add static 3D boxes
        self.box_1 = pv.Box(bounds=(5, 6, -1, 1, 3, 7))
        self.box_2 = pv.Box(bounds=(-6, -5, -1, 1, 3, 7))
        self.plotter.add_mesh(self.box_1, opacity=0.6, color="lightblue")
        self.plotter.add_mesh(self.box_2, opacity=0.6, color="lightblue")
        
        # Initial arrows for box 1
        self.arrow_1_x = pv.Arrow(start=(5.0 - 1.0, 0.0, 5.0), direction=(1.0, 0.0, 0.0))
        self.arrow_1_y = pv.Arrow(start=(5.0, 0.0, 5.0), direction=(0.0, 1.0, 0.0))
        self.arrow_1_z = pv.Arrow(start=(5.0, 0.0, 5.0), direction=(0.0, 0.0, 1.0))
        self.actor_arrow_1_x = self.plotter.add_mesh(self.arrow_1_x, color="red")
        self.actor_arrow_1_y = self.plotter.add_mesh(self.arrow_1_y, color="green")
        self.actor_arrow_1_z = self.plotter.add_mesh(self.arrow_1_z, color="blue")
        # Add fixed arrows for box 2
        self.arrow_2_x = pv.Arrow(start=(-5.0 + 1.0, 0.0, 5.0), direction=(-1.0, 0.0, 0.0))
        self.arrow_2_y = pv.Arrow(start=(-5.0, 0.0, 5.0), direction=(0.0, 1.0, 0.0))
        self.arrow_2_z = pv.Arrow(start=(-5.0, 0.0, 5.0), direction=(0.0, 0.0, 1.0))
        self.actor_arrow_2_x = self.plotter.add_mesh(self.arrow_2_x, color="red")
        self.actor_arrow_2_y = self.plotter.add_mesh(self.arrow_2_y, color="green")
        self.actor_arrow_2_z = self.plotter.add_mesh(self.arrow_2_z, color="blue")

        self.gain = 0.2 # Gain for arrow_x; Adjust!!!

        self.plotter.show_axes()
        self.plotter.disable() # disable interation
        self.plotter.camera_position = [
                            (17, 29, 12),
                            (0.3, -0.2, 5),
                            (-0.1, -0.21, 0.97),
                        ]

        # Timer for updating the sphere size
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_sphere)
        self.timer.start(1000 // self.update_rate)

    def update_sphere(self):
        force_vector_R, force_vector_L = self.subscriber.get_force()
        # print(force_vector_R) # print(force_vector_L)
        if force_vector_R is not None and len(force_vector_R) > 0:
            force_R1 = force_vector_R[0][0]  # x component
            force_R2 = -1 * force_vector_R[0][1]  # y component
            force_R3 = -1 * force_vector_R[0][2]  # z component

            force_L1 = force_vector_L[0][0]  # x component
            force_L2 = -1 * force_vector_L[0][1]  # y component
            force_L3 = -1 * force_vector_L[0][2]  # z component

            # Update the arrows for box 1 using force data
            updated_arrow_1_x = pv.Arrow(start=(5.0 - self.gain*force_R3, 0.0, 5.0), 
                                            direction=(1, 0.0, 0.0), scale=self.gain*float(force_R3))
            updated_arrow_1_y = pv.Arrow(start=(5.0, 0.0, 5.0), direction=(0.0, 1, 0.0), scale=float(force_R1))
            updated_arrow_1_z = pv.Arrow(start=(5.0, 0.0, 5.0), direction=(0.0, 0.0, 1), scale=float(force_R2))

            updated_arrow_2_x = pv.Arrow(start=(-5.0 + self.gain*force_L3, 0.0, 5.0), 
                                            direction=(-1, 0.0, 0.0), scale=self.gain*float(force_L3))
            updated_arrow_2_y = pv.Arrow(start=(-5.0, 0.0, 5.0), direction=(0.0, -1, 0.0), scale=float(force_L1))
            updated_arrow_2_z = pv.Arrow(start=(-5.0, 0.0, 5.0), direction=(0.0, 0.0, 1), scale=float(force_L2))

            # Update actors with new arrows
            self.actor_arrow_1_x.mapper.SetInputData(updated_arrow_1_x)
            self.actor_arrow_1_y.mapper.SetInputData(updated_arrow_1_y)
            self.actor_arrow_1_z.mapper.SetInputData(updated_arrow_1_z)
            self.actor_arrow_2_x.mapper.SetInputData(updated_arrow_2_x)
            self.actor_arrow_2_y.mapper.SetInputData(updated_arrow_2_y)
            self.actor_arrow_2_z.mapper.SetInputData(updated_arrow_2_z)
        self.plotter.render()
        # print(self.plotter.camera_position)

# Main application window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, subscriber):
        super().__init__()
        self.frame = QtWidgets.QFrame()
        self.layout = QtWidgets.QHBoxLayout()
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        # Add PyQtGraph widget
        self.pyqtgraph_widget = PyQtGraphWidget(subscriber)
        self.layout.addWidget(self.pyqtgraph_widget)

        # Add PyVista widget
        self.pyvista_widget = PyVistaWidget(subscriber)
        self.layout.addWidget(self.pyvista_widget)

        self.setWindowTitle("Real-time PyQtGraph and PyVista with ROS")
        self.resize(1200, 600)

# Run the application
if __name__ == '__main__':
    try:
        subscriber = SensorSubscriberNode()
        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow(subscriber)
        window.show()

        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: rospy.rostime.wallsleep(0.01))
        timer.start(10) # 10 ms

        sys.exit(app.exec_())
    except rospy.ROSInterruptException:
        pass
