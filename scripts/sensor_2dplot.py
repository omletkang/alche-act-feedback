#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import sys
import time

class RealTimePlotter:
    def __init__(self):
        # Initialize the Qt application
        self.app = QtWidgets.QApplication(sys.argv)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Set up PyQtGraph window and subplots
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-time Sensor Data")
        self.win.resize(800, 600)
        self.win.setWindowTitle("Real-time Sensor Data Plot")

        # Create upper and lower subplots for the two sensor groups
        self.plot1 = self.win.addPlot(title="Sensor Data Left finger")
        self.plot2 = self.win.addPlot(title="Sensor Data Right finger", row=1, col=0)

        # Configure plots
        for plot in [self.plot1, self.plot2]:
            plot.showGrid(x=True, y=True)
            plot.setLabel('left', 'Sensor Data')
            plot.setLabel('bottom', 'Time', units='s')
            plot.setYRange(-500, 500) # Set y-axis range
            plot.addLegend()

        # Initialize data buffers and curves for 4 lines in each plot
        self.update_rate = 50  # Update plot at 50 Hz
        self.time_window = 10  # Display last 10 seconds
        self.num_points = self.update_rate * self.time_window  # Total points in buffer

        self.data_buffer = np.zeros((8, self.num_points)) # Create empty buffers

        colors = ['r', 'g', 'b', 'm', 'purple', 'darkgreen', 'navy', 'brown']
        labels = ['data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7']
        self.curves = []

        for i in range(4):
            self.curves.append(self.plot1.plot(pen=pg.mkPen(colors[i], width=2), name=labels[i]))
        for i in range(4,8):
            self.curves.append(self.plot2.plot(pen=pg.mkPen(colors[i], width=2), name=labels[i]))

        # Track start time for dynamic x-axis
        self.start_time = time.time()

        # ROS Subscriber
        rospy.init_node('sensor_2dplotter', anonymous=True)
        rospy.Subscriber('sensor_data', Float32MultiArray, self.callback)

        # Timer for updating the plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000 // self.update_rate)  # Update interval in ms

    def callback(self, msg):
        """Callback function to receive data from the ROS topic."""
        data = np.array(msg.data)

        # Shift data left and add the new data point
        self.data_buffer = np.roll(self.data_buffer, -1, axis=1)
        self.data_buffer[:, -1] = data  # Insert new data at the end

    def update_plot(self):
        """Update the plot with the latest data from the buffer."""
        # Calculate elapsed time for dynamic x-axis
        current_time = time.time() - self.start_time
        x = np.linspace(current_time - self.time_window, current_time, self.num_points)

        # Update curves in each subplot
        for i in range(4):
            self.curves[i].setData(x, self.data_buffer[i, :])
        for i in range(4, 8):
            self.curves[i].setData(x, self.data_buffer[i, :])

    def run(self):
        """Run the Qt application loop."""
        rospy.loginfo("Starting real-time 2D plotter...")
        sys.exit(self.app.exec_())

if __name__ == '__main__':
    try:
        plotter = RealTimePlotter()
        plotter.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Real-time plotter node terminated.")
