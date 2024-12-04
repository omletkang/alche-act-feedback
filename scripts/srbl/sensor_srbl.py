#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Seunghoon Kang
Email: alaska97@snu.ac.kr
Affiliation: Soft Robotics & Bionics Lab, Seoul National Univeristy
Date: 2024-10-01
Description: ALCHEMIST project
            Reads sensor data from an ESP32 using serial communication
update: version 4.4 (2024-10-21)
"""

import serial
import time
import numpy as np
from typing import List, Tuple

DEFAULT_HZ = 150  # Default frequency (Hz)

class FingerSensor:
    def __init__(self, 
        port: str = '/dev/ttyUSB0', 
        baudrate: int = 115200, 
        timeout: int = 1,
        hz: int = DEFAULT_HZ
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.hz = hz
        self.mcu = None
        self.offsets = np.zeros(8)  # Initialize offsets
    
    def open_serial(self) -> None:
        """Open the serial connection to the ESP32."""
        try:
            self.mcu = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print("Port Opened.")
            time.sleep(2)  # Wait for the connection to stabilize
        except serial.SerialException as e:
            print(f"Error opening serial connection: {e}")
            raise

    def read(self) -> str:
        self.mcu.write(b'A') # Send 'A' to request sensor data from the ESP32
        while True:
            try:
                data = self.mcu.readline().decode('utf-8').strip() # Read the line data
                if data:
                    return data
            except:
                pass
    
    def split_read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = self.read()
        values = np.array(data.split(','), dtype=float)  # Use NumPy array and convert to float
        adjusted_values = values - self.offsets
 
        sensor1 = -1 * adjusted_values[[0, 2, 4, 6]]  # Select values for sensor1
        sensor2 = -1 * adjusted_values[[1, 3, 5, 7]]  # Select values for sensor2
        return sensor1, sensor2

    
    def initialize_offset(self, max_samples: int = 40) -> None:
        """Initialize sensor offset by averaging values over a period of time or a number of samples."""
        print("Initializing sensor offsets...")

        collected_data = []
        start_time = time.time()

        while len(collected_data) < max_samples :
            data = self.read()
            values = np.array(data.split(','), dtype=float)  # Use NumPy for faster conversion
            collected_data.append(values)

            time.sleep(1.0 / self.hz)  # Wait for the next sample based on the sensor frequency

        # Stack the collected data into a NumPy array and calculate the mean along the columns
        collected_data = np.stack(collected_data)  # Stack into a 2D NumPy array
        self.offsets = np.mean(collected_data, axis=0)  # Compute the mean for each column

        print(f"Sensor offsets initialized to: {self.offsets}")

    def close(self) -> None:
        """Close the serial port"""
        print("Port Closed.")
        self.mcu.close()




def main():
    start_time = time.time()
    fsensor = FingerSensor(port='/dev/ttyUSB0')

    try:
        fsensor.open_serial()
        fsensor.initialize_offset()
        while True:
            sensor_data1, sensor_data2 = fsensor.split_read()
            if sensor_data1.size > 0:
                now = time.time()
                print(f"{now-start_time:.3f} {sensor_data1} {sensor_data2}")
            time.sleep(1/fsensor.hz) # Delay
    
    except KeyboardInterrupt:
        print("Exiting program")

    finally:
        fsensor.close()
    


if __name__ == "__main__":
    main()