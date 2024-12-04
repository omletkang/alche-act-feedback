# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# Load the data from CSV
filename = 'sensor_data_log_2.csv'  # Replace with your file path
data = pd.read_csv(filename)

# Extract input (H_matrix) and output (F_matrix)
H_matrix = data.iloc[30:, 6:10].values  # Columns for hall sensor data (data1~data4)
F_matrix = data.iloc[30:, 0:3].values   # Columns for known force vectors (RFT_1, RFT_2, RFT_3)

# Split the data into training and testing sets
H_train, H_test, F_train, F_test = train_test_split(H_matrix, F_matrix, test_size=0.2, random_state=42)

# Scale the data for better training stability
scaler = StandardScaler()
H_train = scaler.fit_transform(H_train)
H_test = scaler.transform(H_test)

# Save the scaler for later use
scaler_filename = 'scaler_2.pkl'  # Replace with your file path
joblib.dump(scaler, scaler_filename)

# Create an MLP model
model = keras.Sequential([
    layers.InputLayer(input_shape=(4,)),  # Input layer with 4 neurons
    layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons
    layers.Dense(16, activation='relu'),  # Hidden layer with 16 neurons
    layers.Dense(3)                       # Output layer with 3 neurons (for 3-axis force)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Train the model
history = model.fit(H_train, F_train, epochs=200, batch_size=16, validation_split=0.2, verbose=2)

# Evaluate the model on the test set
loss, mse = model.evaluate(H_test, F_test, verbose=2)
print(f'Test MSE: {mse}')

# Save the trained model
model.save('mlp_model_2.h5')  # Replace with your file path