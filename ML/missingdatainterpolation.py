import pandas as pd

# Assume df is your sensor data DataFrame with a 'timestamp' column and sensor readings
df['CO'] = df['CO'].interpolate(method='linear')
df['PM10'] = df['PM10'].interpolate(method='linear')
df['PM2.5'] = df['PM2.5'].interpolate(method='linear')
df.fillna(method='ffill', inplace=True)  # Forward fill
df.fillna(method='bfill', inplace=True)  # Backward fill
