import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import chardet
import matplotlib.pyplot as plt

# Step 1: Detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read(100000)  # Read the first 1,00,000 bytes
        result = chardet.detect(raw_data)
        return result['encoding']

# Step 2: Load sensor data with detected encoding
file_path = r'C:\Users\rawat\.vscode\cli\output.csv'
detected_encoding = detect_encoding(file_path)

try:
    df_sensor = pd.read_csv(file_path, sep=',', encoding=detected_encoding, on_bad_lines='skip')
    print("Sensor data read successfully!")
except Exception as e:
    print(f"An error occurred while reading the sensor data: {e}")

# Step 3: Load CO2 emissions data
co2_file_path = r'C:\Users\rawat\.vscode\cli\output.csv'
try:
    df_co2 = pd.read_csv(co2_file_path, encoding='utf-8')
    print("CO2 emissions data read successfully!")
except FileNotFoundError:
    print(f"CO2 data file not found: {co2_file_path}")
    df_co2 = None
except Exception as e:
    print(f"An error occurred while reading the CO2 data: {e}")
    df_co2 = None

# Proceed if both DataFrames exist
if df_sensor is not None and df_co2 is not None:
    # Strip any leading/trailing spaces from column names
    df_sensor.columns = df_sensor.columns.str.strip()
    df_co2.columns = df_co2.columns.str.strip()

    # Skip merging on 'timestamp'
    print(f"Sensor data shape: {df_sensor.shape}")
    print(f"CO2 data shape: {df_co2.shape}")

    # Step 4: Handle missing data (interpolation)
    df_sensor['CO'] = df_sensor['CO'].interpolate(method='linear')
    df_sensor['PM10'] = df_sensor['PM10'].interpolate(method='linear')
    df_sensor['PM2.5'] = df_sensor['PM2.5'].interpolate(method='linear')

    # Feature Engineering
    df_sensor['CO_MA'] = df_sensor['CO'].rolling(window=5).mean()  # Moving average
    df_sensor['CO_lag_1'] = df_sensor['CO'].shift(1)               # 1-sample time lag
    df_sensor.fillna(method='ffill', inplace=True)                 # Fill NaNs from lag and rolling

    # Define features and target
    features = df_sensor[['CO', 'PM10', 'PM2.5', 'AQI', 'CO_MA', 'CO_lag_1']]
    target = df_co2['CO2_emissions']  # Adjust this if needed based on the columns available in df_co2

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Feature Scaling using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and Evaluate Models

    # (a) Linear Regression
    linear_reg = LinearRegression()
    linear_reg.fit(X_train_scaled, y_train)
    lr_y_pred = linear_reg.predict(X_test_scaled)

    # Evaluate Linear Regression
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_y_pred))
    lr_mae = mean_absolute_error(y_test, lr_y_pred)
    print(f"Linear Regression RMSE: {lr_rmse:.4f}")
    print(f"Linear Regression MAE: {lr_mae:.4f}")

    # (b) Stochastic Gradient Descent
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='optimal', random_state=42)
    sgd_reg.fit(X_train_scaled, y_train)
    sgd_y_pred = sgd_reg.predict(X_test_scaled)

    # Evaluate SGDRegressor
    sgd_rmse = np.sqrt(mean_squared_error(y_test, sgd_y_pred))
    sgd_mae = mean_absolute_error(y_test, sgd_y_pred)
    print(f"SGD Regressor RMSE: {sgd_rmse:.4f}")
    print(f"SGD Regressor MAE: {sgd_mae:.4f}")

    # (c) Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train_scaled, y_train)
    rf_y_pred = rf_reg.predict(X_test_scaled)

    # Evaluate Random Forest
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))
    rf_mae = mean_absolute_error(y_test, rf_y_pred)
    print(f"Random Forest RMSE: {rf_rmse:.4f}")
    print(f"Random Forest MAE: {rf_mae:.4f}")

    # Cross-validation for Random Forest
    rf_cv_scores = cross_val_score(rf_reg, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    rf_cv_rmse = np.sqrt(-rf_cv_scores).mean()
    print(f"Random Forest Cross-validated RMSE: {rf_cv_rmse:.4f}")

    # Plotting
    plt.subplot(3, 1, 1)  # Linear Regression
    plt.scatter(y_test, lr_y_pred, color='blue', label='Linear Regression Predictions', alpha=0.5)
    plt.plot(y_test, y_test, color='red', linewidth=2, label='Actual Values')
    plt.title('Linear Regression: Actual vs Predicted CO2 Emissions')
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)  # SGD Regressor
    plt.scatter(y_test, sgd_y_pred, color='green', label='SGD Predictions', alpha=0.5)
    plt.plot(y_test, y_test, color='red', linewidth=2, label='Actual Values')
    plt.title('SGD Regressor: Actual vs Predicted CO2 Emissions')
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)  # Random Forest
    plt.scatter(y_test, rf_y_pred, color='orange', label='Random Forest Predictions', alpha=0.5)
    plt.plot(y_test, y_test, color='red', linewidth=2, label='Actual Values')
    plt.title('Random Forest: Actual vs Predicted CO2 Emissions')
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions')
    plt.legend()
    plt.grid()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
