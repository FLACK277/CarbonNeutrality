import mysql.connector
import requests
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import chardet
import matplotlib.pyplot as plt

# Function to fetch air pollution data
def fetch_air_pollution_data(api_key, lat, lon, start, end):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={api_key}"
    
    # Debugging: Print the request URL
    print(f"Requesting URL: {url}")
    
    response = requests.get(url)
    
    # Check if the response is successful
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch data. Status Code:", response.status_code)
        print("Response Content:", response.content)
        return None

# Database connection setup
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Dragonball2005"
    )
    
    cursor = conn.cursor()

    cursor.execute("CREATE DATABASE IF NOT EXISTS coal_emissions_db;")
    cursor.execute("USE coal_emissions_db;")

    sql_commands = [
        '''
        CREATE TABLE IF NOT EXISTS Coal_Mines (
            mine_id INT PRIMARY KEY AUTO_INCREMENT,
            mine_name VARBINARY(255) NOT NULL,
            location VARCHAR(255) NOT NULL,
            country VARCHAR(100) NOT NULL,
            type_of_coal VARCHAR(100) NOT NULL,
            annual_production INT CHECK (annual_production > 0),
            year_established YEAR NOT NULL,
            UNIQUE (mine_name, location, country, type_of_coal)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS Carbon_Emissions (
            emission_id INT PRIMARY KEY AUTO_INCREMENT,
            mine_id INT NOT NULL,
            year INT NOT NULL,
            total_emissions DECIMAL(10, 2) CHECK (total_emissions >= 0),
            emission_source VARCHAR(255) NOT NULL,
            FOREIGN KEY (mine_id) REFERENCES Coal_Mines(mine_id) ON DELETE CASCADE
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS Emission_Factors (
            factor_id INT PRIMARY KEY AUTO_INCREMENT,
            type_of_coal VARCHAR(100) NOT NULL,
            emission_factor DECIMAL(10, 4) CHECK (emission_factor > 0),
            unit VARCHAR(50) NOT NULL
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS Air_Pollution_Data (
            pollution_id INT PRIMARY KEY AUTO_INCREMENT,
            mine_id INT NOT NULL,
            year INT NOT NULL,
            aqi INT,
            co DECIMAL(10, 2),
            no DECIMAL(10, 2),
            no2 DECIMAL(10, 2),
            o3 DECIMAL(10, 2),
            so2 DECIMAL(10, 2),
            pm2_5 DECIMAL(10, 2),
            pm10 DECIMAL(10, 2),
            nh3 DECIMAL(10, 2),
            FOREIGN KEY (mine_id) REFERENCES Coal_Mines(mine_id) ON DELETE CASCADE
        )
        '''
    ]

    for command in sql_commands:
        cursor.execute(command)
        print("Table created or already exists.")

    cursor.execute("DROP TRIGGER IF EXISTS validate_emission;")
    cursor.execute(""" 
        CREATE TRIGGER validate_emission 
        BEFORE INSERT ON Carbon_Emissions 
        FOR EACH ROW 
        BEGIN 
            IF NEW.total_emissions < 0 THEN 
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Invalid emission data: emissions must be positive'; 
            END IF; 
        END; 
    """)
    print("Trigger created or already exists.")

    # Fetch air pollution data from Open Weather Map
    lat = 48.8566  # Latitude for Paris
    lon = 2.3522   # Longitude for Paris
    api_key = "6d84ed0f89761fda5c29ee37702a32f9"  # Valid API key

    # Define time range 
    start_time = int(time.mktime(time.strptime('2024-09-01', '%Y-%m-%d')))  # Start date
    end_time = int(time.mktime(time.strptime('2024-09-30', '%Y-%m-%d')))    # End date

    air_pollution_data = fetch_air_pollution_data(api_key, lat, lon, start_time, end_time)

    # Print and process the air pollution data
    if air_pollution_data and air_pollution_data.get('list'):
        for data in air_pollution_data['list']:
            aqi = data['main']['aqi']
            components = data['components']

            print(f"Air Quality Index (AQI): {aqi}")
            print("Pollutants Levels:")
            for pollutant, level in components.items():
                print(f"{pollutant.upper()}: {level} µg/m³")
    else:
        print("No air pollution data found or API call failed.")

    # Step 1: Detect file encoding
    def detect_encoding(file_path):
        with open(file_path, 'rb') as file:
            raw_data = file.read(100000)  # Read the first 100,000 bytes
            result = chardet.detect(raw_data)
            return result['encoding']

    # Step 2: Load sensor data with detected encoding
    file_path = r'C:\Users\rawat\.vscode\cli\output.csv'
    detected_encoding = detect_encoding(file_path)

    try:
        df_sensor = pd.read_csv(file_path, sep=',', encoding=detected_encoding, on_bad_lines='skip')
        print("Sensor data read successfully!")
        print("Columns in df_sensor:", df_sensor.columns.tolist())  # Check columns

        # Strip any leading/trailing spaces from column names
        df_sensor.columns = df_sensor.columns.str.strip()

        # Check if required columns exist before interpolating
        if 'CO' in df_sensor.columns:
            df_sensor['CO'] = df_sensor['CO'].interpolate(method='linear')
        else:
            print("Column 'CO' not found in df_sensor.")

        if 'PM10' in df_sensor.columns:
            df_sensor['PM10'] = df_sensor['PM10'].interpolate(method='linear')
        else:
            print("Column 'PM10' not found in df_sensor.")

        if 'PM2.5' in df_sensor.columns:
            df_sensor['PM2.5'] = df_sensor['PM2.5'].interpolate(method='linear')
        else:
            print("Column 'PM2.5' not found in df_sensor.")

        # Load CO2 emissions data
        co2_file_path = r'C:\Users\rawat\.vscode\cli\output.csv'
        df_co2 = pd.read_csv(co2_file_path, encoding='utf-8')

        if 'CO2_emissions' in df_co2.columns:
            print("CO2 emissions data read successfully!")
            
            # Proceed with feature engineering and model training
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

            # (b) Stochastic Gradient Descent Regressor
            sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3)
            sgd_reg.fit(X_train_scaled, y_train)
            sgd_y_pred = sgd_reg.predict(X_test_scaled)

            # Evaluate SGD Regressor
            sgd_rmse = np.sqrt(mean_squared_error(y_test, sgd_y_pred))
            sgd_mae = mean_absolute_error(y_test, sgd_y_pred)
            print(f"SGD Regressor RMSE: {sgd_rmse:.4f}")
            print(f"SGD Regressor MAE: {sgd_mae:.4f}")

            # (c) Random Forest Regressor from sklearn.ensemble
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_reg.fit(X_train_scaled, y_train)
            rf_y_pred = rf_reg.predict(X_test_scaled)

            # Evaluate Random Forest Regressor
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))
            rf_mae = mean_absolute_error(y_test, rf_y_pred)
            print(f"Random Forest RMSE: {rf_rmse:.4f}")
            print(f"Random Forest MAE: {rf_mae:.4f}")

            # Visualization of Predictions
            plt.figure(figsize=(12, 6))
            plt.scatter(y_test, lr_y_pred, color='blue', label='Linear Regression Predictions')
            plt.scatter(y_test, sgd_y_pred, color='orange', label='SGD Predictions')
            plt.scatter(y_test, rf_y_pred, color='green', label='Random Forest Predictions')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title('Model Predictions vs True Values')
            plt.legend()
            plt.show()

        else:
            print("Column 'CO2_emissions' not found in co2 dataframe.")

except mysql.connector.Error as err:
    print(f"Database error: {err}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'cursor' in locals() and cursor is not None:
        cursor.close()
    if 'conn' in locals() and conn is not None:
        conn.close()
