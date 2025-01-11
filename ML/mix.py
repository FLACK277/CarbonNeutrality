import mysql.connector
import requests
import time
import pandas as pd
import random
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import chardet
import numpy as np

# Function to fetch air pollution data 
def fetch_air_pollution_data(api_key, lat, lon, start, end):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={api_key}"
    
    # Debugging: Print the request URL
    print(f"Requesting URL: {url}")
    
    response = requests.get(url)
    
    # Debugging: Print the response status
    print("Response Status Code:", response.status_code)
    print("Response Content:", response.content)
    
    return response.json()

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
            UNIQUE (mine_name, location, country, type_of_coal)  -- Prevent duplicate entries
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
            co2 DECIMAL(10, 2),  -- Add CO2 column
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

    # Print the air pollution data in table format
    if air_pollution_data.get('list'):
        # Prepare a list to hold the records
        records = []
        
        for data in air_pollution_data['list']:
            aqi = data['main']['aqi']
            components = data['components']
            
            # Generate random CO2 value for demonstration (or replace with actual logic)
            co2_value = random.uniform(100, 400)  # Example: CO2 values between 100-400 µg/m³
            
            # Append record to the list
            records.append({
                "AQI": aqi,
                "CO (µg/m³)": components['co'],
                "NO (µg/m³)": components['no'],
                "NO2 (µg/m³)": components['no2'],
                "O3 (µg/m³)": components['o3'],
                "SO2 (µg/m³)": components['so2'],
                "PM2.5 (µg/m³)": components['pm2_5'],
                "PM10 (µg/m³)": components['pm10'],
                "NH3 (µg/m³)": components['nh3'],
                "CO2 (µg/m³)": co2_value  # Add the generated CO2 value
            })
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Print DataFrame as a table
        print("\nAir Pollution Data:")
        print(df.to_string(index=False))  # Print without the index

        # Save the DataFrame to a CSV file
        output_file = "air_pollution_data.csv"
        df.to_csv(output_file, index=False)  # Save to CSV without the index
        print(f"Data saved to {output_file}")
    else:
        print("No air pollution data found.")

except mysql.connector.Error as err:
    print(f"Database connection error: {err}")
except Exception as e:
    print(f"An unexpected error occurred: {e}, {type(e)}")
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    print("Script finished.")

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

    # Use forward fill to fill NaNs from lag and rolling calculations
    df_sensor.ffill(inplace=True)  # Forward fill NaN values

    # Load CO2 emissions data
    co2_file_path = r'C:\Users\rawat\.vscode\cli\output.csv'
    df_co2 = pd.read_csv(co2_file_path, encoding='utf-8')

    if 'CO2_emissions' in df_co2.columns:
        print("CO2 emissions data read successfully!")
        
        # Proceed with feature engineering and model training
        df_sensor['CO_MA'] = df_sensor['CO'].rolling(window=5).mean()  # Moving average
        df_sensor['CO_lag_1'] = df_sensor['CO'].shift(1)               # 1-sample time lag
        
        # Ensure no NaN values in features and target
        features = df_sensor[['CO', 'PM10', 'PM2.5', 'AQI', 'CO_MA', 'CO_lag_1']]
        target = df_co2['CO2_emissions']  # Adjust this according to your DataFrame

        # Remove rows with NaN values in features or target
        X = features.dropna()
        y = target.loc[X.index]  # Align target with remaining features

        # Check if there's any NaN in the cleaned data
        print("Number of rows after removing NaN values:", X.shape[0])

        # Polynomial features
        poly = PolynomialFeatures(degree=2)  # Change degree for higher order polynomial
        X_poly = poly.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model Training
        model = LinearRegression()  # Use Linear Regression for Polynomial features
        model.fit(X_train_scaled, y_train)

        # Model Prediction
        predictions = model.predict(X_test_scaled)

        # Evaluation
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")

        # Plotting predictions vs actual values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
        plt.xlabel('Actual CO2 Emissions')
        plt.ylabel('Predicted CO2 Emissions')
        plt.title('Actual vs Predicted CO2 Emissions (Polynomial Regression)')
        plt.grid()
        plt.show()

    else:
        print("CO2 emissions column not found in df_co2.")

except FileNotFoundError as fnf_error:
    print(f"File not found: {fnf_error}")
except pd.errors.EmptyDataError as empty_error:
    print(f"Empty data error: {empty_error}")
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")

