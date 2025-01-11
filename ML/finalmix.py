import mysql.connector
import requests
import time
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import chardet
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, filename='script.log', filemode='w', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to fetch air pollution data 
def fetch_air_pollution_data(api_key, lat, lon, start, end):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={api_key}"
    logging.info(f"Requesting URL: {url}")
    
    response = requests.get(url)
    logging.info("Response Status Code: %s", response.status_code)
    
    if response.status_code == 200:
        return response.json()
    else:
        logging.error("Error fetching data: %s", response.content)
        return {}

# Function to insert coal mine data
def insert_coal_mine(cursor, mine_data):
    insert_mine_query = '''
    INSERT INTO Coal_Mines (mine_name, location, country, type_of_coal, annual_production, year_established)
    VALUES (%s, %s, %s, %s, %s, %s)
    '''
    cursor.execute(insert_mine_query, (
        mine_data['mine_name'],
        mine_data['location'],
        mine_data['country'],
        mine_data['type_of_coal'],
        mine_data['annual_production'],
        mine_data['year_established']
    ))
    logging.info("Coal mine data inserted successfully.")

# Function to insert air pollution data
def insert_air_pollution_data(cursor, mine_id, pollution_records):
    insert_pollution_query = '''
    INSERT INTO Air_Pollution_Data (mine_id, year, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3, co2)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''
    cursor.executemany(insert_pollution_query, pollution_records)
    logging.info("Air pollution data inserted successfully.")

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read(100000)  # Read the first 100,000 bytes
        result = chardet.detect(raw_data)
        return result['encoding']

# Function to load sensor data with detected encoding
def load_sensor_data(file_path):
    detected_encoding = detect_encoding(file_path)
    
    df_sensor = pd.read_csv(file_path, sep=',', encoding=detected_encoding, on_bad_lines='skip')
    logging.info("Sensor data read successfully!")
    
    # Strip any leading/trailing spaces from column names
    df_sensor.columns = df_sensor.columns.str.strip()
    
    # Interpolate missing values
    columns_to_interpolate = ['CO', 'PM10', 'PM2.5']
    for col in columns_to_interpolate:
        if col in df_sensor.columns:
            df_sensor[col] = df_sensor[col].interpolate(method='linear')
    
    # Fill NaNs
    df_sensor.ffill(inplace=True)
    
    return df_sensor

# Main script
def main():
    # Database connection setup
    cursor = None
    conn = None
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",  # Change as needed
            password="YourPasswordHere",  # Change to your MySQL password
            database="XYZhjsaAab"  # Connect to the desired database
        )
        logging.info("Database connection successful.")
        cursor = conn.cursor()

        # Insert Coal Mines data
        mine_data = {
            'mine_name': 'Example Mine',
            'location': 'Example Location',
            'country': 'Example Country',
            'type_of_coal': 'Bituminous',
            'annual_production': 50000,
            'year_established': 1990
        }

        try:
            insert_coal_mine(cursor, mine_data)
            conn.commit()
        except mysql.connector.Error as err:
            logging.error("Error inserting coal mine data: %s", err)

        # Fetch the mine_id for inserting emissions
        cursor.execute("SELECT mine_id FROM Coal_Mines WHERE mine_name = %s", (mine_data['mine_name'],))
        result = cursor.fetchone()
        if result:
            mine_id = result[0]  # Get the mine_id
            logging.info("Fetched mine_id: %s", mine_id)
        else:
            logging.warning("No mine_id found. Insertion of air pollution data aborted.")
            return

        # Fetch air pollution data from Open Weather Map
        lat = 48.8566  # Latitude for Paris
        lon = 2.3522   # Longitude for Paris
        api_key = "YourAPIKeyHere"  # Change to your valid API key

        # Define time range 
        start_time = int(time.mktime(time.strptime('2024-09-01', '%Y-%m-%d')))
        end_time = int(time.mktime(time.strptime('2024-09-30', '%Y-%m-%d')))

        air_pollution_data = fetch_air_pollution_data(api_key, lat, lon, start_time, end_time)

        # Prepare pollution records
        pollution_records = []
        if air_pollution_data.get('list'):
            for data in air_pollution_data['list']:
                aqi = data['main']['aqi']
                components = data['components']
                co2_value = random.uniform(100, 400)  # Example: CO2 values between 100-400 µg/m³
                
                pollution_records.append((mine_id, 2024, aqi, components['co'], components['no'], 
                                           components['no2'], components['o3'], components['so2'], 
                                           components['pm2_5'], components['pm10'], components['nh3'], co2_value))

            # Insert air pollution data
            try:
                insert_air_pollution_data(cursor, mine_id, pollution_records)
                conn.commit()
            except mysql.connector.Error as err:
                logging.error("Error inserting air pollution data: %s", err)
        else:
            logging.warning("No air pollution data found.")

        # Load and process sensor data
        file_path = r'C:\Users\rawat\.vscode\cli\output.csv'
        df_sensor = load_sensor_data(file_path)

        # Load CO2 emissions data
        co2_file_path = r'C:\Users\rawat\.vscode\cli\output.csv'
        df_co2 = pd.read_csv(co2_file_path, encoding='utf-8')

        if 'CO2_emissions' in df_co2.columns:
            logging.info("CO2 emissions data read successfully!")

            # Feature engineering
            df_sensor['CO_MA'] = df_sensor['CO'].rolling(window=5).mean()  # Moving average
            df_sensor['CO_lag_1'] = df_sensor['CO'].shift(1)               # 1-sample time lag

            # Prepare features and target
            features = df_sensor[['CO', 'PM10', 'PM2.5', 'AQI', 'CO_MA', 'CO_lag_1']]
            target = df_co2['CO2_emissions']

            # Remove rows with NaN values
            X = features.dropna()
            y = target.loc[X.index]

            # Polynomial features
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

            # Scale features
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Model Training
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

            # Model Prediction
            predictions = model.predict(X_test_scaled)

            # Evaluation
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)

            logging.info(f"Mean Squared Error: {mse}")
            logging.info(f"Mean Absolute Error: {mae}")

            # Plotting predictions vs actual values
            plt.figure(figsize=(10, 5))
            plt.scatter(y_test, predictions, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
            plt.xlabel('Actual CO2 Emissions')
            plt.ylabel('Predicted CO2 Emissions')
            plt.title('Actual vs Predicted CO2 Emissions (Polynomial Regression)')
            plt.grid()
            plt.savefig('actual_vs_predicted_CO2_emissions.png')  # Save plot
            plt.show()
        else:
            logging.warning("CO2 emissions column not found in df_co2.")

    except mysql.connector.Error as err:
        logging.error("Database connection error: %s", err)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
