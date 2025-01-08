import requests
import mysql.connector
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Function to get emissions from Climatiq API
def get_climatiq_emissions(activity_id, amount, unit):
    url = "https://app.climatiq.io/projects/409932408236802625/keys"
    headers = {
        'Authorization': 'Bearer TOMBFLACK 0AQNHXWPCFPD5C78WVXV4VFAMEX8',  # Bearer token
        'Content-Type': 'application/json'
    }
    payload = {
        "emission_factor": {
            "activity_id": activity_id
        },
        "parameters": {
            "amount": amount,
            "unit": unit
        }
    }

    print(f"Making request to URL: {url}")
    print(f"With headers: {headers}")
    print(f"Payload: {payload}")

    try:
        response = session.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        print(f"API Response: {data}")  # Print the response for debugging
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        return None  # Return None

# Function to insert emission data into MySQL
def insert_emission_data(cursor, mine_id, year, total_emissions, source):
    try:
        sql = '''INSERT INTO Carbon_Emissions (mine_id, year, total_emissions, emission_source)
                 VALUES (%s, %s, %s, %s)'''
        values = (mine_id, year, total_emissions, source)
        cursor.execute(sql, values)
        print("Data inserted successfully.")
    except mysql.connector.Error as err:
        print(f"Error inserting data: {err}")

# Main processing function
def process_emission_data(cursor):
    activity_id = 'fugitive_gas-type_carbon_dioxide'  # Activity ID
    amount = 1000
    unit = "ton"

    climatiq_data = get_climatiq_emissions(activity_id, amount, unit)

    if climatiq_data and 'co2e' in climatiq_data:
        total_emissions = climatiq_data['co2e']
        source = "Climatiq API"

        insert_emission_data(cursor, 1, 2024, total_emissions, source)
    else:
        print("Error: Invalid data received from API or 'co2e' not found.")

# Database connection setup
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Dragonball2005",
        database="coal_emissions_db"
    )
    cursor = conn.cursor()
    
    # function to get data from API and insert into the database
    process_emission_data(cursor)
    
    # Commiting changes to the database
    conn.commit()
except mysql.connector.Error as err:
    print(f"Database connection error: {err}")
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    print("Database connection closed.")
