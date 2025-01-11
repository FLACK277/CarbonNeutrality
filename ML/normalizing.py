from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['CO', 'PM10', 'PM2.5', 'AQI']] = scaler.fit_transform(df[['CO', 'PM10', 'PM2.5', 'AQI']])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['CO', 'PM10', 'PM2.5', 'AQI']] = scaler.fit_transform(df[['CO', 'PM10', 'PM2.5', 'AQI']])
