df['CO_MA'] = df['CO'].rolling(window=5).mean()  # 5-sample moving average
df['CO_lag_1'] = df['CO'].shift(1)  # 1-sample time lag
df['CO_lag_2'] = df['CO'].shift(2)  # 2-sample time lag
# Assuming df_sensor is your sensor data and df_co2 contains CO2 emissions
df = pd.merge(df_sensor, df_co2, on='timestamp')
