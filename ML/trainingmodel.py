from sklearn.model_selection import train_test_split

# Assuming `df` is your DataFrame with features and the target variable (CO2 emissions)
features = df[['CO', 'PM10', 'PM2.5', 'AQI', 'CO_MA', 'CO_lag_1']].values
target = df['CO2_emissions'].values

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_reg.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Make predictions on the test set
y_pred = rf_reg.predict(X_test)

# Evaluate the model using MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
scores = cross_val_score(rf_reg, features, target, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE to positive RMSE
rmse_scores = np.sqrt(-scores)

print(f"Cross-validated RMSE: {rmse_scores.mean()} (± {rmse_scores.std()})")
