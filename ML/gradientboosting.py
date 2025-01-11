import xgboost as xgb

# Convert data to DMatrix format (optional but recommended for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Initialize and train the XGBoost Regressor
xgboost_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgboost_reg.fit(X_train, y_train)

# Make predictions
y_pred = xgboost_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"XGBoost RMSE: {rmse}")
import lightgbm as lgb

# Initialize and train the LightGBM Regressor
lgb_reg = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
lgb_reg.fit(X_train, y_train)

# Make predictions
y_pred = lgb_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"LightGBM RMSE: {rmse}")
