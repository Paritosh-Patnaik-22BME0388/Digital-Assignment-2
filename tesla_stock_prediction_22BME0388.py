# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 20:49:49 2025

@author: pc
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
tesla_data = pd.read_csv('C:/Users/pc/OneDrive/Desktop/VIT Documents/My Curriculum/Sem-6/BMEE407L/archiv/tesla_stock_data_2000_2025.csv')

# Convert Date column to datetime and sort
tesla_data['Date'] = pd.to_datetime(tesla_data['Date'])
tesla_data = tesla_data.sort_values(by='Date')
tesla_data.set_index('Date', inplace=True)

# Select features and target variable
features = ['Open', 'High', 'Low', 'Volume']  # Excluding 'Close' to predict it
target = 'Close'

X = tesla_data[features]
y = tesla_data[target]

dates = tesla_data.index  # Store dates for plotting

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, dates, test_size=0.2, shuffle=False, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(dates_test, y_test, label='Actual Prices', color='blue')
plt.plot(dates_test, y_pred, label='Predicted Prices', color='red', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Tesla Stock Price Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.show()
