# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
# === Import libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# === Load the dataset ===
data = pd.read_csv("AirPassengers.csv")

# Convert 'Month' column to datetime format
data['Month'] = pd.to_datetime(data['Month'], errors='coerce')

# Drop missing or invalid dates
data.dropna(subset=['Month'], inplace=True)

# Sort by date
data = data.sort_values(by='Month')

# Set 'Month' as index
data.set_index('Month', inplace=True)

# Display first few rows
print(data.head())

# === Select the target variable for time series analysis ===
target_col = '#Passengers'

# === Plot the time series ===
plt.figure(figsize=(12, 6))
plt.plot(data.index, data[target_col], label=target_col, color='blue')
plt.xlabel('Month')
plt.ylabel(target_col)
plt.title(f'{target_col} Time Series')
plt.legend()
plt.grid()
plt.show()

# === Function to check stationarity ===
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# === Check stationarity of the series ===
print("\n--- Stationarity Test for #Passengers ---")
check_stationarity(data[target_col])

# === Plot ACF and PACF ===
plt.figure(figsize=(10, 4))
plot_acf(data[target_col].dropna(), lags=30)
plt.title("Autocorrelation Function (ACF)")
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(data[target_col].dropna(), lags=30)
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()

# === Train-Test Split ===
train_size = int(len(data) * 0.8)
train, test = data[target_col][:train_size], data[target_col][train_size:]

# === Build and fit SARIMA model ===
# The seasonal_order=(1,1,1,12) suits monthly data like AirPassengers
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False)

# === Forecast ===
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# === Evaluate performance ===
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(f'\nRoot Mean Squared Error (RMSE): {rmse:.4f}')

# === Plot predictions vs actuals ===
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.xlabel('Month')
plt.ylabel(target_col)
plt.title(f'SARIMA Model Predictions for {target_col}')
plt.legend()
plt.grid()
plt.show()
```

### OUTPUT:

<img width="216" height="143" alt="image" src="https://github.com/user-attachments/assets/1df648b1-bee2-4f0f-9298-2cc6d924b0a6" />

<img width="1119" height="604" alt="image" src="https://github.com/user-attachments/assets/88b1f65c-1ab8-4fe5-95a1-1c3e1e1e917d" />

<img width="364" height="157" alt="image" src="https://github.com/user-attachments/assets/19303142-98e7-495e-a308-38ce365f4306" />

<img width="649" height="501" alt="image" src="https://github.com/user-attachments/assets/b38660a3-aa10-49e8-ad0e-998aa0d8ecce" />

<img width="628" height="487" alt="image" src="https://github.com/user-attachments/assets/7ea88130-0213-4348-8c5b-a75e2529c99d" />

<img width="362" height="46" alt="image" src="https://github.com/user-attachments/assets/0342edbe-d011-4dd6-9106-c6dbf1560b36" />

<img width="1115" height="590" alt="image" src="https://github.com/user-attachments/assets/8f26bfdd-0d5e-404a-8504-4e36e8c01ea0" />


### RESULT:
Thus the program run successfully based on the SARIMA model.
