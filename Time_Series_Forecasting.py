# Reading and Displaying BTC(Bitcoin) Time Series Data

import pandas_datareader.data as web
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as mp
from statsmodels.tsa.arima.model import ARIMA


# Relax the display limits on columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Data Collection/Import using datareader
btc = web.get_data_yahoo(['BTC-USD'], start=datetime.datetime(2018, 1, 1), end=datetime.datetime(2020, 12, 2))['Close']
print(btc.head())


# Convert BTC data to a csv file so to avoid having to repeatedly pull data using the Pandas data reader
btc.to_csv("btc.csv")


# Read csv file and display the first five rows : Adj Close
btc = pd.read_csv("btc.csv")
print(btc.head())


# To set the date column to be a data frame index, format that date using the to_datetime method
btc.index = pd.to_datetime(btc['Date'], format='%Y-%m-%d')
del btc['Date'] # Display df

# Plot time-series data : Format visualization using Seaborn
sns.set()
# Using matplotlib
plt.ylabel('BTC Price')
plt.xlabel('Date')
plt.xticks(rotation=45) # Rotate the dates on the x-axis so that they are easier to read
# Generate plot
plt.plot(btc.index, btc['BTC-USD'], )


# Splitting Data for Training and Testing
train = btc[btc.index < pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
test = btc[btc.index > pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
# Everything before November 2020 will serve as training data, with everything after 2020 becoming the testing data
# Train-Test Plot
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel('BTC Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for BTC Data")
plt.show()

# Define Input
y = train['BTC-USD']


# TimeSeries Model
# Autoregressive Moving Average(ARMA)
# Logic ARMA
ARMAmodel = SARIMAX(y, order = (1, 0, 1))
ARMAmodel = ARMAmodel.fit() # Model fit

# Generate Predictions
y_pred = ARMAmodel.get_forecast(len(test.index)) # Generate Predictions on same length as the test dataset
y_pred_df = y_pred.conf_int(alpha = 0.05) # Extract the confidence intervals (default 95%) for the forecasted values
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1]) 
#  To add a new column "Predictions" containing the actual predicted (mean) values from the model
y_pred_df.index = test.index # Align the prediction DataFrame's index with the test dataset index for comparison
y_pred_out = y_pred_df["Predictions"] # Extract only the "Predictions" column as the final predicted output series
# Plot result
plt.plot(y_pred_out, color='green', label = 'Predictions')
plt.legend()

# Evaluate the performance using the root mean-squared error(RMSE)
arma_rmse = np.sqrt(mean_squared_error(test["BTC-USD"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)


# TimeSeries Model
# Autoregressive Integrated Moving Average (ARIMA) Model
# Logic ARIMA
ARIMAmodel = ARIMA(y, order = (2, 2, 2)) # Differencing Parameters
ARIMAmodel = ARIMAmodel.fit()

# Generate Predictions
y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(y_pred_out, color='Yellow', label = 'ARIMA Predictions')
plt.legend()

# Evaluate the performance using the root mean-squared error(RMSE)
arma_rmse = np.sqrt(mean_squared_error(test["BTC-USD"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)


# TimeSeries Model
# Seasonal ARIMA (SARIMA) Model
# Logic SARIMA
SARIMAXmodel = SARIMAX(y, order = (5, 4, 2), seasonal_order=(2,2,2,12)) # Differencing Parameters
SARIMAXmodel = SARIMAXmodel.fit()

# Generate Predictions
y_pred = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(y_pred_out, color='Blue', label = 'SARIMA Predictions')
plt.legend()