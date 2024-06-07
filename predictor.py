import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Download historical data as a dataframe
data = yf.download('META', start='2017-01-01', end='2023-12-31')

# Prepare data for linear regression model
X = np.array(range(len(data))).reshape(-1, 1)
y = data['Adj Close'].values

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict future stock prices (additional year into the future)
future_X = np.array(range(len(data) + 365)).reshape(-1, 1)
predicted_y = model.predict(future_X)

# Plot actual and predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Adj Close'], label='Actual Stock (Meta)', color='black')
plt.plot(
    data.index[-len(predicted_y):],
    predicted_y,
    label='Predicted Stock',
    linestyle='--',
    color='green'
)
plt.title('Meta Stock Price: Historical and Predicted (Adjusted)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()
