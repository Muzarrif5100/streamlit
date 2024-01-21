import streamlit as st
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from art import text2art
import datetime
import pandas as pd

today = datetime.date.today()

# Generate ASCII art as logo
ascii_art = text2art("BCAS")

st.write("============================================================")
# Print the generated ASCII art
st.write(ascii_art)
st.write("Bitcoin price prediction")
st.write("============================================================")
st.write("Created by: Muzarrif Ahamed")
st.write("BCAS_Kalmunai_Campus")
st.write("Network Engineering")
st.write("CSD-17 Batch")
st.write("Support my work:")
st.write("BTC, ETH & BNB (ERC20): 0xeab77fbd758df735ac79e11e95c649e9883ca10f")
st.write("USDT (TRC20): TRtEtci9t6SXSzqW8SyudVXvM1FeQSow65")
st.write("============================================================")

# Download historical data
st.write("Downloading historical Bitcoin data for training...")
bitcoin = yf.download('BTC-USD', start='2010-07-17')
st.write("Downloaded.")
st.write("Training...")

# Prepare data for model
bitcoin['Prediction'] = bitcoin['Close'].shift(-1)
bitcoin.dropna(inplace=True)
X = np.array(bitcoin.drop(['Prediction'], axis=1))
Y = np.array(bitcoin['Prediction'])

# Split data into training set and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Make predictions
bitcoin['Prediction'] = model.predict(np.array(bitcoin.drop(['Prediction'], axis=1)))
st.write("Training complete.")

# Evaluate the model
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.write("Mean Absolute Error on the test set: {:.2f}".format(mae))
st.write("Mean Squared Error on the test set: {:.2f}".format(mse))

# Print the predicted price for yesterday, today, tomorrow, 7 days, 30 days, and 1 year from now
st.write("Actual price for yesterday: ", bitcoin['Close'].iloc[-2])
st.write("Predicted price for today: ", bitcoin['Close'].iloc[-1])
st.write("Predicted price for tomorrow: ", bitcoin['Prediction'].iloc[-1])
st.write("Predicted price for 7 days: ", bitcoin['Prediction'].iloc[-7])
st.write("Predicted price for 30 days: ", bitcoin['Prediction'].iloc[-30])
st.write("Predicted price for 1 year: ", bitcoin['Prediction'].iloc[-365])
st.write("============================================================")

# User interface for entering a specific date
input_date_str = st.text_input("Enter a date (YYYY-MM-DD) to see the Bitcoin price:")
if input_date_str:
    try:
        input_date = pd.to_datetime(input_date_str, format='%Y-%m-%d')
        input_date_price = bitcoin.loc[input_date, 'Close']
        st.write(f"Bitcoin price on {input_date_str}: {input_date_price}")
    except KeyError:
        st.write(f"No data available for {input_date_str}. Please enter a valid date.")
    except ValueError:
        st.write("Invalid date format. Please enter the date in YYYY-MM-DD format.")

# Display the table with detailed information
st.write("Detailed Predictions Table:")
predictions_table = bitcoin[['Close', 'Prediction']].tail(10)  # Adjust the number of rows as needed
st.write(predictions_table)
st.write("============================================================")

# Prevent the window from closing immediately
st.write('Press ENTER to exit')
