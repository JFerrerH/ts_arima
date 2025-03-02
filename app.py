import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Streamlit UI
st.title("ARIMA Time Series Model")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

def load_data(file):
    data = pd.read_csv(file)
    return data

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Data Preview:", data.head())
    
    # Allow user to select the time series column
    time_column = st.selectbox("Select the Date column:", data.columns)
    value_column = st.selectbox("Select the Value column:", [col for col in data.columns if col != time_column], index=0)
    
    # Convert date column to datetime
    data[time_column] = pd.to_datetime(data[time_column], errors='coerce')
    data.set_index(time_column, inplace=True)
    
    # Drop missing values only in the selected column
    data[value_column] = data[value_column].dropna()
    
    # Display time series plot
    st.subheader("Time Series Plot")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data[value_column])
    ax.set_title("Time Series Data")
    st.pyplot(fig)
    
    # Dickey-Fuller Test for stationarity
    st.subheader("Dickey-Fuller Test for Stationarity")
    def adf_test(timeseries):
        result = adfuller(timeseries.dropna())
        st.write(f"ADF Statistic: {result[0]}")
        st.write(f"p-value: {result[1]}")
        st.write("Critical Values:")
        for key, value in result[4].items():
            st.write(f"   {key}: {value}")
        return result[1]  # return p-value
    
    p_value = adf_test(data[value_column])
    
    # Automatically determine differencing order d
    d = 0
    differenced_series = data[value_column]
    while p_value > 0.05 and d < 5:
        differenced_series = differenced_series.diff().dropna()
        d += 1
        p_value = adf_test(differenced_series)
    
    st.write(f"Optimal differencing order (d): {d}")
    
    # Plot ACF and PACF with confidence intervals
    if p_value <= 0.05:
        st.subheader("Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(differenced_series.dropna(), ax=axes[0], lags=20)
        axes[0].set_title("Autocorrelation Function (ACF)")
        plot_pacf(differenced_series.dropna(), ax=axes[1], lags=20)
        axes[1].set_title("Partial Autocorrelation Function (PACF)")
        st.pyplot(fig)
    
    # ARIMA model parameters
    st.subheader("ARIMA Model Hyperparameters")
    p = st.number_input("Enter p (AR term)", min_value=0, max_value=20, value=1)
    d = st.number_input("Enter d (Differencing term)", min_value=0, max_value=5, value=d)
    q = st.number_input("Enter q (MA term)", min_value=0, max_value=20, value=1)

    st.subheader("ARIMA Forecast")
    timePeriod = st.number_input("Enter periods", min_value=0, max_value=50, value=10)

    if st.button("Train ARIMA Model"):
        model = ARIMA(data[value_column].dropna(), order=(p, d, q))
        fitted_model = model.fit()
        
        st.subheader("Model Summary")
        st.text(fitted_model.summary())
        
        # Forecast
        st.subheader("Forecast Plot")
        forecast = fitted_model.forecast(steps=timePeriod)
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(data[value_column], label="Actual")
        ax2.plot(forecast, label="Forecast", linestyle="dashed")
        ax2.legend()
        st.pyplot(fig2)
        st.write("Forecast Data:", forecast)
