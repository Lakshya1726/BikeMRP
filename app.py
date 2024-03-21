import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load data and model
bike = pd.read_csv('bike.csv')
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Prediction function
def predict_price(bike_model, year, driven):
    # Ensure input data is in the correct format for prediction
    try:
        year = int(year)
        driven = float(driven)
        prediction = model.predict(pd.DataFrame({'Name': [bike_model], 'year': [year], 'kms': [driven]}))
        return prediction[0]
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app
st.title('Bike Price Prediction')

# Select box for bike model
bike_models = sorted(bike['Name'].unique())
bike_model = st.selectbox('Select Bike Model', options=bike_models)

# Select box for year
years = sorted(bike['year'].unique(), reverse=True)
year = st.selectbox('Select Year', options=years)

# Input for kilometers driven
driven = st.number_input('Enter Kilometers Driven')

# Button to trigger prediction
if st.button('Predict'):
    prediction = predict_price(bike_model, year, driven)
    if prediction is not None:
        st.write(f'Predicted Price: {np.round(prediction, 2)}')

# Sidebar
st.sidebar.title('About')
st.sidebar.info(
    "This app is a simple bike price prediction tool."
)
