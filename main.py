#
# Simple Linear Regression
# Afaan Bilal (https://afaan.dev)
#

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model

data = pd.read_csv('./data.csv')

st.title("Simple Linear Regression")
st.subheader("Afaan Bilal (https://afaan.dev)")

st.subheader("Input Data")
st.dataframe(data)

st.subheader("Size (sqft) vs Price ($100k)")
st.line_chart(data[['Size', 'Price']], x = 'Size', y = 'Price')

st.subheader("Rooms vs Price ($100k)")
st.line_chart(data[['Rooms', 'Price']], x = 'Rooms', y = 'Price')

st.subheader("Distance (miles) vs Price ($100k)")
st.line_chart(data[['Distance', 'Price']], x = 'Distance', y = 'Price')

model = linear_model.LinearRegression()
model.fit(data[['Size','Rooms','Distance']], data['Price'])

test_data = pd.read_csv('./test.csv')
pred = model.predict(test_data[['Size', 'Rooms', 'Distance']])

st.subheader("Test Data and Prediction")
test_data['Predicted'] = np.round(pred, 2)
test_data['Difference'] = test_data['Price'] - test_data['Predicted']
test_data['Error %'] = np.round((test_data['Price'] - test_data['Predicted']) * 100 / test_data['Price'], 2)
st.dataframe(test_data)
