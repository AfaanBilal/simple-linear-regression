#
# Simple Linear Regression
# Afaan Bilal (https://afaan.dev)
#

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model

st.set_page_config(
    page_title="Simple Linear Regression",
    page_icon="📈",
    menu_items={
        'About': 'Author: Afaan Bilal (https://afaan.dev) \n\nSource code: https://github.com/AfaanBilal/simple-linear-regression',
        'Report a bug': "https://github.com/AfaanBilal/simple-linear-regression/issues",
    }
)

data = pd.read_csv('./data.csv')

st.title("Simple Linear Regression")
st.subheader("Afaan Bilal (https://afaan.dev)")

st.subheader("Input Data")
st.dataframe(data, use_container_width=True)

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
test_data['Predicted'] = np.round(pred, 2)

st.subheader("Test Data and Prediction")
st.line_chart(test_data[['Size', 'Price', 'Predicted']], x = 'Size', y = ['Price', 'Predicted'])

test_data['Difference'] = test_data['Price'] - test_data['Predicted']
test_data['Error %'] = np.round((test_data['Price'] - test_data['Predicted']) * 100 / test_data['Price'], 2)
st.dataframe(test_data, use_container_width=True)
