import streamlit as st
import requests
import pandas as pd

# Streamlit app
st.title("ML Model API Interaction")

st.write("""
This app interacts with an ML model served by a Flask API.
""")

# Input features from the user
st.sidebar.header('User Input Features')
def user_input_features():
    feature_1 = st.sidebar.slider('Feature 1', 0.0, 10.0, 5.0)
    feature_2 = st.sidebar.slider('Feature 2', 0.0, 10.0, 5.0)
    data = {'feature_1': feature_1, 'feature_2': feature_2}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input features
st.subheader('User Input features')
st.write(input_df)

# Make prediction using the Flask API
if st.button('Predict'):
    api_url = "http://127.0.0.1:5000/predict"
    input_data = input_df.values.tolist()[0]
    response = requests.post(api_url, json={'data': input_data})
    if response.status_code == 200:
        prediction = response.json()['prediction']
        st.subheader('Prediction')
        st.write(prediction)
    else:
        st.subheader('Error')
        st.write(response.json())
