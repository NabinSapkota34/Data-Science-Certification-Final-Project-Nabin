import streamlit as st
import pandas as pd
import joblib

# Load the trained machine learning model
# We use a try-except block to handle the case where the model file might be missing.
try:
    model = joblib.load('odi_runs_predictor.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run model_training.py to create the model.")
    st.stop()

# Set the page title and icon
st.set_page_config(page_title="ODI Runs Predictor", page_icon="🏏")

# Main title of the web app
st.title('🏏 ODI Batsman Career Runs Predictor')

# Project description
st.markdown("""
This web application predicts the total career runs of an ODI batsman using a **Random Forest Regressor** model.

**Instructions:**
1.  Enter the player's career statistics in the sidebar.
2.  Click the **Predict** button to see the estimated total career runs.
""")

# Create a sidebar for user inputs
st.sidebar.header('Enter Player Statistics')

# Function to collect user inputs
def user_input_features():
    mat = st.sidebar.number_input('Matches Played (Mat)', min_value=0, value=150)
    inns = st.sidebar.number_input('Innings Batted (Inns)', min_value=0, value=140)
    no = st.sidebar.number_input('Not Outs (NO)', min_value=0, value=15)
    hs = st.sidebar.number_input('Highest Score (HS)', min_value=0, value=110)
    ave = st.sidebar.number_input('Batting Average (Ave)', min_value=0.0, value=40.0, step=0.1, format="%.2f")
    bf = st.sidebar.number_input('Balls Faced (BF)', min_value=0, value=5000)
    sr = st.sidebar.number_input('Strike Rate (SR)', min_value=0.0, value=88.0, step=0.1, format="%.2f")
    num_100 = st.sidebar.number_input('Centuries (100)', min_value=0, value=8)
    num_50 = st.sidebar.number_input('Half-Centuries (50)', min_value=0, value=25)
    num_0 = st.sidebar.number_input('Ducks (0)', min_value=0, value=5)
    num_4s = st.sidebar.number_input('Fours (4s)', min_value=0, value=450)
    num_6s = st.sidebar.number_input('Sixes (6s)', min_value=0, value=80)
    career_length = st.sidebar.number_input('Career Length (Years)', min_value=0, value=12)

    # Store the inputs in a dictionary
    data = {
        'Mat': mat,
        'Inns': inns,
        'NO': no,
        'HS': hs,
        'Ave': ave,
        'BF': bf,
        'SR': sr,
        '100': num_100,
        '50': num_50,
        '0': num_0,
        '4s': num_4s,
        '6s': num_6s,
        'Career_Length': career_length
    }
    # Convert the dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display the user input in the main area
st.header('Player Input Statistics')
st.write(input_df)

# Create a button to make predictions
if st.button('**Predict Career Runs**'):
    # Ensure the column order matches the model's training data
    feature_order = ['Mat', 'Inns', 'NO', 'HS', 'Ave', 'BF', 'SR', '100', '50', '0', '4s', '6s', 'Career_Length']
    input_df = input_df[feature_order]

    # Make prediction using the loaded model
    prediction = model.predict(input_df)

    # Display the prediction result
    st.subheader('Prediction Result')
    st.success(f'Predicted Career Runs: **{int(prediction[0])}**')

st.markdown("---")
st.write("This project fulfills the requirements of a complete data science project, from data cleaning and EDA to model deployment.")