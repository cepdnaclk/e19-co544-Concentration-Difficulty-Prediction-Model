import streamlit as st
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from streamlit_modal import Modal

# Load the trained model
model_path = "best_model.joblib"
model: MLPClassifier = joblib.load(model_path)

# Define the app title and description
st.title("Difficulty Level Prediction App")
st.write("Enter the details below to predict the difficulty level.")

# Create columns for two-column layout
col1, col2 = st.columns(2)

# Define the input fields in two columns
with col1:
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", options=[0, 1, 2])
    relationship_status = st.selectbox("Relationship Status", options=[0, 1, 2])
    occupation = st.selectbox("Occupation", options=[0, 1, 2])
    social_media_user = st.selectbox("Social Media User?", options=["Yes", "No"])
    time_spent = st.number_input("Time Spent on Social Media (in hours)", min_value=0)

with col2:
    q1 = st.number_input("Q1", min_value=1, max_value=5)
    q2 = st.number_input("Q2", min_value=1, max_value=5)
    q3 = st.number_input("Q3", min_value=1, max_value=5)
    q4 = st.number_input("Q4", min_value=1, max_value=5)
    q5 = st.number_input("Q5", min_value=1, max_value=5)
    q6 = st.number_input("Q6", min_value=1, max_value=5)
    q7 = st.number_input("Q7", min_value=1, max_value=5)
    q8 = st.number_input("Q8", min_value=1, max_value=5)

# Social media platforms usage
col3, col4, col5 = st.columns(3)
with col3:
    facebook = st.selectbox("Facebook", options=["Yes", "No"])
    snapchat = st.selectbox("Snapchat", options=["Yes", "No"])
    reddit = st.selectbox("Reddit", options=["Yes", "No"])

with col4:
    instagram = st.selectbox("Instagram", options=["Yes", "No"])
    twitter = st.selectbox("Twitter", options=["Yes", "No"])
    pinterest = st.selectbox("Pinterest", options=["Yes", "No"])

with col5:
    youtube = st.selectbox("YouTube", options=["Yes", "No"])
    discord = st.selectbox("Discord", options=["Yes", "No"])
    tiktok = st.selectbox("TikTok", options=["Yes", "No"])

# Convert categorical inputs to numerical
def convert_to_numeric(value):
    return 1 if value == "Yes" else 0

social_media_user = convert_to_numeric(social_media_user)
facebook = convert_to_numeric(facebook)
instagram = convert_to_numeric(instagram)
youtube = convert_to_numeric(youtube)
snapchat = convert_to_numeric(snapchat)
twitter = convert_to_numeric(twitter)
discord = convert_to_numeric(discord)
reddit = convert_to_numeric(reddit)
pinterest = convert_to_numeric(pinterest)
tiktok = convert_to_numeric(tiktok)

# Prepare the input data for prediction
input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "Relationship Status": [relationship_status],
    "Occupation": [occupation],
    "Social Media User?": [social_media_user],
    "Time Spent": [time_spent],
    "Q1": [q1],
    "Q2": [q2],
    "Q3": [q3],
    "Q4": [q4],
    "Q5": [q5],
    "Q6": [q6],
    "Q7": [q7],
    "Q8": [q8],
    "Facebook": [facebook],
    "Instagram": [instagram],
    "YouTube": [youtube],
    "Snapchat": [snapchat],
    "Twitter": [twitter],
    "Discord": [discord],
    "Reddit": [reddit],
    "Pinterest": [pinterest],
    "TikTok": [tiktok],
})

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"The predicted difficulty level is: {prediction[0]}")
