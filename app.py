import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained model
# pipe = pickle.load(open('pipe1.pkl', 'rb'))
with open("pipe1.pkl", "rb") as file:
    model = pickle.load(file)

# Define lists for teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 'Chennai',
          'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur',
          'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

# Streamlit app layout
st.title('IPL Win Predictor')

# Define columns for layout
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))

# Filter the bowling team options to exclude the selected batting team
filtered_bowling_teams = [team for team in teams if team != batting_team]

# Bowling team selection
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(filtered_bowling_teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=0)

# Define columns for input fields
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0)
with col5:
    wickets = st.number_input('Wickets out', min_value=0)


if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_remaining = 10 - wickets
    crr = score / (overs if overs > 0 else 1)  # Avoid division by zero
    rrr = (runs_left * 6) / (balls_left if balls_left > 0 else 1)  # Avoid division by zero

    # Prepare input dataframe
    input_df = pd.DataFrame({'batting_team': [batting_team],
                             'bowling_team': [bowling_team],
                             'city': [selected_city],
                             'runs_left': [runs_left],
                             'balls_left': [balls_left],
                             'wickets': [wickets_remaining],
                             'total_runs_x': [target],
                             'crr': [crr],
                             'rrr': [rrr]})

    # Predict probabilities
    result = model.predict_proba(input_df)  # Use the correct model variable
    loss = result[0][0]
    win = result[0][1]

    # Display results
    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")
