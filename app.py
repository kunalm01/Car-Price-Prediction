import streamlit as st
import pandas as pd
import pickle

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

# Set page config
st.set_page_config(page_title="Used Car Price Prediction", page_icon=":car:", layout="wide", initial_sidebar_state="expanded")

# Define the predict function
def predict_price(km_driven, no_year, fuel, seller_type, transmission, owner):

    # Create a dictionary of input values
    input_dict = {'km_driven': km_driven,
                  'no_year': no_year,
                  'fuel_Diesel': int(fuel=='Diesel'),
                  'fuel_Electric': int(fuel=='Electric'),
                  'fuel_LPG': int(fuel=='LPG'),
                  'fuel_Petrol': int(fuel=='Petrol'),
                  'seller_type_Individual': int(seller_type=='Individual'),
                  'seller_type_Trustmark Dealer': int(seller_type=='Trustmark Dealer'),
                  'transmission_Manual': int(transmission=='Manual'),
                  'owner_Fourth & Above Owner': int(owner=='Fourth & Above Owner'),
                  'owner_Second Owner': int(owner=='Second Owner'),
                  'owner_Test Drive Car': int(owner=='Test Drive Car'),
                  'owner_Third Owner': int(owner=='Third Owner')
                 }

    # Convert the input dictionary to a pandas dataframe
    input_df = pd.DataFrame([input_dict])

    # Use the model to predict the selling price
    prediction = model.predict(input_df)

    # Return the prediction
    return round(prediction[0], 2)

# Input form
with st.form(key='car_details'):
    st.markdown('## Enter the details of the car to predict its selling price')

    col1, col2 = st.columns(2)

    with col1:
        km_driven = st.number_input("KMs Driven (in thousands)", min_value=0, max_value=1000, step=1, key='km')
        no_year = st.number_input("No. of Years Old", min_value=0, max_value=50, step=1, key='age')
    with col2:
        fuel = st.selectbox('Select the fuel type', ("Petrol", "Diesel", "CNG", "LPG", "Electric"), key='fuel')
        seller_type = st.selectbox('Select the seller type', ('Individual', 'Dealer','Trustmark Dealer'), key='seller')
        transmission = st.selectbox('Select the transmission type', ('Manual', 'Automatic'), key='transmission')
        owner = st.selectbox('Select the number of previous owners', ('First Owner', 'Second Owner', 'Third Owner','Test Drive Car','Fourth & Above Owner'), key='owner')

    predict_button = st.form_submit_button(label='Predict')

# Get the prediction
if predict_button:
    prediction = predict_price(km_driven*1000, no_year, fuel, seller_type, transmission, owner)
    st.success('The predicted selling price is ₹{:.2f} lakhs'.format(prediction/100000))
