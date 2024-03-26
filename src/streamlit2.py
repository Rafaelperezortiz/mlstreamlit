import streamlit as st
from pickle import load
import pickle


st.title('Medical insurance price calculator')
# Introductory message
st.write("Welcome. Please enter the following data to predict charges for your insurance.")


#Users data

age_val = st.slider('Enter your age:',
                    min_value = 16,
                    max_value = 79,
                    step = 1
                    )

sex_val = st.selectbox('Select your gender:',
                       ('Male', 'Female')
                       )

bmi_val = st.slider('Enter your BMI:',
                    min_value = 15.80,
                    max_value = 51.20,
                    step = 0.01
                    )

children_val = st.number_input('Enter the number of children (if you have them):',
                               min_value = 0,
                               max_value = 5,
                               step = 1
                               )

smoker_val = st.selectbox('Are you a smoker:',
                          ('Yes', 'No')
                          )

region_val = st.selectbox('What region are you from?:',
                        ('Southwest', 'Southeast', 'Northwest', 'Northeast')
                        )

#load factorized values

fact_values = load(open('/workspaces/mlstreamlit/data/interim/fact_values.pk', 'rb'))

#button to predict

row = [age_val,
        fact_values['sex'][sex_val.lower()],
        bmi_val,
        children_val,
        fact_values['smoker'][smoker_val.lower()],
        fact_values['region'][region_val.lower()]
         ]

if st.button('Predict:'):
    normal_scaler = load(open('../models/normal_scaler.pk', 'rb'))
    scaler_row = normal_scaler.transform([row])

    model = load(open('../models/linear_regression.pk', 'rb'))
    y_pred = model.predict([row])

    st.text('The price of the insurance would be:' +str(round(y_pred[0, 0], 2)))