import streamlit as st
import numpy as np
import pandas as pd


st.title('🤖 Vehicle Maintainance Prediciton ')

st.info('A machine learning model for prediction')

with st.expander('Data'):
    st.write('**Raw Data**')
    df=pd.read_csv('https://raw.githubusercontent.com/codersnap/Prediction-model/master/car_service_dataset_1000_rows.csv')
    df

with st.expander('X_RAW'):
    st.write('**X**')
    x_raw=df.drop('needs_service',axis=1)
    x_raw

with st.expander('Y_Raw'):
    st.write('**Y**')
    y_raw=df.needs_service
    y_raw


