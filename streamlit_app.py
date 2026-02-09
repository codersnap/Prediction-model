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

with st.expander('Data Visualisation'):
    st.write('**Scatter Plot**')
    st.scatter_chart(data=df,x='km_since_last_service',y='brake_pad_thickness_mm',color='needs_service')

with st.sidebar:
    st.header('Input features')
    km_since_last_service=st.slider('km since last service',0,100000,50000)
    avg_speed_kmph=st.slider('Avg speed(km/hr)',0,120,60)
    engine_oil_km=st.slider('Engine Oil',0,20000,10000)
    brake_pad_thickness_mm=st.slider('Break pad thickness(mm)',0.0,15.0,7.5)
    clutch_pad_thickness_mm=st.slider('Clutch pad thickness(mm)'0.0,10.0,5.0)
    driving_style=st.selectbox('Driving-Style'('Agressive','Smooth'))
