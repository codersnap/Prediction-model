import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Vehicle Maintainance Prediciton ')

st.info('A machine learning model for prediction')

with st.expander('Data'):
    st.write('**Raw Data**')
    df=pd.read_csv('https://github.com/codersnap/Prediction-model/blob/master/car_service_dataset_1000_rows.csv')
    df

    st.write('**x**')
    x_raw=df.drop('species',axis=1)
    x_raw


    st.write('**y**')
    y_raw=df.species
    y_raw


