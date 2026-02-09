import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


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
    km_since_last_service=st.slider('km since last service',0,15000,7500)
    avg_speed_kmph=st.slider('Avg speed(km/hr)',0,120,60)
    engine_oil_km=st.slider('Engine Oil',0,20000,10000)
    brake_pad_thickness_mm=st.slider('Break pad thickness(mm)',0.0,15.0,7.5)
    clutch_pad_thickness_mm=st.slider('Clutch pad thickness(mm)',0.0,10.0,5.0)
    driving_style=st.selectbox('Driving-Style',('Aggressive','Smooth'))


    data={
        'km_since_last_service':km_since_last_service,
        'avg_speed_kmph':avg_speed_kmph,
        'engine_oil_km':engine_oil_km,
        'brake_pad_thickness_mm':brake_pad_thickness_mm,
        'clutch_pad_thickness_mm':clutch_pad_thickness_mm,
        'driving_style':driving_style
         }
    input_df=pd.DataFrame(data,index=[0])
    
    # Encode categorical input
    input_df['driving_style'] = input_df['driving_style'].map({
    'Aggressive': 1,
    'Smooth': 0
    })

    input_cars=pd.concat([input_df,x_raw],axis=0)
    
with st.expander('Input Features'):
    st.write('**Input feature**')
    input_df
    st.write('**Combined feature Data**')
    input_cars

X = input_cars.iloc[1:]      # training features
input_row = input_cars.iloc[:1]  # user input
y = y_raw

#model

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

#mapping the output
service_map = {
    0: 'No Service Required',
    1: 'Service Required'
}

result = service_map[prediction[0]]

st.subheader('Prediction Result')

if prediction[0] == 1:
    st.error(f'🚨 {result}')
else:
    st.success(f'✅ {result}')


