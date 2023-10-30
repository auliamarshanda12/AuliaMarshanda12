import pickle
import numpy as np
import streamlit as st

model = pickle.load(open('PrediksiBoston3.sav', 'rb'))

st.title('Prediksi Harga Rumah Boston')

INDUS = st.number_input('input INDUS')
CHAS = st.number_input('input CHAS')
NOX = st.number_input('input NOX')
RM = st.number_input('input RM')
AGE = st.number_input('input AGE')
TAX = st.number_input('input TAX')
LSTAT = st.number_input('input LSTAT')

    
prediksi=''
if st.button('Prediksi Harga Rumah Boston'):
    prediksi_boston = model.predict([[INDUS, CHAS,	NOX,	RM,	AGE,	TAX,	LSTAT]])
    prediksi=prediksi_boston

st.success(prediksi)
