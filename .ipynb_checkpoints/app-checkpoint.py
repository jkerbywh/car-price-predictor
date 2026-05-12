import streamlit as st
import pandas as pd
import joblib

model = joblib.load("car_price_model.pkl")

st.title("Car Price Predictor")

st.write("Enter car details below:")

horsepower = st.slider("Horsepower", 40, 300, 100)
enginesize = st.slider("Engine Size", 50, 350, 120)
curbweight = st.slider("Curb Weight", 1400, 4200, 2500)
carwidth = st.slider("Car Width", 55.0, 75.0, 65.0)
citympg = st.slider("City MPG", 10, 60, 25)
highwaympg = st.slider("Highway MPG", 10, 60, 30)

input_data = pd.DataFrame({
    'horsepower': [horsepower],
    'enginesize': [enginesize],
    'curbweight': [curbweight],
    'carwidth': [carwidth],
    'citympg': [citympg],
    'highwaympg': [highwaympg]
})

prediction = model.predict(input_data)

st.subheader("Predicted Car Price")
st.write(f"${prediction[0]:,.2f}")

st.subheader("Selected Inputs")
st.write(input_data)

st.write("Model R² Score: 0.955")
st.write("Average Error: ~$1,329")
