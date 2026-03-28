import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="p-kansal/capstone-project", filename="best_capstone_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Engine Status Prediction App")
st.write("""
This application predicts the likelihood of engine failing based on the engine health parameters and sensor information.
Please enter the sensor readings data below to get a prediction.
""")

# User input
rpm = st.number_input("Engine RPM", min_value=10, max_value=3000, value=700, step=1)
LubOilPressure = st.number_input("Lubricating Oil Pressure", min_value=0.0, max_value=10.0, value=3.0, step=0.01)
FuelPressure = st.number_input("Fuel Pressure", min_value=0.0, max_value=25.0, value=7.0, step=0.01)
CoolantPressure = st.number_input("Engine Coolant Pressure", min_value=0.0, max_value=10.0, value=2.0, step=0.01)
LubOilTemp = st.number_input("Lubricating Oil Temperature (°C)", min_value=60.0, max_value=100.0, value=75.0, step=0.01)
CoolantTemp = st.number_input("Engine Coolant Temperature (°C)", min_value=60.0, max_value=100.0, value=75.0, step=0.01)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine_rpm': rpm,
    'Lub_oil_pressure': LubOilPressure,
    'Fuel_pressure': FuelPressure,
    'Coolant_pressure': CoolantPressure,
    'lub_oil_temp': LubOilTemp,
    'Coolant_temp': CoolantTemp
}])

if st.button("Predict Failure Likelihood"):
    prediction = model.predict(input_data)[0]
    result = "Fail" if prediction == 1 else "Not Fail"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts that the engine will **{result}** based on sensor readings")
