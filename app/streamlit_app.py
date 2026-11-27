"""ROUTEX interactive fuel-economy prediction demo.

Run from repository root:
    streamlit run app/streamlit_app.py
"""
from pathlib import Path
import sys
import warnings

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xgboost.joblib"
sys.path.insert(0, str(ROOT))

from src.member3_data_quality import fix_passenger_volume  # noqa: E402

st.set_page_config(page_title="ROUTEX MPG Predictor", page_icon="🚗", layout="wide")
st.title("🚗 ROUTEX — Vehicle Fuel Economy Predictor")
st.caption("Educational prototype using the trained XGBoost regression pipeline.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python -m src.member3_advanced_models` "
            "from the project root first."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

st.info(
    "Enter vehicle specifications below. The result is an estimated combined MPG "
    "for vehicles within the model's training scope; pure battery-electric vehicles "
    "are not supported."
)

left, right = st.columns(2)
with left:
    year = st.number_input("Model year", min_value=1984, max_value=2021, value=2018, step=1)
    make = st.text_input("Manufacturer (make)", value="Toyota")
    vclass = st.selectbox("Vehicle class", [
        "Compact Cars", "Subcompact Cars", "Midsize Cars", "Large Cars",
        "Two Seaters", "Small Sport Utility Vehicle 2WD",
        "Sport Utility Vehicle - 4WD", "Standard Pickup Trucks"
    ])
    drive = st.selectbox("Drive type", [
        "Front-Wheel Drive", "Rear-Wheel Drive",
        "4-Wheel or All-Wheel Drive", "All-Wheel Drive",
        "4-Wheel Drive", "2-Wheel Drive"
    ])
    trany = st.selectbox("Transmission", [
        "Automatic 6-spd", "Automatic 8-spd", "Automatic 10-spd",
        "Manual 5-spd", "Manual 6-spd", "Automatic (AV-S6)"
    ])
    trans_dscr = st.text_input("Transmission description", value="Synchronous")
    fuel_type = st.selectbox("Fuel type", ["Regular", "Premium", "Diesel", "Gasoline or E85"])
    fuel_type1 = st.selectbox("Primary fuel", ["Regular Gasoline", "Premium Gasoline", "Diesel", "Midgrade Gasoline"])
with right:
    displ = st.number_input("Engine displacement (litres)", min_value=0.6, max_value=8.5, value=2.0, step=0.1)
    cylinders = st.selectbox("Number of cylinders", [3, 4, 5, 6, 8, 10, 12, 16], index=1)
    trans_speeds = st.number_input("Transmission speeds", min_value=1, max_value=10, value=6, step=1)
    passenger_volume = st.number_input("Passenger volume (cu ft)", min_value=40.0, max_value=200.0, value=95.0, step=1.0)
    luggage_volume = st.number_input("Luggage volume (cu ft)", min_value=1.0, max_value=80.0, value=15.0, step=1.0)
    turbo = st.selectbox("Turbocharged", [0, 1], format_func=lambda x: "Yes" if x else "No")
    supercharger = st.selectbox("Supercharged", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_cvt = st.selectbox("Continuously variable transmission (CVT)", [0, 1], format_func=lambda x: "Yes" if x else "No")
    phev_blended = st.selectbox("Blended plug-in hybrid", [0, 1], format_func=lambda x: "Yes" if x else "No")

# Keep the input schema identical to src.preprocessing.FEATURES.
trans_type = str(trany).split()[0]
row = {
    "year": float(year), "displ": float(displ), "cylinders": float(cylinders),
    "trans_speeds": float(trans_speeds), "passenger_volume": float(passenger_volume),
    "luggage_volume": float(luggage_volume), "turbo": int(turbo),
    "supercharger": int(supercharger), "is_cvt": int(is_cvt),
    "phevBlended": int(phev_blended), "make": make.strip() or "Unknown",
    "VClass": vclass, "drive": drive, "trany": trany, "trans_type": trans_type,
    "trans_dscr": trans_dscr, "fuelType": fuel_type, "fuelType1": fuel_type1,
    "fuelType2": "Single fuel", "atvType": "Conventional",
    "startStop": "Not reported", "eng_dscr": "Not specified"
}
input_df = pd.DataFrame([row])
input_df = fix_passenger_volume(input_df)

if st.button("Predict combined fuel economy", type="primary", use_container_width=True):
    try:
        prediction = float(model.predict(input_df)[0])
        st.success(f"Estimated combined fuel economy: **{prediction:.1f} MPG**")
        st.caption("This is a model estimate, not an official EPA test result.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

with st.expander("About this model"):
    st.write(
        "The supplied project evaluation selected XGBoost by the lowest five-fold "
        "cross-validation RMSE (0.77558 MPG). On the held-out test set it achieved "
        "RMSE 0.71261 MPG and R² 0.98242. Model predictions depend on the quality "
        "and representativeness of the training data."
    )
