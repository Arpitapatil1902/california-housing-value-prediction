import os
 
import joblib
import pandas as pd
import streamlit as st
 
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
 
FEATURE_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
]
OCEAN_OPTIONS = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
 
st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="wide")
 
 
@st.cache_resource
def load_artifacts():
    """Load the trained model and preprocessing pipeline created by main_new.py."""
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    return model, pipeline
 
 
def predict(df: pd.DataFrame, model, pipeline):
    transformed = pipeline.transform(df[FEATURE_COLUMNS])
    return model.predict(transformed)
 
 
st.title("🏠 California House Price Predictor")
st.caption("Predicts the median house value of a block group using the Random Forest model trained in main_new.py.")
 
# The model must be trained first by running main_new.py
if not (os.path.exists(MODEL_FILE) and os.path.exists(PIPELINE_FILE)):
    st.error("model.pkl and pipeline.pkl were not found in this folder.")
    st.info("Run `python main_new.py` once (with housing.csv in the same folder) to train the model, then restart this app.")
    st.stop()
 
model, pipeline = load_artifacts()
 
tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction (CSV)"])
 
# ---------------- Single prediction ----------------
with tab_single:
    st.subheader("Enter the block group details")
    col1, col2, col3 = st.columns(3)
 
    with col1:
        longitude = st.number_input("Longitude", value=-118.49, min_value=-125.0, max_value=-114.0, step=0.01, format="%.2f")
        latitude = st.number_input("Latitude", value=34.26, min_value=32.0, max_value=42.5, step=0.01, format="%.2f")
        housing_median_age = st.number_input("Housing median age (years)", value=29, min_value=1, max_value=100, step=1)
 
    with col2:
        total_rooms = st.number_input("Total rooms", value=2127, min_value=1, step=10)
        total_bedrooms = st.number_input("Total bedrooms", value=435, min_value=1, step=10)
        population = st.number_input("Population", value=1166, min_value=1, step=10)
 
    with col3:
        households = st.number_input("Households", value=409, min_value=1, step=10)
        median_income = st.number_input(
            "Median income (in tens of thousands of USD)",
            value=3.87,
            min_value=0.0,
            max_value=20.0,
            step=0.1,
            help="For example, 3.5 means about $35,000.",
        )
        ocean_proximity = st.selectbox("Ocean proximity", OCEAN_OPTIONS)
 
    if st.button("Predict price", type="primary"):
        row = pd.DataFrame(
            [
                {
                    "longitude": longitude,
                    "latitude": latitude,
                    "housing_median_age": housing_median_age,
                    "total_rooms": total_rooms,
                    "total_bedrooms": total_bedrooms,
                    "population": population,
                    "households": households,
                    "median_income": median_income,
                    "ocean_proximity": ocean_proximity,
                }
            ]
        )
        prediction = predict(row, model, pipeline)[0]
 
        st.success("Prediction complete")
        st.metric("Predicted median house value", f"${prediction:,.0f}")
        st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=6)
 
# ---------------- Batch prediction ----------------
with tab_batch:
    st.subheader("Upload a CSV file")
    st.write("Required columns: " + ", ".join(f"`{c}`" for c in FEATURE_COLUMNS))
    st.caption("Extra columns (such as median_house_value) are kept in the output but not used for prediction.")
 
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
 
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
 
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            data["predicted_median_house_value"] = predict(data, model, pipeline)
            st.write(f"Predictions for {len(data):,} rows:")
            st.dataframe(data, use_container_width=True)
            st.download_button(
                "Download results as CSV",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
 