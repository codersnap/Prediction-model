import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


st.title("🤖 Vehicle Maintenance Prediction")
st.info("A machine learning model to predict whether vehicle servicing is required")

# -------------------------------------------------
# LOAD DATA (CORRECT RAW GITHUB LINK)
# -------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/codersnap/Prediction-model/master/vehicle_maintenance_dataset_1000_rows.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

# -------------------------------------------------
# SHOW DATA
# -------------------------------------------------
with st.expander("📊 Raw Dataset"):
    st.dataframe(df.head())

# -------------------------------------------------
# PREPARE TRAINING DATA
# -------------------------------------------------
FEATURES = [
    "km_since_last_service",
    "total_km",
    "service_count",
    "months_since_last_service",
    "driving_style"
]

TARGET = "needs_service"

X = df[FEATURES].copy()
y = df[TARGET]

# Encode categorical feature
X["driving_style"] = X["driving_style"].map({
    "Aggressive": 1,
    "Smooth": 0
})

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.success("✅ Model trained successfully")

# -------------------------------------------------
# DATA VISUALIZATION (SAFE COLUMNS ONLY)
# -------------------------------------------------
with st.expander("📈 Data Visualization"):
    st.scatter_chart(
        data=df,
        x="km_since_last_service",
        y="months_since_last_service",
        color="needs_service"
    )

# -------------------------------------------------
# USER INPUT (REALISTIC FEATURES)
# -------------------------------------------------
with st.sidebar:
    st.header("🧑 Vehicle Details")

    km_since_last_service = st.slider(
        "Kilometers since last service", 0, 30000, 5000
    )
    total_km = st.slider(
        "Total kilometers driven", 0, 300000, 50000
    )
    service_count = st.slider(
        "Number of services done", 0, 15, 3
    )
    months_since_last_service = st.slider(
        "Months since last service", 0, 36, 6
    )
    driving_style = st.selectbox(
        "Driving style", ["Smooth", "Aggressive"]
    )

# -------------------------------------------------
# CREATE INPUT DATAFRAME
# -------------------------------------------------
input_df = pd.DataFrame({
    "km_since_last_service": [km_since_last_service],
    "total_km": [total_km],
    "service_count": [service_count],
    "months_since_last_service": [months_since_last_service],
    "driving_style": [1 if driving_style == "Aggressive" else 0]
})

with st.expander("🔎 Input Features Used for Prediction"):
    st.dataframe(input_df)

# PREDICTION

prediction = model.predict(input_df)[0]
confidence = model.predict_proba(input_df).max()

st.subheader("🔮 Prediction Result")

if prediction == 1:
    st.error(f"🚨 Service Required (Confidence: {confidence:.2f})")
else:
    st.success(f"✅ No Service Required (Confidence: {confidence:.2f})")





