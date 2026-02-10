import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Vehicle Maintenance Prediction", layout="wide")
st.title("🤖 Vehicle Maintenance Prediction")
st.info("Model is trained on internal data. User data is used only for prediction.")


# LOAD INTERNAL TRAINING DATA (FIXED)

DATA_URL = "https://raw.githubusercontent.com/codersnap/Prediction-model/master/vehicle_maintenance_dataset_1000_rows.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

with st.expander("📊 Training Dataset"):
    st.dataframe(df.head())

# MODEL FEATURES (FIXED SCHEMA)

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

X["driving_style"] = X["driving_style"].map({
    "Aggressive": 1,
    "Smooth": 0
})

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.success("✅ Model trained successfully")

# VISUALIZATION

with st.expander("📈 Data Visualization"):
    st.scatter_chart(
        data=df,
        x="km_since_last_service",
        y="months_since_last_service",
        color="needs_service"
    )


#  MANUAL INPUT 

st.header(" Manual Vehicle Check")

with st.sidebar:
    km_since_last_service = st.slider("Kilometers since last service", 0, 30000, 5000)
    total_km = st.slider("Total kilometers driven", 0, 300000, 50000)
    service_count = st.slider("Number of services done", 0, 15, 3)
    months_since_last_service = st.slider("Months since last service", 0, 36, 6)
    driving_style = st.selectbox("Driving style", ["Smooth", "Aggressive"])

manual_input = pd.DataFrame({
    "km_since_last_service": [km_since_last_service],
    "total_km": [total_km],
    "service_count": [service_count],
    "months_since_last_service": [months_since_last_service],
    "driving_style": [1 if driving_style == "Aggressive" else 0]
})

manual_pred = model.predict(manual_input)[0]
manual_conf = model.predict_proba(manual_input).max()

if manual_pred == 1:
    st.error(f"🚨 Service Required (Confidence: {manual_conf:.2f})")
else:
    st.success(f"✅ No Service Required (Confidence: {manual_conf:.2f})")


#  FILE UPLOAD OPTION 

st.header(" Predict Using Uploaded CSV File")

uploaded_file = st.file_uploader(
    "Upload your vehicle data CSV (any column names allowed)",
    type=["csv"]
)

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded File Preview")
    st.dataframe(user_df.head())

    st.subheader("🔗 Map Your Columns to Model Features")

    mapped_df = pd.DataFrame()

    column_labels = {
        "km_since_last_service": "Kilometers since last service",
        "total_km": "Total kilometers driven",
        "service_count": "Number of services",
        "months_since_last_service": "Months since last service",
        "driving_style": "Driving style"
    }

    for key, label in column_labels.items():
        selected_col = st.selectbox(
            f"Select column for **{label}**",
            user_df.columns
        )
        mapped_df[key] = user_df[selected_col]

    # Encode categorical column
    
    mapped_df["driving_style"] = (
    mapped_df["driving_style"]
    .astype(str)
    .str.strip()     # removes spaces
    .str.lower()     # normalizes case
    .replace({ "aggressive": 1,
              "smooth": 0})
    )

   

    if mapped_df["driving_style"].isnull().any():
        st.error("Driving style must contain only 'Aggressive' or 'Smooth'")
        st.stop()

    if st.button(" Predict from Uploaded File"):
        preds = model.predict(mapped_df)
        probs = model.predict_proba(mapped_df).max(axis=1)

        result_df = user_df.copy()
        result_df["Service Prediction"] = preds
        result_df["Confidence"] = probs
        result_df["Service Prediction"] = result_df["Service Prediction"].map({
            0: "No Service Required",
            1: "Service Required"
        })

        st.success("✅ Prediction completed")
        st.dataframe(result_df)

        st.download_button(
            "⬇️ Download Results",
            result_df.to_csv(index=False),
            "vehicle_service_predictions.csv",
            "text/csv"
        )

