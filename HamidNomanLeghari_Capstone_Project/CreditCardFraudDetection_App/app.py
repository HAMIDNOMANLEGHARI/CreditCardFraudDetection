import streamlit as st
import pandas as pd
import xgboost as xgb
from pipeline import preprocess_data

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("Credit Card Fraud Detection")
st.write("Upload your transaction CSV file to generate fraud predictions.")

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    # Make sure this file exists in your Colab files!
    model.load_model("xgb_model.json")
    return model

# Load model (handle error if file missing)
try:
    model = load_model()
except:
    st.error("Model file 'xgb_model.json' not found. Please upload it to Colab files.")
    st.stop()

uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])

if uploaded_file is not None:
    st.info("Loading data...")
    df_raw = pd.read_csv(uploaded_file)

    with st.expander("Preview Raw Data"):
        st.dataframe(df_raw.head())

    st.info("Running feature engineering pipeline...")
    try:
        X_processed = preprocess_data(df_raw)

        # Align columns
        expected_features = model.get_booster().feature_names
        for col in expected_features:
            if col not in X_processed.columns:
                X_processed[col] = 0
        X_processed = X_processed[expected_features]

        probabilities = model.predict_proba(X_processed)[:, 1]
        predictions = (probabilities > 0.0028).astype(int)

        df_results = df_raw.copy()
        df_results['Fraud_Prediction'] = predictions
        df_results['Fraud_Probability'] = probabilities

        st.subheader("Prediction Results")
        fraud_cases = df_results[df_results['Fraud_Prediction'] == 1]
        st.warning(f"Detected {len(fraud_cases)} potential fraudulent transactions.")
        st.dataframe(df_results[['TransactionID', 'Fraud_Prediction', 'Fraud_Probability']])

    except Exception as e:
        st.error(f"Error: {e}")
