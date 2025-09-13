import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Cluster labels
cluster_labels = {
    0: "Premium Seniors – Older (~70), wealthy, high spenders, prefer in-store",
    1: "Browsing Budgeters – Middle-aged, low income, frequent browsers, minimal spend",
    2: "Omnichannel Actives – Late middle-aged, mid-high income, frequent online & store buyers",
    3: "Dormant Low Spenders – Middle-aged, low spenders, long time since last purchase",
    4: "Loyal In-Store Buyers – Older, wealthy, heavy in-store shoppers, recent buyers",
    5: "Young Big Spenders – Younger, very wealthy, highest spenders, prefer in-store",
}

# Page setup
st.set_page_config(
    page_title="Customer Segmentation", page_icon="🛒", layout="centered"
)

# Custom CSS for button and result text
st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #FAFAFA;
        }

        .stButton > button {
            background-color: #28a745;
            color: white;
            font-size: 22px;
            font-weight: bold;
            padding: 16px 36px;
            border-radius: 8px;
            border: none;
            transition: background-color 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #218838;
        }

        .result-text {
            font-size: 28px;
            font-weight: 1000;
            margin-top: 25px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown("## 👥 Customer Segmentation Predictor")

st.write(
    "Enter customer details below to predict their segment using K-Means clustering."
)

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
    total_spending = st.number_input(
        "Total Spending", min_value=0, max_value=5000, value=1000
    )

    # Recency + Predict button in same row
    col_recency, col_predict = st.columns([2, 1.5])
    with col_recency:
        recency = st.number_input(
            "Recency (days since last purchase)", min_value=0, max_value=365, value=30
        )
    with col_predict:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("Predict")

with col2:
    num_web_purchases = st.number_input(
        "Number of Web Purchases", min_value=0, max_value=100, value=10
    )
    num_store_purchases = st.number_input(
        "Number of Store Purchases", min_value=0, max_value=100, value=10
    )
    num_web_visits = st.number_input(
        "Number of Web Visits/Month", min_value=0, max_value=50, value=3
    )

# Prediction
if predict_clicked:
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Income": [income],
            "Total_Spending": [total_spending],
            "NumWebPurchases": [num_web_purchases],
            "NumStorePurchases": [num_store_purchases],
            "NumWebVisitsMonth": [num_web_visits],
            "Recency": [recency],
        }
    )

    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]
    label = cluster_labels.get(cluster, "Unknown Segment")

    st.markdown(
        f"<div class='result-text'><strong>Predicted Segment (Cluster {cluster}):</strong> {label}</div>",
        unsafe_allow_html=True,
    )
