import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stAlert {padding: 1rem; border-radius: 0.5rem;}
    h1 {color: #1f77b4; padding-bottom: 1rem;}
    </style>
    """, unsafe_allow_html=True)


# Load
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('venv\diabetes_model.pkl')
        scaler = joblib.load('venv\scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

st.title("🏥 Diabetes Prediction System")
st.markdown("### AI-Powered Risk Assessment Tool")

model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("❌ **Model files not found!**")
    st.info("""
    Please run the following command first:
    ```
    python diabetes_prediction.py
    ```
    This will train and save the model files.
    """)
    st.stop()

# Sidebar
st.sidebar.title("⚙️ Patient Information")

st.sidebar.subheader("Demographics")
age = st.sidebar.slider('Age', 11, 100, 30)
gender = st.sidebar.radio(
    "Gender",
    ("Male", "Female")
)
gender_val = 1 if gender == "Male" else 0


st.sidebar.subheader("Medical Measurements")

pulse = st.sidebar.slider('Pulse Rate (BPM)', 50, 110, 70)
s_bp = st.sidebar.slider('Systolic Blood Pressure (mm Hg)', 0, 200, 100)
d_bp = st.sidebar.slider('Diastolic Blood Pressure (mm Hg)', 0, 130, 80)
glucose = st.sidebar.number_input('Glucose (mmol/L)', 0.00, 30.00, 10.00, 0.01)
height = st.sidebar.number_input('Height (m)', 1.00, 2.00, 1.50, 0.1)
weight = st.sidebar.number_input('Weight (kg)', 20.0, 120.0, 50.0, 0.1)
bmi = st.sidebar.number_input('BMI', 10.0, 70.0, 25.0, 0.1)
family_diabaties = st.sidebar.radio(
    "Family Diabaties",
    ("Yes", "No")
)
family_diabaties_val = 1 if family_diabaties == "Yes" else 0
hypertensive = st.sidebar.radio(
    "Hypertensive",
    ("Yes", "No")
)
hypertensive_val = 1 if hypertensive == "Yes" else 0
family_hypertension = st.sidebar.radio(
    "Family Hypertension",
    ("Yes", "No")
)
family_hypertension_val = 1 if family_hypertension == "Yes" else 0
cardiovascular_disease = st.sidebar.radio(
    "Cardiovascular Disease",
    ("Yes", "No")
)
cardiovascular_disease_val = 1 if cardiovascular_disease == "Yes" else 0
stroke = st.sidebar.radio(
    "Stroke",
    ("Yes", "No")
)
stroke_val = 1 if stroke == "Yes" else 0


# btn
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Predict", type="primary", use_container_width=True)

# Main
if predict_btn:
    # input
    input_data = np.array([[age, gender_val, pulse, s_bp, d_bp, glucose, height, weight, bmi, family_diabaties_val, hypertensive_val, family_hypertension_val, cardiovascular_disease_val, stroke_val]])
    
    # Standardize
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]

    try:
        probability = model.predict_proba(input_scaled)[0]
        prob_negative = probability[0] * 100
        prob_positive = probability[1] * 100
    except:
        prob_positive = 100 if prediction == 1 else 0
        prob_negative = 100 - prob_positive

    st.markdown("---")
    st.header("🎯 Prediction Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:

        if prediction == 0:
            if prob_positive < 30:
                st.success("### ✅ LOW RISK - Not Diabetic")
            else:
                st.warning("### ⚠️ MODERATE RISK - Not Diabetic")
        else:
            if prob_positive > 70:
                st.error("### 🔴 HIGH RISK - Diabetic")
            else:
                st.warning("### ⚠️ MODERATE RISK - Diabetic")
           
        st.subheader("Probability Breakdown")
        pcol1, pcol2 = st.columns(2)
        pcol1.metric("Non-Diabetic", f"{prob_negative:.1f}%")
        pcol2.metric("Diabetic", f"{prob_positive:.1f}%")
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_positive,
            title={'text': "Risk Level"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
       
    st.markdown("---")
    st.subheader("⚠️ Risk Factor Analysis")

    risk_factors = []
    positive_factors = []
    
    if glucose > 10.00:
        risk_factors.append("🔴 High Glucose Level (>10 mmol/L)")
    elif glucose < 5.00:
        positive_factors.append("🟢 Normal Glucose Level")
    
    if bmi > 30:
        risk_factors.append("🔴 High BMI - Obesity (>30)")
    elif 18.5 <= bmi <= 24.9:
        positive_factors.append("🟢 Healthy BMI (18.5-24.9)")
    
    if age > 45:
        risk_factors.append("🟡 Age Factor (>45)")
    
    if d_bp > 80:
        risk_factors.append("🔴 High Diastolic Blood Pressure (>80 mm Hg)")
    elif 60 <= d_bp <= 80:
        positive_factors.append("🟢 Normal Diastolic Blood Pressure")

    if s_bp > 120:
        risk_factors.append("🔴 High Systolic Blood Pressure (>120 mm Hg)")
    elif 90 <= s_bp <= 120:
        positive_factors.append("🟢 Normal Systolic Blood Pressure")
    
    
    if risk_factors:
        st.warning("**Identified Risk Factors:**")
        for factor in risk_factors:
            st.markdown(f"- {factor}")
    
    if positive_factors:
        st.success("**Positive Health Indicators:**")
        for factor in positive_factors:
            st.markdown(f"- {factor}")
    
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    if prediction == 1:
        st.error("""
        **Important Actions:**
        - Consult a healthcare professional immediately
        - Get comprehensive diabetes screening
        - Monitor blood glucose regularly
        - Consider lifestyle modifications
        """)
    else:
        st.success("""
        **Maintain Healthy Practices:**
        - Regular health check-ups
        - Balanced diet
        - Exercise regularly (30+ min daily)
        - Monitor weight and BMI
        """)
    
    st.markdown("---")
    st.warning("""
    **⚠️ MEDICAL DISCLAIMER**
    
    This prediction is for educational purposes only. It should NOT replace 
    professional medical advice. Always consult qualified healthcare professi8
    """)
    
else:
    
    st.markdown("---")
    st.info("👈 Enter patient information in the sidebar and click **Predict**")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", "Decision Tree")
    col2.metric("Accuracy", "~93%")
    col3.metric("Dataset", "5288 samples")
    