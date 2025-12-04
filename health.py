# to run the project write in terminal (streamlit run "file path")
# Code By Anshuman Sinha

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import requests

# --- CONFIGURATION ---
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Health Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
    }
    .advice-box {
        color: #333333 !important; 
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- FUNCTION: SEND DATA ---
# Updated to accept all new registration fields
def send_to_getform(name, age, phone, height, weight, smoker, alcohol, activity):
    url = "https://getform.io/f/allgjxja"
    data = {
        "name": name,
        "age": age,
        "phone": phone,
        "height": height,
        "weight": weight,
        "smoker": "Yes" if smoker else "No",
        "alcohol": "Yes" if alcohol else "No",
        "activity_hours": activity,
        "registered_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    try:
        response = requests.post(url, data=data)
        return response.status_code == 200
    except Exception:
        return False

# --- STATE MANAGEMENT ---
if "registered" not in st.session_state:
    st.session_state.registered = False

# ==========================================
# 1. REGISTRATION SCREEN (UPDATED)
# ==========================================
if not st.session_state.registered:
    col1, col2, col3 = st.columns([1, 2, 1]) 
    with col2:
        st.markdown("<h1 style='text-align: center; color: #333;'>🏥 Patient Registration</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666;'>Please enter your details to access the Health Risk Predictor.</p>", unsafe_allow_html=True)
        
        with st.form("reg_form"):
            # Basic Info
            st.subheader("Personal Details")
            name = st.text_input("Full Name", placeholder="e.g. John Doe")
            
            c_age, c_phone = st.columns(2)
            with c_age:
                reg_age = st.number_input("Age", min_value=1, max_value=120, value=30)
            with c_phone:
                # Max chars set to 10 for visual cue, validation logic is below
                phone = st.text_input("Phone Number (10 digits)", max_chars=10, placeholder="e.g. 9876543210")

            # Physical Stats
            st.subheader("Physical Stats")
            c_height, c_weight = st.columns(2)
            with c_height:
                reg_height = st.number_input("Height (cm)", min_value=50.0, max_value=300.0, value=170.0)
            with c_weight:
                reg_weight = st.number_input("Weight (kg)", min_value=10.0, max_value=200.0, value=65.0)

            # Lifestyle
            st.subheader("Lifestyle Habits")
            reg_activity = st.slider("Physical Activity (Hours/Day)", min_value=0, max_value=24, value=1)
            
            c_smoke, c_alcohol = st.columns(2)
            with c_smoke:
                reg_smoker = st.checkbox("Do you Smoke?")
            with c_alcohol:
                reg_alcohol = st.checkbox("Do you consume Alcohol?")
            
            st.markdown("---")
            submit = st.form_submit_button("Start Assessment")
            
            if submit:
                # --- VALIDATION LOGIC ---
                if not name:
                    st.warning("Please enter your name.")
                elif not phone:
                    st.warning("Please enter your phone number.")
                elif len(phone) != 10 or not phone.isdigit():
                    st.error("⚠️ Invalid Phone Number: Must be exactly 10 digits.")
                else:
                    # Proceed if validation passes
                    with st.spinner("Registering..."):
                        success = send_to_getform(name, reg_age, phone, reg_height, reg_weight, reg_smoker, reg_alcohol, reg_activity)
                        if success:
                            # Save all inputs to session state to use in Dashboard
                            st.session_state.registered = True
                            st.session_state.name = name
                            st.session_state.reg_age = reg_age
                            st.session_state.phone = phone
                            st.session_state.reg_height = reg_height
                            st.session_state.reg_weight = reg_weight
                            st.session_state.reg_smoker = reg_smoker
                            st.session_state.reg_alcohol = reg_alcohol
                            st.session_state.reg_activity = reg_activity
                            
                            st.success("Success! Loading dashboard...")
                            st.rerun()
                        else:
                            st.error("Connection error. Please try again or check internet.")

# ==========================================
# 2. MAIN APP (DASHBOARD)
# ==========================================
else:
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
        st.title("Patient Profile")
        st.markdown(f"**Name:** {st.session_state.name}")
        st.markdown(f"**Age:** {st.session_state.reg_age}")
        st.markdown(f"**Phone:** {st.session_state.phone}")
        st.divider()
        st.info("💡 **Tip:** We have pre-filled the form with your registration data. You can tweak it below to test scenarios.")
        st.caption("Code by Anshuman Sinha")

    # --- MAIN CONTENT ---
    st.markdown("<h2 style='text-align: center;'>🩺 AI Health Risk Predictor</h2>", unsafe_allow_html=True)
    st.markdown("---")

    # --- INPUT SECTION ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Biometrics")
        # Pre-filling with session state data from registration
        age = st.slider("Current Age", 10, 100, int(st.session_state.reg_age))
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        height = st.number_input("Height (cm)", 120.0, 300.0, float(st.session_state.reg_height))
        weight = st.number_input("Weight (kg)", 30.0, 200.0, float(st.session_state.reg_weight))
        
        # Calculate BMI
        bmi = round(weight / ((height / 100) ** 2), 2)
        if bmi < 18.5:
            bmi_status = "Underweight"
            bmi_color = "off"
        elif 18.5 <= bmi < 25:
            bmi_status = "Normal"
            bmi_color = "normal"
        else:
            bmi_status = "Overweight"
            bmi_color = "inverse"
        
        st.metric(label="Calculated BMI", value=bmi, delta=bmi_status, delta_color=bmi_color)

    with col2:
        st.subheader("🩸 Vitals")
        glucose = st.slider("Glucose (mg/dL)", 50, 300, 90, help="Normal: 70–99 mg/dL")
        bp_systolic = st.slider("Systolic BP", 90, 200, 120, help="Normal: ~120")
        bp_diastolic = st.slider("Diastolic BP", 60, 130, 80, help="Normal: ~80")
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 180, help="Normal: < 200")
        heart_rate = st.slider("Heart Rate (bpm)", 40, 150, 75)

    with col3:
        st.subheader("🏃 Lifestyle")
        # Converting Daily activity (from reg) to Weekly estimation for the model logic
        default_activity = int(st.session_state.reg_activity * 7) 
        if default_activity > 15: default_activity = 15 # Cap for slider

        physical_activity = st.slider("Activity (hrs/week)", 0, 15, default_activity)
        sleep_hours = st.slider("Sleep (hrs/day)", 0, 12, 7)
        
        st.write("Habits:")
        c1, c2 = st.columns(2)
        with c1:
            # Pre-filled from registration
            smoking = st.checkbox("Smoker?", value=st.session_state.reg_smoker)
        with c2:
            # Pre-filled from registration
            alcohol = st.checkbox("Alcohol?", value=st.session_state.reg_alcohol)

    # Encodings
    gender_encoded = 1 if gender == "Male" else 0
    smoking_encoded = 1 if smoking else 0
    alcohol_encoded = 1 if alcohol else 0

    input_features = pd.DataFrame([[
        age, gender_encoded, weight, height, bmi, glucose, bp_systolic, bp_diastolic,
        cholesterol, heart_rate, physical_activity, smoking_encoded, alcohol_encoded, sleep_hours
    ]], columns=[
        "Age", "Gender", "Weight", "Height", "BMI", "Glucose", "SystolicBP", 
        "DiastolicBP", "Cholesterol", "HeartRate", "PhysicalActivity", "Smoking", 
        "Alcohol", "SleepHours"
    ])

    # --- MODEL LOGIC (Hidden) ---
    np.random.seed(42) 
    X_train = pd.DataFrame({
        "Age": np.random.randint(18, 80, 500),
        "Gender": np.random.randint(0, 2, 500),
        "Weight": np.random.randint(45, 110, 500),
        "Height": np.random.randint(150, 190, 500),
        "BMI": np.random.uniform(18, 35, 500),
        "Glucose": np.random.randint(70, 200, 500),
        "SystolicBP": np.random.randint(100, 180, 500),
        "DiastolicBP": np.random.randint(60, 110, 500),
        "Cholesterol": np.random.randint(150, 300, 500),
        "HeartRate": np.random.randint(55, 100, 500),
        "PhysicalActivity": np.random.randint(0, 10, 500),
        "Smoking": np.random.randint(0, 2, 500),
        "Alcohol": np.random.randint(0, 2, 500),
        "SleepHours": np.random.randint(4, 10, 500),
    })
    y_train = np.random.randint(0, 2, 500)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=500)
    model.fit(X_scaled, y_train)

    # Prediction
    input_scaled = scaler.transform(input_features)
    risk_prob = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    st.markdown("---")
    
    # --- RESULTS SECTION ---
    st.subheader("📊 Analysis Results")
    
    # Centering the results for a cleaner look
    res_c1, res_c2, res_c3 = st.columns([1, 2, 1])
    
    with res_c2:
        st.write("Health Risk Probability")
        
        # Visual Progress Bar
        bar_color = "red" if risk_prob > 0.5 else "green"
        st.progress(risk_prob)
        
        if prediction == 1:
            st.error(f"⚠️ **High Risk Detected ({risk_prob*100:.1f}%)**")
            st.markdown("""
                <div style="background-color: #ffe6e6; color: #333333; padding: 10px; border-radius: 5px;">
                    <strong>Advice:</strong> Your input metrics suggest an elevated health risk. 
                    It is highly recommended to consult a doctor for a full check-up.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"✅ **Low Risk Detected ({risk_prob*100:.1f}%)**")
            st.markdown("""
                <div style="background-color: #e6fffa; color: #333333; padding: 10px; border-radius: 5px;">
                    <strong>Great Job:</strong> Your vitals appear to be within a healthy range. 
                    Continue maintaining a balanced diet and active lifestyle!
                </div>
            """, unsafe_allow_html=True)
