
"""
AI-Based Industrial Machine Health & Failure Prediction
23CSE301 Machine Learning Capstone Project (AY 2026-27)
Interactive Predictive Maintenance Web Application
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Industrial Machine Health & Failure Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .alert-danger {
        background-color: #fef2f2;
        color: #991b1b;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #ef4444;
    }
    .alert-success {
        background-color: #f0fdf4;
        color: #166534;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #22c55e;
    }
</style>
""", unsafe_allow_html=True)

# Path resolution for serialized model artifact
APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "best_classifier.pkl")

@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model artifact not found at {MODEL_PATH}. Please ensure model training has been completed.")
        return None
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model artifact: {e}")
        return None

pipeline = load_pipeline()

# Header
st.markdown('<div class="main-header">⚙️ Industrial Machine Failure Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">23CSE301 Machine Learning Capstone · Real-Time Telemetry Diagnostic Dashboard</div>', unsafe_allow_html=True)

# Sidebar: Controls & Presets
st.sidebar.header("🛠️ Diagnostic Parameters")
st.sidebar.markdown("Configure operational machine telemetry or select a test scenario:")

preset = st.sidebar.selectbox(
    "Load Scenario Preset",
    ["Custom Input", "Nominal Operating State (Normal)", "High Thermal Dissipation Strain", "Overstrain & High Torque", "Severe Tool Wear Failure"]
)

# Set defaults based on preset
if preset == "Nominal Operating State (Normal)":
    def_type = "M"
    def_air = 298.1
    def_proc = 308.6
    def_speed = 1551
    def_torque = 42.8
    def_wear = 20
elif preset == "High Thermal Dissipation Strain":
    def_type = "L"
    def_air = 303.5
    def_proc = 313.2
    def_speed = 1380
    def_torque = 58.5
    def_wear = 110
elif preset == "Overstrain & High Torque":
    def_type = "L"
    def_air = 300.2
    def_proc = 310.4
    def_speed = 1250
    def_torque = 68.2
    def_wear = 215
elif preset == "Severe Tool Wear Failure":
    def_type = "H"
    def_air = 299.0
    def_proc = 309.1
    def_speed = 1420
    def_torque = 52.0
    def_wear = 245
else:
    def_type = "M"
    def_air = 300.0
    def_proc = 310.0
    def_speed = 1500
    def_torque = 40.0
    def_wear = 50

type_input = st.sidebar.selectbox("Product Quality Tier (Type)", ["L", "M", "H"], index=["L", "M", "H"].index(def_type))
air_temp = st.sidebar.slider("Air Temperature [K]", 290.0, 310.0, float(def_air), 0.1)
proc_temp = st.sidebar.slider("Process Temperature [K]", 300.0, 320.0, float(def_proc), 0.1)
rot_speed = st.sidebar.slider("Rotational Speed [rpm]", 1100, 2900, int(def_speed), 10)
torque = st.sidebar.slider("Torque [Nm]", 3.0, 80.0, float(def_torque), 0.5)
tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 300, int(def_wear), 1)

# Feature Engineering
temp_diff = proc_temp - air_temp
power_kw = (torque * rot_speed * 2 * np.pi) / 60000.0
mech_stress = torque / (rot_speed + 1e-5)

# Main Dashboard Layout
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Spindle Power", f"{power_kw:.2f} kW")
with col2:
    st.metric("Temperature Gradient", f"{temp_diff:.2f} K")
with col3:
    st.metric("Mechanical Stress Ratio", f"{mech_stress:.5f}")

st.markdown("---")

# Inference Input Vector
input_dict = {
    'Type': [type_input],
    'Air temperature [K]': [air_temp],
    'Process temperature [K]': [proc_temp],
    'Rotational speed [rpm]': [rot_speed],
    'Torque [Nm]': [torque],
    'Tool wear [min]': [tool_wear],
    'Temp_Difference': [temp_diff],
    'Power_kW': [power_kw],
    'Mechanical_Stress': [mech_stress]
}

input_df = pd.DataFrame(input_dict)

st.subheader("📊 Diagnostic Prediction & Failure Risk Assessment")

if pipeline is not None:
    try:
        prediction = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0]
        fail_prob = prob[1] * 100.0
        norm_prob = prob[0] * 100.0

        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            if prediction == 1 or fail_prob >= 40.0:
                st.markdown(f"""
                <div class="alert-danger">
                    <h3>⚠️ HIGH FAILURE RISK DETECTED</h3>
                    <p><b>Predicted Status:</b> Machine Failure Impending</p>
                    <p><b>Failure Probability:</b> {fail_prob:.2f}%</p>
                    <p><b>Action Required:</b> Trigger scheduled maintenance immediately; inspect spindle lubrication and tool cutting edge.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-success">
                    <h3>✅ NORMAL OPERATING REGIME</h3>
                    <p><b>Predicted Status:</b> Machine Operating Safely</p>
                    <p><b>Normal Confidence:</b> {norm_prob:.2f}% (Failure Probability: {fail_prob:.2f}%)</p>
                    <p><b>Action Required:</b> Continue standard operating telemetry logging.</p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.write("#### Failure Risk Probability Gauge")
            st.progress(min(int(fail_prob), 100))
            st.caption(f"Estimated probability of machine failure: **{fail_prob:.2f}%**")

            st.write("#### Telemetry Feature Vector Summary")
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Inference computation error: {e}")
else:
    st.warning("Model pipeline currently unavailable.")

st.markdown("---")
st.markdown("### 📘 Model & Pipeline Context")
st.markdown("""
- **Classifier Architecture:** Gradient Boosting Classifier with GridSearchCV Hyperparameter Optimization
- **Preprocessing:** Scikit-Learn `ColumnTransformer` with `StandardScaler` (numerical) and `OneHotEncoder` (categorical `Type`)
- **Leakage Prevention:** Diagnostic flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) strictly excluded from model features
- **Benchmark F1-Score:** 0.9850 on held-out test split
""")
