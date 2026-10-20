# Interactive Predictive Maintenance Web Application
## 23CSE301 Machine Learning Capstone (Review 2 Bonus Deliverable)

### Overview
This web application provides a real-time predictive maintenance dashboard built with **Streamlit**. It loads the serialized pipeline (`models/best_classifier.pkl`) trained on the AI4I 2020 Predictive Maintenance Dataset and allows plant operators to assess machine failure risk interactively.

---

### Features
1. **Interactive Sliders:** Adjust temperatures, spindle speed, torque, and tool wear in real-time.
2. **Preset Scenarios:** Pre-loaded operating states (Nominal Normal, Thermal Strain, Overstrain/High Torque, Severe Tool Wear).
3. **Automated Feature Engineering:** Computes instantaneous spindle power ($kW$), temperature difference ($\\Delta T$), and mechanical stress ratio.
4. **Risk Gauge & Action Guidance:** Visual progress bar and clear operational maintenance advice based on predicted failure probability.

---

### How to Run Locally

```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Launch the Streamlit application from the repository root
streamlit run app/app.py
```

The application will launch automatically in your default browser at `http://localhost:8501`.
