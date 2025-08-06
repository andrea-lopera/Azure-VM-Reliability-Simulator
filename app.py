# app.py
import streamlit as st
import joblib
import pickle
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from xgboost import XGBClassifier

# Load pre-trained artifacts
model = joblib.load('model.pkl')
ctv = CoxTimeVaryingFitter()
ctv = joblib.load('survival_model.pkl')
try:
    explainer = joblib.load('shap_explainer.joblib')
except:
    with open('shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f) 
df = pd.read_csv('vm_telemetry.csv')
failure_history = np.load('failure_history.npy', allow_pickle=True).item()
with open('feature_names.txt', 'r') as f:
    feature_names = f.read().split(',')
    print(feature_names) 

# 7. Validate Loaded Artifacts
assert model is not None, "Model failed to load"
assert ctv.params_ is not None, "Survival model failed to load"
assert explainer is not None, "SHAP explainer failed to load"
assert not df.empty, "Dataset failed to load"
assert failure_history, "Failure history failed to load"

# Dashboard

st.title("Azure VM Reliability Simulator")

# Real-time gauge
st.subheader("Failure Probability Gauge")
vm_id = st.selectbox("Select VM", df['vm_id'].unique())
vm_data = df[df['vm_id'] == vm_id].iloc[-1]

# Prepare features (ensure same order as training)
# features = ['cpu_util', 'mem_util', 'disk_io', 'net_latency', 'is_peak', 'cpu_mem_ratio', 'sys_failures']
vm_features = vm_data[feature_names].values.reshape(1, -1)

prob = model.predict_proba(vm_features)[0][1]
st.plotly_chart(px.bar(x=[prob], range_x=[0,1], title=f"Failure Probability: {prob:.2%}"))

# Survival curve sliders
st.subheader("Survival Curve Explorer")
cpu = st.slider("CPU Utilization", 0, 100, int(vm_data['cpu_util']))
disk = st.slider("Disk I/O", 0, 500, int(vm_data['disk_io']))

# Calculate survival curve
baseline = ctv.baseline_survival_  # Fixed typo (cv -> ctv)
adjusted = baseline ** np.exp(
    ctv.params_.get('cpu_util', 0) * cpu + 
    ctv.params_.get('disk_io', 0) * disk
)

# Create a DataFrame for Plotly
surv_df = pd.DataFrame({
    'Days': adjusted.index, 
    'Survival Probability' : adjusted.values.flatten()
})

# Create Plotly survival curve
surv_fig = px.line(
    surv_df,
    x='Days', 
    y='Survival Probability',
    title="VM Survival Probability Over Time"
)
surv_fig.update_layout(yaxis_range=[0,1])
st.plotly_chart(surv_fig)

# SHAP waterfall
st.subheader("Failure Explanation")
shap_value = explainer(vm_features)

# Create waterfall plot
plt.figure(figsize=(10, 4))
shap.plots.waterfall(shap_value[0], show=False)
st.pyplot(plt.gcf())  # Use current figure

# Feature importance beeswarm
st.subheader("Global Feature Impact")
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_value, show=False)
st.pyplot(plt.gcf())

# LLM Scenarios
st.subheader("Generated Failure Scenarios")
with st.expander("View LLM-generated Scenarios"):
    # Extract scenarios for this VM
    vm_scenarios = [s[1] for s in failure_history.get(vm_id, [])]
    
    if vm_scenarios:
        for scenario in set(vm_scenarios):
            st.markdown(f"- {scenario}")
    else:
        st.info("No failure scenarios recorded for this VM")
        
# Raw telemetry data
st.subheader("Recent Telemetry")
st.dataframe(df[df['vm_id'] == vm_id].tail(10)[['cpu_util', 'mem_util', 'disk_io', 'net_latency', 'is_peak', 'cpu_mem_ratio', 'sys_failures']])

