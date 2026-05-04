import streamlit as st
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import time
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="Live IDS Monitor", layout="wide")
st.title(" Live Zero-Day Intrusion Detection System")
st.markdown("Monitoring network traffic in **10-Second Micro-Batches** via LSTM-Autoencoder.")

# 2. Load the AI Brain and Data
@st.cache_resource
def load_ai_engine():
    model = tf.keras.models.load_model('zero_day_ids_model.h5',compile=False)
    live_data_stream = np.load('X_test_tensor.npy')
    return model, live_data_stream
try:
    model, live_data_stream = load_ai_engine()
except OSError:
    st.error(" Could not find 'zero_day_ids_model.h5' or 'X_test_tensor.npy'.")
    st.stop()

# 3. Sidebar Controls
st.sidebar.header("Command Center")
threshold = st.sidebar.slider("Strictness Threshold (MSE)", min_value=0.01, max_value=0.50, value=0.05, step=0.01)
batch_size = st.sidebar.slider("Packets per 10-Second Batch", min_value=100, max_value=5000, value=1000, step=100)
start_button = st.sidebar.button("▶️ START 10s BATCH MONITORING", use_container_width=True)

# 4. UI Placeholders
col1, col2, col3 = st.columns(3)
metric_scanned = col1.empty()
metric_attacks = col2.empty()
metric_status = col3.empty()

st.markdown("---")
st.subheader("Live Latent Space Reconstruction Error")
chart_placeholder = st.empty()

st.markdown("---")
st.subheader(" Live Threat Intelligence Log")
# This creates an empty frame for our live updating table
table_placeholder = st.empty() 

# 5. The 10-Second Micro-Batch Loop
if start_button:
    simulated_time = pd.Timestamp.now()
    
    history_timestamps = []
    history_mse = []
    history_flags = []
    threat_logs = [] 
    
    total_attacks = 0
    total_scanned = 0
    
    for i in range(0, len(live_data_stream), batch_size):
        current_batch = live_data_stream[i : i + batch_size]
        
        if len(current_batch) == 0:
            break
            
        reconstructed_batch = model.predict(current_batch, verbose=0)
        batch_mse = np.mean(np.power(current_batch - reconstructed_batch, 2), axis=(1, 2))
        
        batch_anomalies = batch_mse > threshold
        new_attacks_in_batch = np.sum(batch_anomalies)
        
        total_attacks += new_attacks_in_batch
        total_scanned += len(current_batch)
        start_time = simulated_time
        simulated_time = simulated_time + pd.Timedelta(seconds=10)
        time_steps = pd.date_range(start=start_time, end=simulated_time, periods=len(batch_mse))
        if new_attacks_in_batch > 0:
            attack_indices = np.where(batch_anomalies)[0]
            for idx in attack_indices:
                packet_errors = np.abs(current_batch[idx] - reconstructed_batch[idx])
                feature_errors = np.mean(packet_errors, axis=0)
                worst_feature_idx = np.argmax(feature_errors)
                
                threat_logs.append({
                    "Timestamp": time_steps[idx].strftime("%H:%M:%S.%f")[:-3],
                    "MSE Score": round(batch_mse[idx], 4),
                    "Root Cause (Feature Index)": f"Feature #{worst_feature_idx}"
                })
        
        history_timestamps.extend(time_steps)
        history_mse.extend(batch_mse)
        history_flags.extend(batch_anomalies)
        metric_scanned.metric("Total Sequences Scanned", f"{total_scanned:,}")
        metric_attacks.metric("Threats Detected", f"{total_attacks:,}")
        
        if new_attacks_in_batch > 0:
            metric_status.markdown("### Status: <span style='color:red'> BREACH DETECTED</span>", unsafe_allow_html=True)
        else:
            metric_status.markdown("### Status: <span style='color:green'> SECURE</span>", unsafe_allow_html=True)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_timestamps, y=history_mse, mode='lines', line=dict(color='#1E90FF', width=2), name='Network MSE'))
        fig.add_hline(y=threshold, line_dash="dash", line_color="orange", annotation_text="Threshold")
        
        attack_times = [history_timestamps[j] for j in range(len(history_flags)) if history_flags[j]]
        attack_mses = [history_mse[j] for j in range(len(history_flags)) if history_flags[j]]
        
        if attack_times:
            fig.add_trace(go.Scatter(
                x=attack_times, y=attack_mses, 
                mode='markers', 
                marker=dict(color='red', size=12, symbol='x'), 
                name='Attack',
                hovertemplate='<b> ATTACK DETECTED</b><br>Time: %{x}<br>MSE: %{y}<extra></extra>'
            ))
            
        fig.update_layout(
            height=450, 
            xaxis_title="Live System Time", 
            yaxis_title="MSE Score", 
            margin=dict(l=0, r=0, t=30, b=0), 
            yaxis_range=[0, max(0.2, max(history_mse) * 1.2)],
            hovermode="x unified"
        )
        latest_time = history_timestamps[-1]
        window_start = latest_time - pd.Timedelta(seconds=60)
        fig.update_xaxes(range=[window_start, latest_time], rangeslider_visible=True)
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        if len(threat_logs) > 0:
            log_df = pd.DataFrame(threat_logs)
            table_placeholder.dataframe(log_df[::-1], use_container_width=True)
            
        time.sleep(2)
        
    st.success(" SIMULATION COMPLETE: All network packets have been successfully scanned and logged.")