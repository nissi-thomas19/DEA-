import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import time
import random


st.set_page_config(page_title="Smart Garbage Monitoring", layout="wide")

st.title("Smart Garbage Bin Monitoring Dashboard")
st.markdown("IoT Based Waste Management - Advanced ML Monitoring")

# Load data and model
df = pd.read_csv("data/garbage_large_dataset.csv")
import os

MODEL_PATH = os.path.join("models", "garbage_model.pkl")
model, anomaly_model, encoder = joblib.load(MODEL_PATH)


# Sidebar controls
st.sidebar.header("Simulation Control")
bin_id = st.sidebar.selectbox("Select Bin ID", df["Bin_ID"].unique())

filtered_df = df[df["Bin_ID"] == bin_id]

# Latest Data
latest = filtered_df.iloc[-1]

fill = latest["Fill_Level"]
temp = latest["Temperature"]
humidity = latest["Humidity"]

st.subheader(f"Live Status - Bin {bin_id}")

col1, col2, col3 = st.columns(3)

col1.metric("Fill Level (%)", round(fill,2))
col2.metric("Temperature (°C)", round(temp,2))
col3.metric("Humidity (%)", round(humidity,2))

# ML Prediction
input_data = np.array([[fill, temp, humidity]])
prediction = model.predict(input_data)
predicted_status = encoder.inverse_transform(prediction)[0]

st.subheader("ML Prediction")
st.success(f"Predicted Status: {predicted_status}")

# Alert System
if fill > 80:
    st.error("CRITICAL ALERT: Bin Almost Full!")
elif fill > 60:
    st.warning("Warning: Bin Reaching Capacity")

# Anomaly Detection
anomaly = anomaly_model.predict(input_data)
if anomaly[0] == -1:
    st.error("Anomaly Detected! Possible Sensor Irregularity")

# Visualization
st.subheader("Historical Fill Level Trend")

fig = px.line(
    filtered_df.tail(200),
    x="Timestamp",
    y="Fill_Level",
    title="Last 200 Records Fill Level Trend"
)

st.plotly_chart(fig, use_container_width=True)
import os
print(os.getcwd())
print(os.listdir("models"))
import joblib
# ================= RL SIMULATION SECTION =================

st.markdown("---")
st.subheader("Smart City Reinforcement Learning Simulation")

Q = joblib.load("models/rl_q_table.pkl")

def get_state(fill):
    if fill < 30:
        return 0
    elif fill < 75:
        return 1
    else:
        return 2


# -------- SINGLE BIN SIMULATION --------

st.subheader("Single Bin Simulation")

if "current_fill" not in st.session_state:
    st.session_state.current_fill = random.randint(10, 40)

if st.button("Run Single Bin Step"):

    st.session_state.current_fill += random.randint(5, 15)

    if st.session_state.current_fill > 100:
        st.session_state.current_fill = 100

    current_fill = st.session_state.current_fill

    st.metric("Simulated Fill Level (%)", current_fill)

    state = get_state(current_fill)
    action = np.argmax(Q[state])

    if action == 0:
        st.info("RL Decision: No Action Required")
    elif action == 1:
        st.warning("RL Decision: Schedule Collection")
    else:
        st.error("RL Decision: Priority Collection Immediately!")
        st.session_state.current_fill = 0
        st.success("Garbage Collected! Bin Reset to 0%")


# -------- MULTI BIN SIMULATION --------

st.markdown("---")
st.subheader("Multi-Bin Smart City Simulation")

NUM_BINS = 5

if "city_bins" not in st.session_state:
    st.session_state.city_bins = {
        f"City_Bin_{i}": random.randint(10, 40)
        for i in range(1, NUM_BINS + 1)
    }

if st.button("Run Multi-Bin Step"):

    updated_bins = {}
    bin_output = []

    for bin_name, fill_value in st.session_state.city_bins.items():

        new_fill = fill_value + random.randint(3, 12)

        if new_fill > 100:
            new_fill = 100

        state = get_state(new_fill)
        action = np.argmax(Q[state])

        if action == 0:
            decision = "No Action"
        elif action == 1:
            decision = "Schedule Collection"
        else:
            decision = "Priority Collection"
            new_fill = 0

        updated_bins[bin_name] = new_fill

        bin_output.append({
            "Bin": bin_name,
            "Fill Level (%)": new_fill,
            "State": state,
            "RL Decision": decision
        })

    st.session_state.city_bins = updated_bins

    multi_df = pd.DataFrame(bin_output)

    st.dataframe(multi_df, use_container_width=True)




