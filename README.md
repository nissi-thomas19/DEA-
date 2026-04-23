Smart Garbage Bin Monitoring System (IoT + ML + RL)
Overview

This project presents a Smart Garbage Bin Monitoring System designed to simulate real-world IoT-based waste management. It generates sensor data from 100 smart bins and uses Machine Learning and Reinforcement Learning techniques to monitor waste levels in real time and optimize garbage collection schedules.

The system is integrated into an interactive Streamlit dashboard, making it suitable for smart city applications where efficiency and automation are essential.

Key Features:
Simulates IoT sensor data for 100 garbage bins with over 50,000 records
Performs Exploratory Data Analysis (EDA) to understand fill-level patterns
Visualizes historical trends of bin usage
Classifies bin status (Empty, Half-Filled, Full) using a Random Forest model
Detects anomalies (e.g., faulty sensors) using Isolation Forest
Uses Q-Learning (Reinforcement Learning) for adaptive waste collection scheduling
Implements a rule-based alert system for warning and critical fill levels
Supports both single-bin and multi-bin simulation scenarios
Provides an interactive real-time dashboard using Streamlit and Plotly

Technologies Used:
Python
Pandas & NumPy
Matplotlib & Seaborn
Scikit-learn
XGBoost
Streamlit
Plotly
Joblib

Future Enhancements:
Integration with real-time IoT sensors using MQTT protocol
Route optimization using Vehicle Routing Problem (VRP) algorithms
Implementation of Deep Q-Networks (DQN) for advanced reinforcement learning
Fill-level forecasting using LSTM models
Development of a mobile application for field operators
Cloud deployment with a scalable database backend

How to Run the Project:
Generate the dataset: python generate_dataset.py
Train the machine learning models: python train_model.py
Train the reinforcement learning agent: python rl_model.py
Launch the dashboard: streamlit run app.py
