"""
Real-Time Fraud Detection System - Interactive Dashboard
Streamlit Web Application
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json
import os

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .fraud-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .normal-transaction {
        background-color: #00cc00;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []
if 'fraud_detected' not in st.session_state:
    st.session_state.fraud_detected = 0
if 'total_processed' not in st.session_state:
    st.session_state.total_processed = 0

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def predict_fraud(transaction):
    """Send transaction to API for prediction"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=transaction,
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

def load_sample_data():
    """Load sample transactions from CSV"""
    try:
        # Try multiple paths (local and Docker)
        paths = [
            'data/creditcard.csv',           # Local
            './data/creditcard.csv',         # Docker
            '../data/creditcard.csv'         # Alternative
        ]
        
        for path in paths:
            try:
                df = pd.read_csv(path)
                return df
            except FileNotFoundError:
                continue
        
        # If no file found
        st.error("Could not load creditcard.csv")
        return None
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_transaction_from_row(row, idx):
    """Create transaction object from dataframe row"""
    features = {}
    for i in range(1, 29):
        features[f'V{i}'] = float(row[f'V{i}'])
    
    features['Amount'] = float(row['Amount'])
    features['Hour'] = float((row['Time'] / 3600) % 24)
    features['Log_Amount'] = float(np.log1p(row['Amount']))
    
    return {
        'transaction_id': f'tx_{idx}',
        'timestamp': datetime.now().isoformat(),
        'features': features,
        'amount': float(row['Amount']),
        'actual_label': int(row['Class'])
    }

# Header
st.title("🚨 Real-Time Fraud Detection System")
st.markdown("### AI-Powered Transaction Monitoring with PyTorch & Kafka")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # API Status
    st.subheader("🔌 API Status")
    api_healthy = check_api_health()
    
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.info("Start API: `cd src/api && python model_server.py`")
    
    st.markdown("---")
    
    # Mode selection
    st.subheader("🎯 Mode")
    mode = st.radio(
        "Choose operation mode:",
        ["Single Transaction", "Batch Processing", "Live Stream Demo"]
    )
    
    st.markdown("---")
    
    # Threshold adjustment
    st.subheader("🎚️ Detection Sensitivity")
    threshold_multiplier = st.slider(
        "Threshold Multiplier",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Higher = fewer false positives, Lower = catch more frauds"
    )
    
    st.info(f"""
    **Current Setting: {threshold_multiplier}x**
    - 0.5x: Very Sensitive
    - 1.0x: Balanced
    - 1.5x: Conservative
    - 2.0x: Very Conservative
    """)
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Session Statistics")
    st.metric("Total Processed", st.session_state.total_processed)
    st.metric("Frauds Detected", st.session_state.fraud_detected)
    if st.session_state.total_processed > 0:
        fraud_rate = (st.session_state.fraud_detected / st.session_state.total_processed) * 100
        st.metric("Detection Rate", f"{fraud_rate:.2f}%")
    
    # Reset button
    if st.button("🔄 Reset Statistics"):
        st.session_state.transaction_history = []
        st.session_state.fraud_detected = 0
        st.session_state.total_processed = 0
        st.rerun()

# Main content area
if not api_healthy:
    st.warning("⚠️ Please start the API server to use the system")
    st.code("cd src/api && python model_server.py", language="bash")
    st.stop()

# Mode: Single Transaction
if mode == "Single Transaction":
    st.header("💳 Single Transaction Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Transaction Input")
        
        # Input method
        input_method = st.radio("Input Method:", ["Random Sample", "Manual Entry"])
        
        if input_method == "Random Sample":
            df = load_sample_data()
            if df is not None:
                if st.button("🎲 Generate Random Transaction", type="primary"):
                    random_idx = np.random.randint(0, len(df))
                    row = df.iloc[random_idx]
                    transaction = create_transaction_from_row(row, random_idx)
                    
                    # Store in session state
                    st.session_state.current_transaction = transaction
                    st.session_state.show_prediction = True
                
                if 'current_transaction' in st.session_state:
                    tx = st.session_state.current_transaction
                    st.json({
                        "Transaction ID": tx['transaction_id'],
                        "Amount": f"${tx['amount']:.2f}",
                        "Timestamp": tx['timestamp'],
                        "Actual Label": "FRAUD" if tx['actual_label'] == 1 else "NORMAL"
                    })
        
        else:  # Manual Entry
            st.info("Enter transaction details manually")
            amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)
            hour = st.slider("Hour of Day", 0, 23, 12)
            
            if st.button("🔍 Analyze Transaction", type="primary"):
                # Create manual transaction
                features = {f'V{i}': float(np.random.randn()) for i in range(1, 29)}
                features['Amount'] = float(amount)
                features['Hour'] = float(hour)
                features['Log_Amount'] = float(np.log1p(amount))
                
                transaction = {
                    'transaction_id': f'manual_{int(time.time())}',
                    'timestamp': datetime.now().isoformat(),
                    'features': features,
                    'amount': float(amount),
                    'actual_label': 0  # Unknown for manual
                }
                
                st.session_state.current_transaction = transaction
                st.session_state.show_prediction = True
    
    with col2:
        st.subheader("Analysis Result")
        
        if 'show_prediction' in st.session_state and st.session_state.show_prediction:
            with st.spinner("Analyzing transaction..."):
                prediction = predict_fraud(st.session_state.current_transaction)
                
                if prediction:
                    # Update statistics
                    st.session_state.total_processed += 1
                    if prediction['is_fraud']:
                        st.session_state.fraud_detected += 1
                    
                    # Display result
                    if prediction['is_fraud']:
                        st.markdown(f"""
                        <div class="fraud-alert">
                            <h2>🚨 FRAUD DETECTED</h2>
                            <h3>Risk Score: {prediction['fraud_probability']:.1%}</h3>
                            <p>Transaction ID: {prediction['transaction_id']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="normal-transaction">
                            <h2>✅ LEGITIMATE TRANSACTION</h2>
                            <h3>Risk Score: {prediction['fraud_probability']:.1%}</h3>
                            <p>Transaction ID: {prediction['transaction_id']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Fraud Probability", f"{prediction['fraud_probability']:.1%}")
                    col_b.metric("Reconstruction Error", f"{prediction['reconstruction_error']:.4f}")
                    col_c.metric("Processing Time", f"{prediction['processing_time_ms']:.2f}ms")
                    
                    # Details
                    with st.expander("📋 Detailed Analysis"):
                        st.json(prediction)
                    
                    st.session_state.show_prediction = False

# Mode: Batch Processing
elif mode == "Batch Processing":
    st.header("📦 Batch Transaction Processing")
    
    df = load_sample_data()
    if df is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuration")
            
            num_transactions = st.number_input(
                "Number of transactions to process",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
            
            start_idx = st.number_input(
                "Start from row",
                min_value=0,
                max_value=len(df)-num_transactions,
                value=0
            )
            
            if st.button("🚀 Start Batch Processing", type="primary"):
                st.session_state.batch_running = True
                st.session_state.batch_results = []
        
        with col2:
            if 'batch_running' in st.session_state and st.session_state.batch_running:
                st.subheader("Processing...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                fraud_count = 0
                
                for i in range(num_transactions):
                    idx = start_idx + i
                    row = df.iloc[idx]
                    transaction = create_transaction_from_row(row, idx)
                    
                    prediction = predict_fraud(transaction)
                    
                    if prediction:
                        results.append({
                            'transaction_id': transaction['transaction_id'],
                            'amount': transaction['amount'],
                            'actual': 'FRAUD' if transaction['actual_label'] == 1 else 'NORMAL',
                            'predicted': 'FRAUD' if prediction['is_fraud'] else 'NORMAL',
                            'probability': prediction['fraud_probability'],
                            'latency_ms': prediction['processing_time_ms']
                        })
                        
                        if prediction['is_fraud']:
                            fraud_count += 1
                    
                    # Update progress
                    progress = (i + 1) / num_transactions
                    progress_bar.progress(progress)
                    status_text.text(f"Processed: {i+1}/{num_transactions} | Frauds: {fraud_count}")
                
                st.session_state.batch_results = results
                st.session_state.batch_running = False
                st.success(f"✅ Completed! Processed {num_transactions} transactions, detected {fraud_count} frauds")
                st.rerun()
        
        # Display results
        if 'batch_results' in st.session_state and st.session_state.batch_results:
            st.markdown("---")
            st.subheader("📊 Results")
            
            results_df = pd.DataFrame(st.session_state.batch_results)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total = len(results_df)
            frauds_detected = len(results_df[results_df['predicted'] == 'FRAUD'])
            actual_frauds = len(results_df[results_df['actual'] == 'FRAUD'])
            avg_latency = results_df['latency_ms'].mean()
            
            col1.metric("Total Processed", total)
            col2.metric("Frauds Detected", frauds_detected)
            col3.metric("Actual Frauds", actual_frauds)
            col4.metric("Avg Latency", f"{avg_latency:.2f}ms")
            
            # Confusion Matrix
            tp = len(results_df[(results_df['actual'] == 'FRAUD') & (results_df['predicted'] == 'FRAUD')])
            fp = len(results_df[(results_df['actual'] == 'NORMAL') & (results_df['predicted'] == 'FRAUD')])
            tn = len(results_df[(results_df['actual'] == 'NORMAL') & (results_df['predicted'] == 'NORMAL')])
            fn = len(results_df[(results_df['actual'] == 'FRAUD') & (results_df['predicted'] == 'NORMAL')])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                cm_data = [[tn, fp], [fn, tp]]
                fig = go.Figure(data=go.Heatmap(
                    z=cm_data,
                    x=['Predicted Normal', 'Predicted Fraud'],
                    y=['Actual Normal', 'Actual Fraud'],
                    text=cm_data,
                    texttemplate='%{text}',
                    textfont={"size": 20},
                    colorscale='RdYlGn_r'
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Performance Metrics")
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics_data = {
                    'Metric': ['Precision', 'Recall', 'F1 Score'],
                    'Value': [precision, recall, f1]
                }
                fig = px.bar(metrics_data, x='Metric', y='Value', 
                           title='Model Performance',
                           color='Value',
                           color_continuous_scale='RdYlGn')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Transaction table
            st.subheader("Transaction Details")
            st.dataframe(results_df, use_container_width=True)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Mode: Live Stream Demo
elif mode == "Live Stream Demo":
    st.header("📡 Live Transaction Stream Demo")
    
    st.info("This simulates real-time transaction processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Transaction Stream")
        
        stream_speed = st.select_slider(
            "Stream Speed",
            options=[0.5, 1, 2, 5],
            value=1,
            format_func=lambda x: f"{x}x"
        )
        
        num_stream = st.number_input(
            "Number of transactions",
            min_value=5,
            max_value=100,
            value=20
        )
        
        if st.button("▶️ Start Stream", type="primary"):
            st.session_state.streaming = True
    
    with col2:
        st.subheader("Live Statistics")
        metric_placeholder = st.empty()
    
    # Stream area
    stream_placeholder = st.empty()
    
    if 'streaming' in st.session_state and st.session_state.streaming:
        df = load_sample_data()
        
        if df is not None:
            processed = 0
            detected = 0
            
            for i in range(num_stream):
                idx = np.random.randint(0, len(df))
                row = df.iloc[idx]
                transaction = create_transaction_from_row(row, idx)
                
                prediction = predict_fraud(transaction)
                
                if prediction:
                    processed += 1
                    if prediction['is_fraud']:
                        detected += 1
                    
                    # Update metrics
                    with metric_placeholder.container():
                        col_a, col_b = st.columns(2)
                        col_a.metric("Processed", processed)
                        col_b.metric("Frauds", detected)
                    
                    # Show transaction
                    with stream_placeholder.container():
                        if prediction['is_fraud']:
                            st.error(f"🚨 FRAUD DETECTED | Amount: ${transaction['amount']:.2f} | "
                                   f"Risk: {prediction['fraud_probability']:.1%} | "
                                   f"Latency: {prediction['processing_time_ms']:.1f}ms")
                        else:
                            st.success(f"✅ Normal | Amount: ${transaction['amount']:.2f} | "
                                     f"Risk: {prediction['fraud_probability']:.1%} | "
                                     f"Latency: {prediction['processing_time_ms']:.1f}ms")
                    
                    time.sleep(1 / stream_speed)
            
            st.session_state.streaming = False
            st.success(f"Stream complete! Processed {processed} transactions, detected {detected} frauds")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p><strong>Real-Time Fraud Detection System</strong></p>
        <p>Built with PyTorch • FastAPI • Kafka • Streamlit</p>
        <p>Technologies: Deep Learning (Autoencoder) • Time-Series Analysis • Stream Processing</p>
    </div>
    """, unsafe_allow_html=True)