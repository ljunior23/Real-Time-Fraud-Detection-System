"""
Kafka Producer - Sends credit card transactions to Kafka topic
"""

import json
import time
import os
import pandas as pd
import numpy as np
from kafka import KafkaProducer
from datetime import datetime

# Configuration from environment variables
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'transactions')
CSV_PATH = os.getenv('CSV_PATH', '/app/data/creditcard.csv')
MAX_RECORDS = int(os.getenv('MAX_RECORDS', '1000'))
DELAY_SECONDS = float(os.getenv('DELAY_SECONDS', '0.1'))

def create_transaction_message(row, idx):
    """Create transaction message from DataFrame row"""
    features = {}
    for i in range(1, 29):
        features[f'V{i}'] = float(row[f'V{i}'])
    
    features['Amount'] = float(row['Amount'])
    features['Hour'] = float((row['Time'] / 3600) % 24)
    features['Log_Amount'] = float(np.log1p(row['Amount']))
    
    return {
        'transaction_id': f'tx_{idx}_{int(time.time())}',
        'timestamp': datetime.now().isoformat(),
        'amount': float(row['Amount']),
        'features': features,
        'actual_label': int(row['Class'])  # For evaluation
    }

def main():
    print("=" * 60)
    print("KAFKA PRODUCER - Credit Card Fraud Detection")
    print("=" * 60)
    print(f"Kafka Brokers: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Topic: {KAFKA_TOPIC}")
    print(f"Data Source: {CSV_PATH}")
    print(f"Max Records: {MAX_RECORDS}")
    print(f"Delay: {DELAY_SECONDS}s")
    print("=" * 60)
    
    # Wait for Kafka to be ready
    print("\nWaiting for Kafka to be ready...")
    time.sleep(10)
    
    # Create Kafka producer
    print("Connecting to Kafka...")
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        retries=3
    )
    print("✓ Connected to Kafka")
    
    # Load data
    print(f"\nLoading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Loaded {len(df):,} transactions")
    
    # Send transactions
    print(f"\nSending transactions to topic '{KAFKA_TOPIC}'...")
    print("-" * 60)
    
    sent_count = 0
    fraud_count = 0
    
    try:
        for idx, row in df.iterrows():
            if sent_count >= MAX_RECORDS:
                break
            
            # Create message
            message = create_transaction_message(row, idx)
            
            # Send to Kafka
            producer.send(KAFKA_TOPIC, value=message)
            
            sent_count += 1
            if message['actual_label'] == 1:
                fraud_count += 1
            
            # Log progress
            if sent_count % 100 == 0:
                fraud_rate = (fraud_count / sent_count) * 100
                print(f"Sent: {sent_count:>5} | Frauds: {fraud_count:>3} | "
                      f"Rate: {fraud_rate:>5.2f}% | "
                      f"Latest: ${message['amount']:.2f}")
            
            # Delay to simulate real-time
            time.sleep(DELAY_SECONDS)
        
        # Flush remaining messages
        producer.flush()
        
    except KeyboardInterrupt:
        print("\n\nStopping producer...")
    
    finally:
        # Summary
        print("\n" + "=" * 60)
        print("PRODUCER SUMMARY")
        print("=" * 60)
        print(f"Total Sent: {sent_count:,}")
        print(f"Frauds: {fraud_count} ({(fraud_count/sent_count)*100:.2f}%)")
        print(f"Normal: {sent_count - fraud_count} ({((sent_count-fraud_count)/sent_count)*100:.2f}%)")
        print("=" * 60)
        
        producer.close()
        print("\n✓ Producer closed")

if __name__ == "__main__":
    main()