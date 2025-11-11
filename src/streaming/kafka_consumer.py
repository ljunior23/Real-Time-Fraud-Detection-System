"""
Kafka Consumer - Reads transactions and calls fraud detection API
"""

import json
import time
import os
import requests
from kafka import KafkaConsumer
from datetime import datetime

# Configuration from environment variables
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'transactions')
API_URL = os.getenv('API_URL', 'http://localhost:8000')
PRINT_INTERVAL = int(os.getenv('PRINT_INTERVAL', '10'))

def predict_fraud(transaction):
    """Call API to predict fraud"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=transaction,
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"✗ API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return None

def main():
    print("=" * 60)
    print("KAFKA CONSUMER - Fraud Detection Pipeline")
    print("=" * 60)
    print(f"Kafka Brokers: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Topic: {KAFKA_TOPIC}")
    print(f"API: {API_URL}")
    print("=" * 60)
    
    # Wait for services to be ready
    print("\nWaiting for Kafka and API to be ready...")
    time.sleep(15)
    
    # Check API health
    print("Checking API health...")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            print("✓ API is healthy")
        else:
            print(f"⚠ API returned status {health.status_code}")
    except Exception as e:
        print(f"⚠ Could not reach API: {e}")
    
    # Create Kafka consumer
    print("\nConnecting to Kafka...")
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='fraud-detection-consumer'
    )
    print(f"✓ Connected to Kafka topic '{KAFKA_TOPIC}'")
    
    # Statistics
    processed = 0
    fraud_detected = 0
    true_frauds = 0
    true_positives = 0
    false_positives = 0
    start_time = time.time()
    
    print("\nProcessing transactions...")
    print("-" * 60)
    
    try:
        for message in consumer:
            transaction = message.value
            
            # Get prediction
            prediction = predict_fraud(transaction)
            
            if prediction:
                processed += 1
                
                # Count frauds
                actual_fraud = transaction.get('actual_label', 0) == 1
                predicted_fraud = prediction['is_fraud']
                
                if actual_fraud:
                    true_frauds += 1
                
                if predicted_fraud:
                    fraud_detected += 1
                    
                    if actual_fraud:
                        true_positives += 1
                    else:
                        false_positives += 1
                
                # Print progress
                if processed % PRINT_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    
                    precision = (true_positives / fraud_detected * 100) if fraud_detected > 0 else 0
                    recall = (true_positives / true_frauds * 100) if true_frauds > 0 else 0
                    
                    print(f"Processed: {processed:>5} | "
                          f"Rate: {rate:>6.1f} tx/s | "
                          f"Detected: {fraud_detected:>3} | "
                          f"Precision: {precision:>5.1f}% | "
                          f"Recall: {recall:>5.1f}%")
                
    except KeyboardInterrupt:
        print("\n\nStopping consumer...")
    
    finally:
        # Summary
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("CONSUMER SUMMARY")
        print("=" * 60)
        print(f"Total Processed: {processed:,}")
        print(f"Processing Time: {elapsed:.1f}s")
        print(f"Average Rate: {processed/elapsed:.1f} transactions/sec")
        print()
        print(f"Actual Frauds: {true_frauds}")
        print(f"Detected Frauds: {fraud_detected}")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print()
        
        if fraud_detected > 0:
            precision = (true_positives / fraud_detected) * 100
            print(f"Precision: {precision:.2f}%")
        
        if true_frauds > 0:
            recall = (true_positives / true_frauds) * 100
            print(f"Recall: {recall:.2f}%")
        
        print("=" * 60)
        
        consumer.close()
        print("\n✓ Consumer closed")

if __name__ == "__main__":
    main()