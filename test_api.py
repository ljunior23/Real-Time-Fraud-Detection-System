import requests
import time
import numpy as np

# Test prediction endpoint
url = "http://localhost:8000/predict"

# Create sample transaction
transaction = {
    "transaction_id": "test_001",
    "timestamp": "2024-10-27T21:00:00",
    "amount": 150.50,
    "features": {
        **{f"V{i}": float(np.random.randn()) for i in range(1, 29)},
        "Amount": 150.50,
        "Hour": 21.0,
        "Log_Amount": 5.01
    }
}

# Benchmark
num_requests = 100
latencies = []

print(f"Sending {num_requests} requests...")

for i in range(num_requests):
    start = time.time()
    response = requests.post(url, json=transaction)
    latency = (time.time() - start) * 1000  # ms
    latencies.append(latency)
    
    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/{num_requests}")

# Results
print("\n" + "="*50)
print("API Performance Test Results")
print("="*50)
print(f"Total Requests: {num_requests}")
print(f"Average Latency: {np.mean(latencies):.2f}ms")
print(f"P50 Latency: {np.percentile(latencies, 50):.2f}ms")
print(f"P95 Latency: {np.percentile(latencies, 95):.2f}ms")
print(f"P99 Latency: {np.percentile(latencies, 99):.2f}ms")
print(f"Min Latency: {np.min(latencies):.2f}ms")
print(f"Max Latency: {np.max(latencies):.2f}ms")
print(f"Throughput: {num_requests / (sum(latencies)/1000):.1f} req/sec")