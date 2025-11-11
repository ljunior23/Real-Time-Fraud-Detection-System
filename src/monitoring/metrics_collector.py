
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time

"""
Collect and store metrics for monitoring
Uses Prometheus-style metrics
"""

class MetricsCollector:
    def __init__(self, port=8001):
        self.port = port
        
        # Define metrics
        self.transactions_total = Counter(
            'fraud_transactions_total',
            'Total number of transactions processed'
        )
        
        self.fraud_detected_total = Counter(
            'fraud_detected_total',
            'Total number of frauds detected'
        )
        
        self.actual_fraud_total = Counter(
            'fraud_actual_total',
            'Total number of actual frauds'
        )
        
        self.prediction_latency = Histogram(
            'fraud_prediction_latency_seconds',
            'Prediction latency in seconds',
            buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
        )
        
        self.reconstruction_error = Histogram(
            'fraud_reconstruction_error',
            'Reconstruction error from autoencoder',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.true_positives = Counter('fraud_true_positives', 'True positives')
        self.false_positives = Counter('fraud_false_positives', 'False positives')
        self.true_negatives = Counter('fraud_true_negatives', 'True negatives')
        self.false_negatives = Counter('fraud_false_negatives', 'False negatives')
        
        # Current precision/recall gauges
        self.precision_gauge = Gauge('fraud_precision', 'Current precision')
        self.recall_gauge = Gauge('fraud_recall', 'Current recall')
        self.f1_gauge = Gauge('fraud_f1_score', 'Current F1 score')
    
    def start_server(self):
        """Start Prometheus metrics server"""
        start_http_server(self.port)
        print(f"✓ Metrics server started on port {self.port}")
        print(f"  Access metrics at: http://localhost:{self.port}/metrics")
    
    def update_metrics(self, prediction_result, actual_label):
        """Update metrics based on prediction"""
        # Increment counters
        self.transactions_total.inc()
        
        is_fraud_predicted = prediction_result['is_fraud']
        is_fraud_actual = actual_label == 1
        
        if is_fraud_predicted:
            self.fraud_detected_total.inc()
        
        if is_fraud_actual:
            self.actual_fraud_total.inc()
        
        # Latency
        latency_seconds = prediction_result['processing_time_ms'] / 1000
        self.prediction_latency.observe(latency_seconds)
        
        # Reconstruction error
        self.reconstruction_error.observe(prediction_result['reconstruction_error'])
        
        # Confusion matrix
        if is_fraud_predicted and is_fraud_actual:
            self.true_positives.inc()
        elif is_fraud_predicted and not is_fraud_actual:
            self.false_positives.inc()
        elif not is_fraud_predicted and is_fraud_actual:
            self.false_negatives.inc()
        else:
            self.true_negatives.inc()
        
        # Update gauges
        self._update_performance_gauges()
    
    def _update_performance_gauges(self):
        """Calculate and update precision, recall, F1"""
        tp = self.true_positives._value._value
        fp = self.false_positives._value._value
        fn = self.false_negatives._value._value
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        self.precision_gauge.set(precision)
        self.recall_gauge.set(recall)
        self.f1_gauge.set(f1)


if __name__ == "__main__":
    # Example: Run metrics collector
    collector = MetricsCollector(port=8001)
    collector.start_server()
    
    print("Metrics collector running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")