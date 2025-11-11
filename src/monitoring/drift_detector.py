
import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_set import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns
)
from datetime import datetime, timedelta
import logging
from collections import deque
from typing import Dict, List

"""
Data Drift and Model Performance Monitoring using Evidently AI
Detects distribution shifts and model degradation
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Real-time drift detection for fraud detection model
    """
    
    def __init__(self, 
                 reference_data_path='../../data/reference_data.csv',
                 window_size=1000,
                 drift_threshold=0.5):
        """
        Args:
            reference_data_path: Path to reference dataset (training data)
            window_size: Size of rolling window for current data
            drift_threshold: Threshold for drift detection (0-1)
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Load reference data
        self.reference_data = None
        self.load_reference_data(reference_data_path)
        
        # Current data buffer
        self.current_data_buffer = deque(maxlen=window_size)
        
        # Drift history
        self.drift_history = []
        
        # Feature columns
        self.feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour', 'Log_Amount']
        
        # Column mapping for Evidently
        self.column_mapping = ColumnMapping(
            target='Class',
            prediction='prediction',
            numerical_features=self.feature_columns
        )
    
    def load_reference_data(self, path):
        """Load reference dataset (training data)"""
        try:
            self.reference_data = pd.read_csv(path)
            logger.info(f"Loaded reference data: {len(self.reference_data)} samples")
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            # Create synthetic reference data if file doesn't exist
            self.reference_data = None
    
    def add_transaction(self, transaction: Dict, prediction: Dict):
        """Add new transaction with prediction to current data buffer"""
        record = {
            **transaction['features'],
            'Class': transaction.get('actual_label', 0),
            'prediction': 1 if prediction['is_fraud'] else 0,
            'fraud_probability': prediction['fraud_probability'],
            'reconstruction_error': prediction['reconstruction_error'],
            'timestamp': transaction['timestamp']
        }
        self.current_data_buffer.append(record)
    
    def check_drift(self) -> Dict:
        """
        Check for data drift
        Returns drift report with metrics
        """
        if len(self.current_data_buffer) < self.window_size // 2:
            return {
                'drift_detected': False,
                'message': 'Insufficient data for drift detection',
                'samples_collected': len(self.current_data_buffer)
            }
        
        # Convert current data to DataFrame
        current_df = pd.DataFrame(list(self.current_data_buffer))
        
        # Create drift report
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_df,
            column_mapping=self.column_mapping
        )
        
        # Extract drift metrics
        report_dict = report.as_dict()
        
        # Parse results
        dataset_drift = report_dict['metrics'][0]['result']
        drift_detected = dataset_drift['dataset_drift']
        drift_share = dataset_drift['drift_share']
        
        # Get drifted features
        drifted_features = []
        if 'drift_by_columns' in dataset_drift:
            for feature, metrics in dataset_drift['drift_by_columns'].items():
                if metrics.get('drift_detected', False):
                    drifted_features.append({
                        'feature': feature,
                        'drift_score': metrics.get('drift_score', 0),
                        'stattest_name': metrics.get('stattest_name', 'unknown')
                    })
        
        result = {
            'drift_detected': drift_detected,
            'drift_share': drift_share,
            'drifted_features': drifted_features,
            'num_drifted_features': len(drifted_features),
            'total_features': len(self.feature_columns),
            'timestamp': datetime.now().isoformat(),
            'samples_analyzed': len(current_df)
        }
        
        # Store in history
        self.drift_history.append(result)
        
        return result
    
    def generate_drift_report(self, output_path='../../reports/drift_report.html'):
        """Generate comprehensive drift report"""
        if len(self.current_data_buffer) < 100:
            logger.warning("Insufficient data for comprehensive report")
            return
        
        current_df = pd.DataFrame(list(self.current_data_buffer))
        
        # Create comprehensive report
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_df,
            column_mapping=self.column_mapping
        )
        
        # Save report
        report.save_html(output_path)
        logger.info(f"Drift report saved to {output_path}")
        
        return output_path
    
    def run_drift_tests(self) -> Dict:
        """
        Run drift tests with pass/fail criteria
        """
        if len(self.current_data_buffer) < self.window_size // 2:
            return {'status': 'insufficient_data'}
        
        current_df = pd.DataFrame(list(self.current_data_buffer))
        
        # Create test suite
        tests = TestSuite(tests=[
            TestNumberOfDriftedColumns(lt=5),  # Less than 5 drifted columns
            TestShareOfDriftedColumns(lt=0.3),  # Less than 30% drift
        ])
        
        tests.run(
            reference_data=self.reference_data,
            current_data=current_df,
            column_mapping=self.column_mapping
        )
        
        # Get results
        test_results = tests.as_dict()
        
        return {
            'all_tests_passed': test_results['summary']['all_passed'],
            'tests_passed': test_results['summary']['success_tests'],
            'tests_failed': test_results['summary']['failed_tests'],
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_psi(self, reference_values: np.ndarray, current_values: np.ndarray, buckets=10) -> float:
        """
        Calculate Population Stability Index (PSI)
        PSI < 0.1: No significant shift
        PSI 0.1-0.25: Moderate shift
        PSI > 0.25: Significant shift
        """
        def scale_range(input_array, min_val, max_val):
            return (input_array - min_val) / (max_val - min_val)
        
        # Normalize both arrays
        min_val = min(reference_values.min(), current_values.min())
        max_val = max(reference_values.max(), current_values.max())
        
        reference_scaled = scale_range(reference_values, min_val, max_val)
        current_scaled = scale_range(current_values, min_val, max_val)
        
        # Create bins
        breakpoints = np.linspace(0, 1, buckets + 1)
        
        # Calculate distributions
        reference_dist = np.histogram(reference_scaled, bins=breakpoints)[0] / len(reference_scaled)
        current_dist = np.histogram(current_scaled, bins=breakpoints)[0] / len(current_scaled)
        
        # Add small epsilon to avoid division by zero
        reference_dist = reference_dist + 1e-10
        current_dist = current_dist + 1e-10
        
        # Calculate PSI
        psi = np.sum((current_dist - reference_dist) * np.log(current_dist / reference_dist))
        
        return psi
    
    def monitor_feature_psi(self) -> Dict:
        """Monitor PSI for all features"""
        if len(self.current_data_buffer) < 100:
            return {}
        
        current_df = pd.DataFrame(list(self.current_data_buffer))
        psi_scores = {}
        
        for feature in self.feature_columns:
            if feature in self.reference_data.columns and feature in current_df.columns:
                psi = self.calculate_psi(
                    self.reference_data[feature].values,
                    current_df[feature].values
                )
                psi_scores[feature] = {
                    'psi': float(psi),
                    'status': 'stable' if psi < 0.1 else 'moderate_shift' if psi < 0.25 else 'significant_shift'
                }
        
        return psi_scores


class PerformanceMonitor:
    """
    Monitor model performance metrics over time
    """
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.predictions_buffer = deque(maxlen=window_size)
        self.performance_history = []
    
    def add_prediction(self, actual: int, predicted: int, probability: float):
        """Add prediction result"""
        self.predictions_buffer.append({
            'actual': actual,
            'predicted': predicted,
            'probability': probability,
            'timestamp': datetime.now()
        })
    
    def calculate_metrics(self) -> Dict:
        """Calculate current performance metrics"""
        if len(self.predictions_buffer) < 10:
            return {}
        
        df = pd.DataFrame(list(self.predictions_buffer))
        
        # Confusion matrix
        tp = ((df['actual'] == 1) & (df['predicted'] == 1)).sum()
        fp = ((df['actual'] == 0) & (df['predicted'] == 1)).sum()
        tn = ((df['actual'] == 0) & (df['predicted'] == 0)).sum()
        fn = ((df['actual'] == 1) & (df['predicted'] == 0)).sum()
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'timestamp': datetime.now().isoformat(),
            'samples': len(df)
        }
        
        self.performance_history.append(metrics)
        
        return metrics
    
    def detect_performance_degradation(self, baseline_metrics: Dict, threshold=0.1) -> Dict:
        """
        Detect if model performance has degraded
        threshold: acceptable degradation in F1 score (default 10%)
        """
        current_metrics = self.calculate_metrics()
        
        if not current_metrics or not baseline_metrics:
            return {'degradation_detected': False, 'message': 'Insufficient data'}
        
        f1_drop = baseline_metrics['f1_score'] - current_metrics['f1_score']
        precision_drop = baseline_metrics['precision'] - current_metrics['precision']
        recall_drop = baseline_metrics['recall'] - current_metrics['recall']
        
        degradation_detected = f1_drop > threshold
        
        return {
            'degradation_detected': degradation_detected,
            'f1_drop': float(f1_drop),
            'precision_drop': float(precision_drop),
            'recall_drop': float(recall_drop),
            'current_f1': current_metrics['f1_score'],
            'baseline_f1': baseline_metrics['f1_score'],
            'timestamp': datetime.now().isoformat()
        }


class AlertManager:
    """
    Manage alerts for drift and performance issues
    """
    
    def __init__(self, alert_callback=None):
        self.alerts = []
        self.alert_callback = alert_callback
    
    def check_and_alert(self, drift_result: Dict, performance_result: Dict):
        """Check conditions and trigger alerts"""
        alerts = []
        
        # Drift alerts
        if drift_result.get('drift_detected', False):
            alert = {
                'type': 'DRIFT_DETECTED',
                'severity': 'WARNING',
                'message': f"Data drift detected: {drift_result['drift_share']:.2%} of features drifted",
                'details': drift_result,
                'timestamp': datetime.now().isoformat()
            }
            alerts.append(alert)
        
        # Performance alerts
        if performance_result.get('degradation_detected', False):
            alert = {
                'type': 'PERFORMANCE_DEGRADATION',
                'severity': 'CRITICAL',
                'message': f"Model performance degraded: F1 dropped by {performance_result['f1_drop']:.3f}",
                'details': performance_result,
                'timestamp': datetime.now().isoformat()
            }
            alerts.append(alert)
        
        # Store and trigger alerts
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(f"🚨 ALERT: {alert['message']}")
            
            if self.alert_callback:
                self.alert_callback(alert)
        
        return alerts
    
    def get_recent_alerts(self, hours=24) -> List[Dict]:
        """Get alerts from last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff
        ]


# Integration example
if __name__ == "__main__":
    # Initialize monitors
    drift_detector = DriftDetector()
    performance_monitor = PerformanceMonitor()
    alert_manager = AlertManager()
    
    # Simulate monitoring
    print("Drift detection initialized")
    print("Add transactions via drift_detector.add_transaction()")