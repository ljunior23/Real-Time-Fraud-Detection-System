
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

"""
Advanced time-series feature engineering for fraud detection
Includes: Velocity, Temporal Patterns, Statistical Features
"""

class TimeSeriesFeatureEngine:
    """
    Feature engineering for streaming fraud detection with time-series features
    """
    
    def __init__(self, windows=[60, 300, 3600, 86400]):
        """
        Args:
            windows: Time windows in seconds [1min, 5min, 1hour, 1day]
        """
        self.windows = windows
        # In-memory store for feature calculation (use Redis in production)
        self.transaction_history = deque(maxlen=10000)
        self.user_stats = {}  # user_id -> stats
        
    def add_transaction(self, transaction):
        """Add transaction to history"""
        self.transaction_history.append(transaction)
    
    def extract_velocity_features(self, current_transaction, user_id=None):
        """
        Transaction Velocity Features:
        - Number of transactions in time windows
        - Transaction frequency
        - Amount velocity (sum, avg in windows)
        - Time since last transaction
        """
        features = {}
        current_time = current_transaction.get('timestamp', datetime.now())
        current_amount = current_transaction.get('Amount', 0)
        
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)
        
        # Filter relevant transactions (same user if available, or all)
        if user_id:
            relevant_txs = [tx for tx in self.transaction_history 
                          if tx.get('user_id') == user_id]
        else:
            # For this dataset without user_ids, use all transactions
            relevant_txs = list(self.transaction_history)
        
        # Calculate for each time window
        for window in self.windows:
            window_start = current_time - timedelta(seconds=window)
            
            # Transactions in window
            txs_in_window = [
                tx for tx in relevant_txs 
                if pd.to_datetime(tx.get('timestamp', 0)) >= window_start
            ]
            
            # Count
            features[f'tx_count_{window}s'] = len(txs_in_window)
            
            # Frequency (transactions per hour)
            hours = window / 3600
            features[f'tx_frequency_{window}s'] = len(txs_in_window) / hours if hours > 0 else 0
            
            # Amount statistics
            if txs_in_window:
                amounts = [tx.get('Amount', 0) for tx in txs_in_window]
                features[f'amount_sum_{window}s'] = sum(amounts)
                features[f'amount_mean_{window}s'] = np.mean(amounts)
                features[f'amount_std_{window}s'] = np.std(amounts) if len(amounts) > 1 else 0
                features[f'amount_max_{window}s'] = max(amounts)
                features[f'amount_min_{window}s'] = min(amounts)
            else:
                features[f'amount_sum_{window}s'] = 0
                features[f'amount_mean_{window}s'] = 0
                features[f'amount_std_{window}s'] = 0
                features[f'amount_max_{window}s'] = 0
                features[f'amount_min_{window}s'] = 0
            
            # Velocity features (amount per hour)
            features[f'amount_velocity_{window}s'] = features[f'amount_sum_{window}s'] / hours if hours > 0 else 0
        
        # Time since last transaction
        if relevant_txs:
            last_tx_time = pd.to_datetime(relevant_txs[-1].get('timestamp', current_time))
            features['time_since_last_tx'] = (current_time - last_tx_time).total_seconds()
        else:
            features['time_since_last_tx'] = 0
        
        # Comparison with current transaction
        if relevant_txs:
            recent_amounts = [tx.get('Amount', 0) for tx in relevant_txs[-10:]]
            if recent_amounts:
                avg_recent = np.mean(recent_amounts)
                features['amount_vs_recent_avg'] = current_amount / avg_recent if avg_recent > 0 else 0
                features['amount_deviation'] = (current_amount - avg_recent) / np.std(recent_amounts) if len(recent_amounts) > 1 else 0
        
        return features
    
    def extract_temporal_pattern_features(self, transaction):
        """
        Temporal Pattern Features:
        - Hour of day
        - Day of week
        - Is weekend
        - Is night (11pm - 6am)
        - Time of day category
        - Cyclical encoding of time
        """
        features = {}
        timestamp = transaction.get('timestamp', datetime.now())
        
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Basic temporal features
        features['hour'] = timestamp.hour
        features['day_of_week'] = timestamp.dayofweek
        features['day_of_month'] = timestamp.day
        features['is_weekend'] = 1 if timestamp.dayofweek >= 5 else 0
        features['is_night'] = 1 if timestamp.hour >= 23 or timestamp.hour < 6 else 0
        features['is_business_hours'] = 1 if 9 <= timestamp.hour <= 17 else 0
        
        # Time of day category (0: night, 1: morning, 2: afternoon, 3: evening)
        if 0 <= timestamp.hour < 6:
            features['time_of_day'] = 0  # night
        elif 6 <= timestamp.hour < 12:
            features['time_of_day'] = 1  # morning
        elif 12 <= timestamp.hour < 18:
            features['time_of_day'] = 2  # afternoon
        else:
            features['time_of_day'] = 3  # evening
        
        # Cyclical encoding (sine/cosine for periodicity)
        features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * timestamp.dayofweek / 7)
        features['day_cos'] = np.cos(2 * np.pi * timestamp.dayofweek / 7)
        
        # Minute of hour (for finer granularity)
        features['minute'] = timestamp.minute
        features['minute_sin'] = np.sin(2 * np.pi * timestamp.minute / 60)
        features['minute_cos'] = np.cos(2 * np.pi * timestamp.minute / 60)
        
        return features
    
    def extract_statistical_features(self, current_transaction, user_id=None):
        """
        Statistical Time-Series Features:
        - Moving averages (EMA)
        - Volatility measures
        - Trend indicators
        - Z-scores
        - Percentile ranks
        """
        features = {}
        current_amount = current_transaction.get('Amount', 0)
    
        # Filter relevant transactions
        if user_id:
            relevant_txs = [tx for tx in self.transaction_history 
                      if tx.get('user_id') == user_id]
        else:
            relevant_txs = list(self.transaction_history)
    
        if not relevant_txs:
            return self._get_default_statistical_features()
    
        # Get recent amounts
        recent_amounts = np.array([tx.get('Amount', 0) for tx in relevant_txs[-100:]])
    
        if len(recent_amounts) == 0:
            return self._get_default_statistical_features()
    
        # Exponential Moving Average (EMA)
        if len(recent_amounts) >= 10:
            ema_10 = self._calculate_ema(recent_amounts, span=10)
            ema_30 = self._calculate_ema(recent_amounts, span=30)
            features['ema_10'] = float(ema_10)
            features['ema_30'] = float(ema_30)
            features['ema_ratio'] = float(ema_10 / ema_30) if ema_30 > 0 else 1.0
        else:
            features['ema_10'] = float(np.mean(recent_amounts))
            features['ema_30'] = float(np.mean(recent_amounts))
            features['ema_ratio'] = 1.0
    
        # Volatility (rolling standard deviation)
        if len(recent_amounts) >= 10:
            features['volatility_10'] = float(np.std(recent_amounts[-10:]))
            features['volatility_30'] = float(np.std(recent_amounts[-30:])) if len(recent_amounts) >= 30 else float(np.std(recent_amounts))
        else:
            features['volatility_10'] = 0.0
            features['volatility_30'] = 0.0
    
        # Z-score (how many standard deviations from mean)
        mean_amount = np.mean(recent_amounts)
        std_amount = np.std(recent_amounts)
        features['z_score'] = float((current_amount - mean_amount) / std_amount) if std_amount > 0 else 0.0
    
        # Percentile rank
        features['percentile_rank'] = float((recent_amounts < current_amount).sum() / len(recent_amounts) * 100)
    
        # Trend indicators
        if len(recent_amounts) >= 5:
            # Simple linear trend
            x = np.arange(len(recent_amounts[-20:]))
            y = recent_amounts[-20:]
            if len(x) > 1:
                trend = np.polyfit(x, y, 1)[0]  # slope
                features['trend'] = float(trend)
            else:
                features['trend'] = 0.0
        else:
            features['trend'] = 0.0
    
        # Rate of change
        if len(recent_amounts) >= 2:
            features['roc_1'] = float((recent_amounts[-1] - recent_amounts[-2]) / recent_amounts[-2]) if recent_amounts[-2] > 0 else 0.0
            if len(recent_amounts) >= 5:
                features['roc_5'] = float((recent_amounts[-1] - recent_amounts[-5]) / recent_amounts[-5]) if recent_amounts[-5] > 0 else 0.0
            else:
                features['roc_5'] = 0.0
        else:
            features['roc_1'] = 0.0
            features['roc_5'] = 0.0
    
        # Coefficient of variation (CV)
        features['coef_variation'] = float(std_amount / mean_amount) if mean_amount > 0 else 0.0
    
        # Interquartile range
        if len(recent_amounts) >= 4:
            q75, q25 = np.percentile(recent_amounts, [75, 25])
            iqr = q75 - q25
            features['iqr'] = float(iqr)
            features['iqr_ratio'] = float((current_amount - q25) / iqr) if iqr > 0 else 0.0
        else:
            features['iqr'] = 0.0
            features['iqr_ratio'] = 0.0
    
        # Skewness (measure of asymmetry)
        if len(recent_amounts) >= 3:
            from scipy import stats
            features['skewness'] = float(stats.skew(recent_amounts))
            features['kurtosis'] = float(stats.kurtosis(recent_amounts))
        else:
            features['skewness'] = 0.0
            features['kurtosis'] = 0.0
    
        # CRITICAL: Replace any NaN or Inf values with defaults
        for key, value in features.items():
            if not np.isfinite(value):  # Catches NaN and Inf
                features[key] = 0.0
    
        return features
    
    def _get_default_statistical_features(self):
        """Return default values when no history available"""
        return {
            'ema_10': 0.0, 
            'ema_30': 0.0, 
            'ema_ratio': 1.0,
            'volatility_10': 0.0, 
            'volatility_30': 0.0,
            'z_score': 0.0, 
            'percentile_rank': 50.0,
            'trend': 0.0, 
            'roc_1': 0.0, 
            'roc_5': 0.0,
            'coef_variation': 0.0, 
            'iqr': 0.0, 
            'iqr_ratio': 0.0,
            'skewness': 0.0, 
            'kurtosis': 0.0
        }
    
    def extract_all_features(self, transaction, user_id=None):
        """Extract all time-series features"""
        features = {}
        
        # Velocity features
        velocity_features = self.extract_velocity_features(transaction, user_id)
        features.update(velocity_features)
        
        # Temporal pattern features
        temporal_features = self.extract_temporal_pattern_features(transaction)
        features.update(temporal_features)
        
        # Statistical features
        statistical_features = self.extract_statistical_features(transaction, user_id)
        features.update(statistical_features)
        
        # Add transaction to history
        self.add_transaction(transaction)
        
        return features


# Test the feature engine
if __name__ == "__main__":
    # Example usage
    feature_engine = TimeSeriesFeatureEngine()
    
    # Simulate transactions
    transactions = [
        {'timestamp': datetime.now() - timedelta(seconds=i*60), 
         'Amount': np.random.uniform(10, 500),
         'user_id': 'user_123'}
        for i in range(100, 0, -1)
    ]
    
    # Add historical transactions
    for tx in transactions:
        feature_engine.add_transaction(tx)
    
    # Extract features for new transaction
    new_tx = {
        'timestamp': datetime.now(),
        'Amount': 1000,
        'user_id': 'user_123'
    }
    
    features = feature_engine.extract_all_features(new_tx, user_id='user_123')
    
    print("Extracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")