
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score
import joblib
from datetime import datetime
import logging
import json
from pathlib import Path
from typing import Dict

"""
Automated model retraining pipeline
Triggers retraining based on drift/performance degradation
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """Autoencoder model architecture"""
    def __init__(self, input_dim, encoding_dim=14):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class FraudDataset(Dataset):
    """Dataset for fraud detection"""
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class RetrainingPipeline:
    """
    Automated retraining pipeline with versioning
    """
    
    def __init__(self, 
                 data_path='../../data/creditcard.csv',
                 model_dir='../../models',
                 config_path='../../config/training_config.json'):
        self.data_path = data_path
        self.model_dir = Path(model_dir)
        self.config_path = config_path
        
        # Load config
        self.config = self.load_config()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_config(self) -> dict:
        """Load training configuration"""
        default_config = {
            'batch_size': 256,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'encoding_dim': 14,
            'validation_split': 0.2,
            'early_stopping_patience': 5
        }
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return {**default_config, **config}
        except:
            logger.warning("Config file not found, using defaults")
            return default_config
    
    def load_and_prepare_data(self, use_recent_data=True, recent_days=30):
        """
        Load and prepare data for training
        Optionally use only recent data for incremental learning
        """
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Add features
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Log_Amount'] = np.log1p(df['Amount'])
        
        if use_recent_data:
            # Use only recent transactions (simulated with last N rows)
            recent_threshold = len(df) - (recent_days * 10000)  # Approx
            df = df[recent_threshold:]
            logger.info(f"Using recent data: {len(df)} samples")
        
        # Features and labels
        feature_cols = [col for col in df.columns if col not in ['Time', 'Class']]
        X = df[feature_cols].values
        y = df['Class'].values
        
        # Use only normal transactions for training
        X_normal = X[y == 0]
        
        logger.info(f"Training samples (normal only): {len(X_normal)}")
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Fraud rate: {y.sum() / len(y) * 100:.4f}%")
        
        return X_normal, X, y, feature_cols
    
    def train_model(self, X_train, validation_split=0.2):
        """
        Train new model
        """
        # Split into train/validation
        val_size = int(len(X_train) * validation_split)
        X_val = X_train[-val_size:]
        X_train = X_train[:-val_size]
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create datasets
        train_dataset = FraudDataset(X_train_scaled)
        val_dataset = FraudDataset(X_val_scaled)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        # Initialize model
        input_dim = X_train.shape[1]
        model = Autoencoder(
            input_dim=input_dim,
            encoding_dim=self.config['encoding_dim']
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        logger.info("Starting training...")
        
        for epoch in range(self.config['num_epochs']):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                
                outputs = model(batch)
                loss = criterion(outputs, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    outputs = model(batch)
                    loss = criterion(outputs, batch)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            logger.info(f"Epoch [{epoch+1}/{self.config['num_epochs']}] "
                       f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Training complete!")
        
        return model, scaler, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
    
    def evaluate_model(self, model, scaler, X_test, y_test):
        """
        Evaluate model on test set
        """
        model.eval()
        
        # Scale test data
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate reconstruction errors
        reconstruction_errors = []
        batch_size = 1024
        
        with torch.no_grad():
            for i in range(0, len(X_test_scaled), batch_size):
                batch = X_test_scaled[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                outputs = model(batch_tensor)
                errors = torch.mean((batch_tensor - outputs) ** 2, dim=1)
                reconstruction_errors.extend(errors.cpu().numpy())
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Find optimal threshold
        threshold = np.percentile(reconstruction_errors, 95)
        
        # Calculate metrics
        from sklearn.metrics import (classification_report, precision_recall_curve,
                                     average_precision_score, roc_auc_score)
        
        y_pred = (reconstruction_errors > threshold).astype(int)
        
        # Metrics
        roc_auc = roc_auc_score(y_test, reconstruction_errors)
        pr_auc = average_precision_score(y_test, reconstruction_errors)
        
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION")
        logger.info("="*60)
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"PR-AUC: {pr_auc:.4f}")
        logger.info(f"Threshold: {threshold:.6f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        return {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'threshold': float(threshold),
            'reconstruction_errors': reconstruction_errors.tolist()
        }
    
    def save_model(self, model, scaler, metrics, version=None):
        """
        Save model with versioning (PyTorch 2.6+ compatible)
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
        # Create version directory
        version_dir = self.model_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
    
        # Save model - ensure all values are Python native types
        model_path = version_dir / 'autoencoder_fraud.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': int(model.encoder[0].in_features),  # Convert to Python int
            'encoding_dim': int(self.config['encoding_dim']),  # Convert to Python int
            'threshold': float(metrics['threshold'])  # Convert to Python float
        }, model_path)
    
        # Save scaler
        scaler_path = version_dir / 'scaler.pkl'
        joblib.dump(scaler, scaler_path)
    
        # Save metrics - convert numpy types to Python types
        metrics_path = version_dir / 'metrics.json'
        metrics_safe = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_safe[key] = float(value)
            elif isinstance(value, np.ndarray):
                metrics_safe[key] = value.tolist()
            else:
                metrics_safe[key] = value
    
        with open(metrics_path, 'w') as f:
            json.dump(metrics_safe, f, indent=2)
    
        # Update "latest" symlink
        latest_link = self.model_dir / 'latest'
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(version_dir.name)
    
        logger.info(f"✓ Model saved to {version_dir}")
        logger.info(f"✓ Latest model linked")
    
        return str(version_dir)
    
    def run_full_pipeline(self):
        """
        Run complete retraining pipeline
        """
        logger.info("="*60)
        logger.info("STARTING AUTOMATED RETRAINING PIPELINE")
        logger.info("="*60)
        
        # 1. Load data
        X_normal, X_test, y_test, feature_cols = self.load_and_prepare_data()
        
        # 2. Train model
        model, scaler, training_metrics = self.train_model(X_normal)
        
        # 3. Evaluate model
        eval_metrics = self.evaluate_model(model, scaler, X_test, y_test)
        
        # 4. Save model
        all_metrics = {**training_metrics, **eval_metrics, 
                      'timestamp': datetime.now().isoformat()}
        model_path = self.save_model(model, scaler, all_metrics)
        
        logger.info("="*60)
        logger.info("RETRAINING PIPELINE COMPLETE")
        logger.info("="*60)
        
        return model_path, all_metrics


class RetrainingScheduler:
    """
    Schedule and trigger retraining based on conditions
    """
    
    def __init__(self, pipeline: RetrainingPipeline):
        self.pipeline = pipeline
        self.last_training = None
        self.training_history = []
    
    def should_retrain(self, drift_result: Dict, performance_result: Dict) -> bool:
        """
        Determine if retraining should be triggered
        """
        # Condition 1: Significant drift
        if drift_result.get('drift_detected', False):
            drift_share = drift_result.get('drift_share', 0)
            if drift_share > 0.3:  # More than 30% features drifted
                logger.info(f"Retraining triggered: High drift ({drift_share:.2%})")
                return True
        
        # Condition 2: Performance degradation
        if performance_result.get('degradation_detected', False):
            f1_drop = performance_result.get('f1_drop', 0)
            if f1_drop > 0.1:  # F1 dropped by more than 10%
                logger.info(f"Retraining triggered: Performance degradation (F1 drop: {f1_drop:.3f})")
                return True
        
        # Condition 3: Scheduled retraining (e.g., weekly)
        if self.last_training:
            days_since_training = (datetime.now() - self.last_training).days
            if days_since_training >= 7:
                logger.info(f"Retraining triggered: Scheduled ({days_since_training} days)")
                return True
        
        return False
    
    def trigger_retraining(self):
        """Execute retraining"""
        logger.info("🔄 Starting retraining...")
        
        try:
            model_path, metrics = self.pipeline.run_full_pipeline()
            
            self.last_training = datetime.now()
            self.training_history.append({
                'timestamp': self.last_training.isoformat(),
                'model_path': model_path,
                'metrics': metrics
            })
            
            logger.info(f"✓ Retraining successful: {model_path}")
            return True, model_path
        
        except Exception as e:
            logger.error(f"✗ Retraining failed: {e}")
            return False, None


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RetrainingPipeline()
    
    # Run retraining
    model_path, metrics = pipeline.run_full_pipeline()
    
    print(f"\nModel saved to: {model_path}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")