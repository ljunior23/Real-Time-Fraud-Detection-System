from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import joblib
from typing import Dict, List
import time
from datetime import datetime
import sys
sys.path.append('..')

from features.timeseries_features import TimeSeriesFeatureEngine

# Initialize FastAPI
app = FastAPI(title="Fraud Detection API", version="1.0")

# Load model and scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, encoding_dim=14):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 24),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(24, encoding_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, 24),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(24, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Global variables
model = None
scaler = None
feature_engine = None
threshold = None
original_feature_names = None

def load_model_artifacts():
    """Load model, scaler, and feature engine"""
    global model, scaler, feature_engine, threshold, original_feature_names
    
    print("Loading model artifacts...")
    
    try:
        # Load model
        model_paths = [
            '/app/models/autoencoder_fraud.pth',  # Docker path
            '../../models/autoencoder_fraud.pth',  # Local path
            'models/autoencoder_fraud.pth'
        ]
        
        checkpoint = None
        for path in model_paths:
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                print(f"✓ Loaded model from: {path}")
                break
            except FileNotFoundError:
                continue
        
        if checkpoint is None:
            raise FileNotFoundError("Could not find autoencoder_fraud.pth in any location")
        
        input_dim = checkpoint['input_dim']
        model = Autoencoder(input_dim=input_dim, encoding_dim=checkpoint['encoding_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        threshold = checkpoint['threshold']
        
        # Load scaler
        scaler_paths = [
            '/app/models/scaler.pkl',      # Docker path
            '../../models/scaler.pkl',     # Local path
            'models/scaler.pkl'
        ]
        
        scaler = None
        for path in scaler_paths:
            try:
                scaler = joblib.load(path)
                print(f"✓ Loaded scaler from: {path}")
                break
            except FileNotFoundError:
                continue
        
        if scaler is None:
            raise FileNotFoundError("Could not find scaler.pkl in any location")
        
        # Initialize feature engine (if available)
        try:
            feature_engine = TimeSeriesFeatureEngine(windows=[60, 300, 3600])
            print("✓ Feature engine initialized")
        except:
            feature_engine = None
            print("⚠ Feature engine not available")
        
        # Original feature names (from creditcard.csv)
        original_feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour', 'Log_Amount']
        
        print(f"✓ Model loaded successfully on {device}")
        print(f"✓ Original Threshold: {threshold:.6f}")
        
        # Threshold adjustment for precision/recall tuning
        threshold_multiplier = 1.5
        threshold = threshold * threshold_multiplier
        print(f"✓ Adjusted Threshold: {threshold:.6f} (multiplier: {threshold_multiplier})")
        
    except FileNotFoundError as e:
        print(f"✗ Model files not found: {e}")
        print("Please ensure model files are in /app/models/ (Docker) or models/ (local)")
        raise
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

# Load on startup
@app.on_event("startup")
async def startup_event():
    load_model_artifacts()

# Request/Response models
class Transaction(BaseModel):
    transaction_id: str
    timestamp: str
    features: Dict[str, float]
    amount: float

class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    reconstruction_error: float
    processing_time_ms: float
    timestamp: str
    timeseries_features: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str

# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device),
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    """
    Predict fraud for a single transaction
    Target: <100ms latency
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not transaction.features:
            raise HTTPException(status_code=400, detail="No features provided")
        
        # Extract time-series features (skip for now to avoid NaN)
        tx_dict = {
            'timestamp': transaction.timestamp,
            'Amount': transaction.amount,
            **transaction.features
        }
        
        # Use empty time-series features for now
        ts_features = {}
        
        # Prepare feature vector
        feature_vector = []
        for name in original_feature_names:
            value = transaction.features.get(name, 0)
            if value is None or not np.isfinite(value):
                value = 0.0
            feature_vector.append(float(value))
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Check for invalid values
        if not np.all(np.isfinite(feature_vector)):
            print(f"Warning: Invalid values in feature vector for {transaction.transaction_id}")
            # Replace invalid values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        try:
            feature_vector_scaled = scaler.transform(feature_vector)
        except Exception as e:
            print(f"Scaling error: {e}")
            print(f"Feature vector shape: {feature_vector.shape}")
            print(f"Feature vector sample: {feature_vector[0][:5]}")
            raise HTTPException(status_code=500, detail=f"Feature scaling failed: {str(e)}")
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(feature_vector_scaled).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            reconstruction_error = torch.mean((input_tensor - output) ** 2).item()
        
        # Check for invalid reconstruction error
        if not np.isfinite(reconstruction_error):
            print(f"Warning: Invalid reconstruction error for {transaction.transaction_id}")
            reconstruction_error = 0.0
        
        # Prediction
        is_fraud = reconstruction_error > threshold
        fraud_probability = min(reconstruction_error / threshold, 1.0) if threshold > 0 else 0.0
        
        # Ensure all values are JSON-serializable
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return PredictionResponse(
            transaction_id=str(transaction.transaction_id),
            is_fraud=bool(is_fraud),
            fraud_probability=float(fraud_probability) if np.isfinite(fraud_probability) else 0.0,
            reconstruction_error=float(reconstruction_error) if np.isfinite(reconstruction_error) else 0.0,
            processing_time_ms=float(processing_time) if np.isfinite(processing_time) else 0.0,
            timestamp=datetime.now().isoformat(),
            timeseries_features=ts_features
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error processing transaction {transaction.transaction_id}:")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(transactions: List[Transaction]):
    """Batch prediction endpoint"""
    results = []
    for tx in transactions:
        result = await predict_fraud(tx)
        results.append(result)
    return results

@app.get("/metrics")
async def get_metrics():
    """Get model metrics and statistics"""
    return {
        "model_type": "Autoencoder",
        "threshold": float(threshold),
        "device": str(device),
        "feature_count": len(original_feature_names),
        "transactions_processed": len(feature_engine.transaction_history),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)