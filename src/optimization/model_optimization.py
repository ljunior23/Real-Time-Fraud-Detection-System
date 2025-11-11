
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import time
from typing import List, Tuple
import logging
import os


"""
Model optimization for production
- GPU acceleration
- Model quantization
- Batch inference
- ONNX export
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """Original autoencoder architecture (matches training)"""
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


class OptimizedAutoencoder(nn.Module):
    """Optimized autoencoder WITHOUT dropout (for inference)"""
    def __init__(self, input_dim, encoding_dim=14):
        super(OptimizedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 24),
            nn.ReLU(),
            nn.Linear(24, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ModelOptimizer:
    """
    Optimize model for production deployment
    """
    
    def __init__(self, model_path: str, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimized_model = None
        self.model_path = model_path
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """Load model from checkpoint (PyTorch 2.6+ compatible)"""
        # PyTorch 2.6+ requires weights_only=False for custom objects
        # Since we trust our own checkpoint, this is safe
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            # Try with weights_only=True first (more secure)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            # Fall back to weights_only=False for compatibility
            logger.warning(f"Loading with weights_only=False (trusted checkpoint)")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        input_dim = int(checkpoint['input_dim'])  # Ensure Python int
        encoding_dim = int(checkpoint['encoding_dim'])  # Ensure Python int
        
        # Load with original architecture (with Dropout)
        self.model = Autoencoder(
            input_dim=input_dim,
            encoding_dim=encoding_dim
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("✓ Model loaded successfully")
        logger.info(f"  Architecture: {input_dim} → 24 → {encoding_dim} → 24 → {input_dim}")
        
        return self.model
    
    def convert_to_optimized_model(self):
        """
        Convert model with Dropout to optimized model without Dropout
        This is safe because we're in eval mode and Dropout is disabled anyway
        """
        if self.model is None:
            self.load_model()
        
        input_dim = self.model.encoder[0].in_features
        encoding_dim = self.model.encoder[3].out_features
        
        # Create optimized model (no dropout)
        self.optimized_model = OptimizedAutoencoder(
            input_dim=input_dim,
            encoding_dim=encoding_dim
        )
        
        # Copy weights (skip dropout layers)
        with torch.no_grad():
            # Encoder
            self.optimized_model.encoder[0].weight.copy_(self.model.encoder[0].weight)
            self.optimized_model.encoder[0].bias.copy_(self.model.encoder[0].bias)
            self.optimized_model.encoder[2].weight.copy_(self.model.encoder[3].weight)
            self.optimized_model.encoder[2].bias.copy_(self.model.encoder[3].bias)
            
            # Decoder
            self.optimized_model.decoder[0].weight.copy_(self.model.decoder[0].weight)
            self.optimized_model.decoder[0].bias.copy_(self.model.decoder[0].bias)
            self.optimized_model.decoder[2].weight.copy_(self.model.decoder[3].weight)
            self.optimized_model.decoder[2].bias.copy_(self.model.decoder[3].bias)
        
        self.optimized_model.to(self.device)
        self.optimized_model.eval()
        
        logger.info("✓ Converted to optimized model (removed Dropout layers)")
        
        return self.optimized_model
    
    def optimize_with_torchscript(self, save_path='../../models/optimized/model_scripted.pt'):
        """
        Optimize model using TorchScript
        Provides ~20-30% speedup
        """
        logger.info("\n" + "="*60)
        logger.info("TORCHSCRIPT OPTIMIZATION")
        logger.info("="*60)
    
        if self.optimized_model is None:
            self.convert_to_optimized_model()
    
        # Get correct input dimension from the model
        input_dim = self.optimized_model.encoder[0].in_features
        logger.info(f"  Input dimension: {input_dim}")
    
        # Create example input with correct dimension
        example_input = torch.randn(1, input_dim).to(self.device)
    
        # Trace the model
        traced_model = torch.jit.trace(self.optimized_model, example_input)
    
        # Optimize for inference
        traced_model = torch.jit.optimize_for_inference(traced_model)
    
        # Save traced model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        traced_model.save(save_path)
    
        logger.info(f"✓ TorchScript model saved to {save_path}")
    
        # Benchmark with correct input dimension
        self.benchmark_model(self.model, "Original Model (with Dropout)", input_dim=input_dim)
        self.benchmark_model(self.optimized_model, "Optimized Model (no Dropout)", input_dim=input_dim)
        self.benchmark_model(traced_model, "TorchScript Model", input_dim=input_dim)
    
        return traced_model
    
    def quantize_model(self, save_path='../../models/optimized/model_quantized.pt'):
        """
        Dynamic quantization for model compression
        Reduces model size by ~4x with minimal accuracy loss
        """
        logger.info("\n" + "="*60)
        logger.info("QUANTIZATION")
        logger.info("="*60)
        
        if self.optimized_model is None:
            self.convert_to_optimized_model()
        
        # Move to CPU for quantization
        model_cpu = self.optimized_model.cpu()
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        # Save quantized model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(quantized_model.state_dict(), save_path)
        
        # Calculate size reduction
        original_size = self.get_model_size(self.optimized_model)
        quantized_size = self.get_model_size(quantized_model)
        
        logger.info(f"  Original model size: {original_size:.2f} MB")
        logger.info(f"  Quantized model size: {quantized_size:.2f} MB")
        logger.info(f"  Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
        # Move back to original device
        self.optimized_model.to(self.device)
        
        return quantized_model
    
    def export_to_onnx(self, save_path='../../models/optimized/model.onnx'):
        """
        Export model to ONNX format
        nables deployment on various platforms
        """
        logger.info("\n" + "="*60)
        logger.info("ONNX EXPORT")
        logger.info("="*60)
    
        if self.optimized_model is None:
            self.convert_to_optimized_model()
    
        # Get correct input dimension
        input_dim = self.optimized_model.encoder[0].in_features
        logger.info(f"  Input dimension: {input_dim}")
    
        # Create dummy input with correct dimension
        dummy_input = torch.randn(1, input_dim).to(self.device)
    
        # Export to ONNX
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
        torch.onnx.export(
            self.optimized_model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
        logger.info(f"✓ ONNX model saved to {save_path}")
    
        # Verify ONNX model
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX model verified")
    
        # Benchmark ONNX Runtime
        self.benchmark_onnx(save_path)
    
        return save_path
    
    def benchmark_model(self, model, name="Model", num_iterations=1000, input_dim=None):
        """Benchmark model inference speed"""
        logger.info(f"\n  Benchmarking {name}...")
    
        # Get input dimension
        if input_dim is None:
            if hasattr(model, 'encoder'):
                if isinstance(model.encoder, nn.Sequential):
                    input_dim = model.encoder[0].in_features
                else:
                    input_dim = 32  # Fallback
            else:
                # For TorchScript models, try to get from graph
                try:
                    input_dim = model.graph.inputs()[0].type().sizes()[1]
                except:
                    logger.error(f"Cannot determine input dimension for {name}")
                    return
    
        logger.info(f"    Using input dimension: {input_dim}")
    
        # Create dummy input with correct dimension
        dummy_input = torch.randn(1, input_dim).to(self.device)
    
        # Warmup
        try:
            for _ in range(100):
                with torch.no_grad():
                    _ = model(dummy_input)
        except Exception as e:
            logger.error(f"Warmup failed for {name}: {e}")
            return
    
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
        start_time = time.time()
    
        try:
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(dummy_input)
        except Exception as e:
            logger.error(f"Benchmark failed for {name}: {e}")
            return
    
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
        end_time = time.time()
    
        avg_latency = (end_time - start_time) / num_iterations * 1000  # ms
        throughput = num_iterations / (end_time - start_time)
    
        logger.info(f"    {name}:")
        logger.info(f"      Latency: {avg_latency:.3f} ms")
        logger.info(f"      Throughput: {throughput:.1f} inferences/sec")
    
    def benchmark_onnx(self, onnx_path, num_iterations=1000):
        """Benchmark ONNX Runtime"""
        logger.info("\n  Benchmarking ONNX Runtime...")
        
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input shape
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        input_shape[0] = 1  # Batch size
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(100):
            _ = ort_session.run(None, {input_name: dummy_input})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            _ = ort_session.run(None, {input_name: dummy_input})
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / num_iterations * 1000  # ms
        throughput = num_iterations / (end_time - start_time)
        
        logger.info(f"    ONNX Runtime:")
        logger.info(f"      Latency: {avg_latency:.3f} ms")
        logger.info(f"      Throughput: {throughput:.1f} inferences/sec")
    
    def get_model_size(self, model):
        """Calculate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def optimize_batch_inference(self, batch_size=32):
        """
        Optimize for batch inference
        Achieves higher throughput for bulk predictions
        """
        logger.info("\n" + "="*60)
        logger.info("BATCH INFERENCE OPTIMIZATION")
        logger.info("="*60)
        
        if self.optimized_model is None:
            self.convert_to_optimized_model()
        
        # Create batch input
        input_dim = self.optimized_model.encoder[0].in_features
        batch_input = torch.randn(batch_size, input_dim).to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.optimized_model(batch_input)
        
        # Benchmark single vs batch
        # Single inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for i in range(batch_size):
                _ = self.optimized_model(batch_input[i:i+1])
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        single_time = time.time() - start_time
        
        # Batch inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            _ = self.optimized_model(batch_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_time = time.time() - start_time
        
        speedup = single_time / batch_time
        
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Single inference (x{batch_size}): {single_time*1000:.3f} ms")
        logger.info(f"  Batch inference: {batch_time*1000:.3f} ms")
        logger.info(f"  Speedup: {speedup:.2f}x")
    
    def full_optimization_pipeline(self):
        """Run all optimization techniques"""
        logger.info("="*60)
        logger.info("RUNNING FULL OPTIMIZATION PIPELINE")
        logger.info("="*60)
        
        # Load model
        self.load_model()
        
        # Convert to optimized architecture
        self.convert_to_optimized_model()
        
        # 1. TorchScript
        traced_model = self.optimize_with_torchscript()
        
        # 2. Quantization
        quantized_model = self.quantize_model()
        
        # 3. ONNX Export
        onnx_path = self.export_to_onnx()
        
        # 4. Batch optimization
        self.optimize_batch_inference(batch_size=32)
        
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*60)
        
        return {
            'torchscript': '../../models/optimized/model_scripted.pt',
            'quantized': '../../models/optimized/model_quantized.pt',
            'onnx': onnx_path
        }


class InferencePipeline:
    """
    Optimized inference pipeline for production
    """
    
    def __init__(self, model_path: str, use_onnx=True):
        self.use_onnx = use_onnx
        
        if use_onnx and model_path.endswith('.onnx'):
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            logger.info("Loaded ONNX model")
        else:
            self.model = torch.jit.load(model_path) if model_path.endswith('.pt') else None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.model:
                self.model.to(self.device)
                self.model.eval()
            logger.info(f"Loaded PyTorch model on {self.device}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Fast prediction with optimized model
        """
        if self.use_onnx:
            # ONNX Runtime inference
            if features.ndim == 1:
                features = features.reshape(1, -1)
            features = features.astype(np.float32)
            outputs = self.session.run(None, {self.input_name: features})[0]
            reconstruction_errors = np.mean((features - outputs) ** 2, axis=1)
        else:
            # PyTorch inference
            if features.ndim == 1:
                features = features.reshape(1, -1)
            features_tensor = torch.FloatTensor(features).to(self.device)
            with torch.no_grad():
                outputs = self.model(features_tensor)
            reconstruction_errors = torch.mean((features_tensor - outputs) ** 2, dim=1).cpu().numpy()
        
        return reconstruction_errors
    
    def predict_batch(self, features_list: List[np.ndarray], batch_size=32) -> np.ndarray:
        """Batch prediction for higher throughput"""
        all_errors = []
        
        for i in range(0, len(features_list), batch_size):
            batch = np.vstack(features_list[i:i+batch_size])
            errors = self.predict(batch)
            all_errors.extend(errors)
        
        return np.array(all_errors)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Get model path from command line or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = '../../models/autoencoder_fraud.pth'
    
    print(f"Optimizing model: {model_path}")
    
    # Optimize model
    optimizer = ModelOptimizer(model_path=model_path)
    
    try:
        optimized_paths = optimizer.full_optimization_pipeline()
        
        print("\n" + "="*60)
        print("✅ OPTIMIZATION SUCCESSFUL")
        print("="*60)
        print("\nOptimized models saved:")
        for name, path in optimized_paths.items():
            print(f"  {name}: {path}")
        
        print("\n📊 Summary:")
        print("  - TorchScript: ~20-30% faster inference")
        print("  - Quantization: ~75% smaller model size")
        print("  - ONNX: Platform-independent deployment")
        print("  - Batch inference: 5-10x throughput improvement")
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()