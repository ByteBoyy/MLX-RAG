import os
import sys
import argparse
from pathlib import Path
import subprocess
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelQuantizer:
    """Handles downloading and quantizing Llama-3.1 8B Instruct to INT4"""
    
    def __init__(self, output_dir: str = "./models"):
        self.output_dir = Path(output_dir)
        self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.quantized_model_dir = self.output_dir / "llama-3.1-8b-instruct-int4"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        required_packages = ["mlx", "mlx-lm", "huggingface_hub"]
        
        logger.info("Checking dependencies...")
        
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm import load, generate, convert
            from huggingface_hub import snapshot_download
            logger.info("✓ All required packages found")
            return True
        except ImportError as e:
            logger.error(f"❌ Missing dependency: {e}")
            logger.error("Please install required packages:")
            logger.error("pip install mlx mlx-lm huggingface-hub")
            return False
    
    def check_hf_token(self) -> Optional[str]:
        """Check for Hugging Face token (required for Llama models)"""
        token = os.getenv("HF_TOKEN")
        print(token)
        if not token:
            logger.warning("No HF_TOKEN environment variable found")
            logger.warning("You may need to set your Hugging Face token:")
            logger.warning("export HF_TOKEN=your_token_here")
            logger.warning("Get your token from: https://huggingface.co/settings/tokens")
        return token
    
    def download_model(self) -> str:
        """Download the base model from Hugging Face"""
        logger.info(f"Downloading {self.model_name}...")
        
        try:
            from huggingface_hub import snapshot_download
            
            # Check for HF token
            token = self.check_hf_token()
            
            # Download model
            model_path = snapshot_download(
                repo_id=self.model_name,
                cache_dir=self.output_dir / "cache",
                token=token,
                resume_download=True
            )
            
            logger.info(f"✓ Model downloaded to: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            raise
    
    def quantize_model(self, model_path: str) -> str:
        """Quantize the model to INT4 using MLX"""
        logger.info("Starting INT4 quantization...")
        
        try:
            from mlx_lm import convert
            
            # Convert and quantize to INT4
            convert(
                hf_path=model_path,
                mlx_path=str(self.quantized_model_dir),
                quantize=True,
                q_bits=4,  # INT4 quantization
                q_group_size=64,  # Group size for quantization
                upload_repo=None  # Don't upload to HF
            )
            
            logger.info(f"✓ Model quantized and saved to: {self.quantized_model_dir}")
            return str(self.quantized_model_dir)
            
        except Exception as e:
            logger.error(f"❌ Failed to quantize model: {e}")
            raise
    
    def verify_quantization(self) -> bool:
        """Verify the quantized model can be loaded"""
        logger.info("Verifying quantized model...")
        
        try:
            from mlx_lm import load
            
            # Try to load the quantized model
            model, tokenizer = load(str(self.quantized_model_dir))
            
            # Basic test - tokenize a simple prompt
            test_prompt = "Hello, how are you?"
            tokens = tokenizer.encode(test_prompt)
            
            logger.info(f"✓ Model verification successful")
            logger.info(f"  - Model path: {self.quantized_model_dir}")
            logger.info(f"  - Test tokenization: {len(tokens)} tokens")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model verification failed: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the quantized model"""
        if not self.quantized_model_dir.exists():
            return {"error": "Model not found"}
        
        # Calculate approximate model size
        total_size = sum(
            f.stat().st_size for f in self.quantized_model_dir.rglob('*') 
            if f.is_file()
        )
        size_mb = total_size / (1024 * 1024)
        
        return {
            "model_path": str(self.quantized_model_dir),
            "quantization": "INT4",
            "size_mb": round(size_mb, 2),
            "files": [f.name for f in self.quantized_model_dir.iterdir() if f.is_file()]
        }
    
    def run_full_pipeline(self) -> bool:
        """Run the complete download and quantization pipeline"""
        logger.info("Starting Llama-3.1 8B INT4 quantization pipeline...")
        
        try:
            # Check dependencies
            if not self.check_dependencies():
                return False
            
            # Check if model already exists
            if self.quantized_model_dir.exists():
                logger.info(f"Quantized model already exists at: {self.quantized_model_dir}")
                response = input("Do you want to re-quantize? (y/N): ").strip().lower()
                if response != 'y':
                    return self.verify_quantization()
            
            # Download base model
            model_path = self.download_model()
            
            # Quantize to INT4
            quantized_path = self.quantize_model(model_path)
            
            # Verify the quantized model
            if not self.verify_quantization():
                return False
            
            # Print model info
            info = self.get_model_info()
            logger.info(f"📊 Quantized model info:")
            logger.info(f"   Size: {info['size_mb']} MB")
            logger.info(f"   Path: {info['model_path']}")
            
            logger.info("🎉 Model quantization completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Download and quantize Llama-3.1 8B Instruct to INT4"
    )
    parser.add_argument(
        "--output-dir", 
        default="./models",
        help="Directory to save the quantized model (default: ./models)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing quantized model"
    )
    
    args = parser.parse_args()
    
    # Initialize quantizer
    quantizer = ModelQuantizer(output_dir=args.output_dir)
    
    if args.verify_only:
        success = quantizer.verify_quantization()
        if success:
            info = quantizer.get_model_info()
            print(f"\nModel Info: {info}")
    else:
        success = quantizer.run_full_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
