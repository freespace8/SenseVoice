"""
Configuration management for model verification scripts.
Centralized configuration to improve maintainability and flexibility.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json


@dataclass
class VerificationConfig:
    """Configuration class for model verification settings."""
    
    # File paths
    pytorch_output_dir: str = "verification_output_pytorch_internal"
    mlx_weights_path: str = "sensevoice_mlx.safetensors"
    
    # Verification settings
    tolerance: float = 1e-5
    strict_shape_check: bool = True
    enable_statistical_analysis: bool = True
    
    # Output settings
    verbose: bool = True
    save_verification_report: bool = True
    report_output_dir: str = "verification_reports"
    
    # Model settings
    synthetic_input_shape: tuple = (1, 123, 80)  # (batch, seq_len, feature_dim)
    
    @classmethod
    def from_env(cls) -> 'VerificationConfig':
        """Load configuration from environment variables."""
        return cls(
            pytorch_output_dir=os.getenv('PYTORCH_OUTPUT_DIR', cls.pytorch_output_dir),
            mlx_weights_path=os.getenv('MLX_WEIGHTS_PATH', cls.mlx_weights_path),
            tolerance=float(os.getenv('VERIFICATION_TOLERANCE', cls.tolerance)),
            strict_shape_check=os.getenv('STRICT_SHAPE_CHECK', 'true').lower() == 'true',
            verbose=os.getenv('VERBOSE', 'true').lower() == 'true',
        )
    
    @classmethod
    def from_json(cls, config_path: str) -> 'VerificationConfig':
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            print(f"Warning: Config file {config_path} not found, using defaults")
            return cls()
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Only use verification-related keys
        verification_keys = {
            'pytorch_output_dir', 'mlx_weights_path', 'tolerance', 
            'strict_shape_check', 'verbose', 'save_verification_report'
        }
        
        filtered_config = {k: v for k, v in config_data.items() if k in verification_keys}
        return cls(**filtered_config)
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if self.tolerance <= 0:
            print(f"Error: tolerance must be positive, got {self.tolerance}")
            return False
        
        if not os.path.exists(self.pytorch_output_dir):
            print(f"Error: PyTorch output directory does not exist: {self.pytorch_output_dir}")
            return False
        
        return True


def get_verification_config() -> VerificationConfig:
    """
    Get verification configuration with priority order:
    1. JSON config file (if exists)
    2. Environment variables
    3. Default values
    """
    config_path = "verification_config.json"
    
    if os.path.exists(config_path):
        return VerificationConfig.from_json(config_path)
    else:
        return VerificationConfig.from_env()