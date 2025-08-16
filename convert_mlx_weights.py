# filename: convert_weights_fixed.py
#
# A robust, refactored script to convert PyTorch state_dict files 
# to MLX-compatible .safetensors files without funasr dependency.

import torch
import numpy as np
from safetensors.torch import save_file
import argparse
import re
from typing import Dict

def convert_pytorch_to_mlx_name(pytorch_name: str) -> str:
    """
    Converts PyTorch parameter names to the MLX convention used in our model.
    e.g., 'encoder.encoders.0.norm1.weight' -> 'encoder.encoders_0.norm1.weight'
    """
    # This regex handles all encoder layer naming patterns (encoders0, encoders, tp_encoders)
    # It finds a dot followed by digits followed by another dot, and replaces the dots with underscores.
    # e.g., .0. -> _0.
    mlx_name = re.sub(r'\.([0-9]+)\.', r'_\1.', pytorch_name)

    # Handle the FSMN block name difference
    mlx_name = mlx_name.replace('fsmn_block', 'fsmn_conv')

    return mlx_name

def convert_and_save(pytorch_weights_path: str, output_path: str):
    """
    Loads a PyTorch state_dict, converts it, and saves it as an
    MLX-compatible safetensors file.
    """
    print(f"🚀 Loading PyTorch state_dict from: {pytorch_weights_path}")

    # 1. Load the PyTorch state_dict directly from the file.
    # This is the most direct and reliable way to get the raw weights.
    try:
        # Use map_location='cpu' to ensure it loads on any machine.
        state_dict = torch.load(pytorch_weights_path, map_location="cpu")
        print(f"✅ Successfully loaded {len(state_dict)} tensors from PyTorch file.")
    except Exception as e:
        print(f"❌ FATAL: Failed to load PyTorch weights file: {e}")
        return False

    # 2. Iterate through the PyTorch state_dict and create the new MLX weights dict.
    mlx_weights: Dict[str, torch.Tensor] = {}
    print("\n🔄 Converting tensor formats and names...")
    for pt_name, pt_tensor in state_dict.items():
        # Convert the parameter name to the MLX convention.
        mlx_name = convert_pytorch_to_mlx_name(pt_name)

        # Convert to float32 to handle potential issues with bfloat16 or other types.
        # Keep as torch tensor for safetensors.torch.save_file
        tensor_float32 = pt_tensor.float().cpu()
        
        # Handle FSMN weight transpose: (512, 1, 11) -> (512, 11, 1)
        if 'fsmn_conv.weight' in mlx_name and tensor_float32.shape == torch.Size([512, 1, 11]):
            tensor_float32 = tensor_float32.transpose(1, 2)  # Swap dimensions 1 and 2
            print(f"   ✨ Transposed FSMN weight: {pt_name} from (512, 1, 11) to {tuple(tensor_float32.shape)}")

        mlx_weights[mlx_name] = tensor_float32

    print(f"✅ Conversion logic complete. Total tensors to save: {len(mlx_weights)}")

    # 4. Save the new dictionary as a safetensors file.
    print(f"\n💾 Saving converted weights to: {output_path}")
    try:
        # The save_file function from safetensors.torch expects torch tensors
        save_file(mlx_weights, output_path)
        print(f"🎉 Successfully saved MLX weights to '{output_path}'")
        return True
    except Exception as e:
        print(f"❌ FATAL: Failed to save safetensors file: {e}")
        return False

def main():
    from pathlib import Path
    
    # 动态获取用户主目录
    home_dir = Path.home()
    default_input = home_dir / '.cache' / 'modelscope' / 'hub' / 'models' / 'iic' / 'SenseVoiceSmall' / 'model.pt'
    
    parser = argparse.ArgumentParser(
        description='A robust script to convert PyTorch .pt or .bin weight files to MLX-compatible .safetensors.'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        default=str(default_input),
        help='Path to the source PyTorch weights file (e.g., model.pt or pytorch_model.bin)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/model/model_mlx.safetensors',
        help='Output path for the MLX-compatible .safetensors file'
    )

    args = parser.parse_args()

    convert_and_save(args.input, args.output)

if __name__ == '__main__':
    main()