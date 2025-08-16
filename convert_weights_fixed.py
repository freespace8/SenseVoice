# convert_weights_fixed.py
# 修正的PyTorch到MLX权重转换脚本

import torch
import numpy as np
from safetensors.torch import save_file
import os
import argparse
from typing import Dict, Any, Tuple
from funasr import AutoModel

# 导入模型定义
from model_mlx import SenseVoiceMLX

print("Fixed Weight Conversion Script Initialized.")


def load_pytorch_model(model_path: str = "iic/SenseVoiceSmall"):
    """加载PyTorch模型并获取权重"""
    print(f"Loading PyTorch model from '{model_path}'...")
    
    pytorch_model = AutoModel(
        model=model_path,
        trust_remote_code=True,
        remote_code="./model.py",
        vad_model=None,
        device="cpu",
    )
    
    actual_model = pytorch_model.model
    actual_model.eval()
    
    print(f"PyTorch model loaded successfully.")
    return actual_model


def create_parameter_mapping(pytorch_model, mlx_model):
    """创建PyTorch到MLX的参数映射"""
    print("\n" + "="*60)
    print("CREATING PARAMETER MAPPING")
    print("="*60)
    
    # 获取PyTorch参数
    pytorch_params = pytorch_model.state_dict()
    pytorch_param_names = set(pytorch_params.keys())
    print(f"PyTorch parameters: {len(pytorch_param_names)}")
    
    # 获取MLX参数（使用递归方式）
    def get_mlx_parameters(param_dict, prefix=""):
        """递归获取MLX模型的所有参数"""
        params = {}
        
        for name, value in param_dict.items():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(value, dict):
                # 这是一个嵌套字典，递归处理
                sub_params = get_mlx_parameters(value, full_name)
                params.update(sub_params)
            elif hasattr(value, 'shape') and hasattr(value, 'dtype'):
                # 这是一个MLX数组/参数
                params[full_name] = value
                
        return params
    
    mlx_params = get_mlx_parameters(mlx_model.parameters())
    mlx_param_names = set(mlx_params.keys())
    print(f"MLX parameters: {len(mlx_param_names)}")
    
    # 创建映射字典
    mapping = {}
    
    # 直接映射策略
    for pt_name in pytorch_param_names:
        # 将PyTorch命名转换为MLX命名
        mlx_name = convert_pytorch_to_mlx_name(pt_name)
        
        if mlx_name in mlx_param_names:
            # 检查形状是否匹配
            pt_shape = pytorch_params[pt_name].shape
            mlx_shape = mlx_params[mlx_name].shape
            
            if pt_shape == mlx_shape:
                mapping[pt_name] = mlx_name
                print(f"✅ {pt_name} -> {mlx_name} {pt_shape}")
            else:
                print(f"❌ Shape mismatch: {pt_name} {pt_shape} vs {mlx_name} {mlx_shape}")
        else:
            print(f"❓ No MLX match for: {pt_name}")
    
    print(f"\n📊 Mapping Summary:")
    print(f"   Successfully mapped: {len(mapping)} / {len(pytorch_param_names)}")
    
    return mapping, pytorch_params, mlx_params


def convert_pytorch_to_mlx_name(pytorch_name: str) -> str:
    """将PyTorch参数名转换为MLX参数名"""
    # 处理编码器层命名的差异：
    # PyTorch: encoder.encoders0.0.xxx -> MLX: encoder.encoders0_0.xxx
    # PyTorch: encoder.encoders.5.xxx -> MLX: encoder.encoders_5.xxx
    # PyTorch: encoder.tp_encoders.10.xxx -> MLX: encoder.tp_encoders_10.xxx
    
    mlx_name = pytorch_name
    
    # 替换编码器层命名模式
    import re
    
    # 模式: encoders0.数字 -> encoders0_数字
    mlx_name = re.sub(r'encoders0\.(\d+)', r'encoders0_\1', mlx_name)
    
    # 模式: encoders.数字 -> encoders_数字
    mlx_name = re.sub(r'(?<!encoders0)\.encoders\.(\d+)', lambda m: f".encoders_{m.group(1)}", mlx_name)
    mlx_name = re.sub(r'^encoders\.(\d+)', r'encoders_\1', mlx_name)
    
    # 模式: tp_encoders.数字 -> tp_encoders_数字  
    mlx_name = re.sub(r'tp_encoders\.(\d+)', r'tp_encoders_\1', mlx_name)
    
    # 处理FSMN块的命名差异
    # PyTorch: fsmn_block -> MLX: fsmn_conv
    mlx_name = mlx_name.replace('fsmn_block', 'fsmn_conv')
    
    return mlx_name


def convert_weights(pytorch_model, mlx_model, output_path: str):
    """执行权重转换"""
    print(f"\n🚀 Starting weight conversion to '{output_path}'...")
    
    # 创建参数映射
    mapping, pytorch_params, mlx_params = create_parameter_mapping(pytorch_model, mlx_model)
    
    if len(mapping) == 0:
        print("❌ No parameter mappings found! Conversion failed.")
        return False
    
    # 转换权重
    converted_weights = {}
    
    print(f"\n🔄 Converting weights...")
    for pt_name, mlx_name in mapping.items():
        try:
            # 获取PyTorch权重
            pt_weight = pytorch_params[pt_name]
            
            # 转换为numpy，然后保存
            if pt_weight.dtype == torch.bfloat16:
                # 转换bfloat16到float32
                np_weight = pt_weight.to(torch.float32).detach().cpu().numpy()
            else:
                np_weight = pt_weight.detach().cpu().numpy()
            
            # 处理FSMN权重的形状转换
            if 'fsmn_conv.weight' in mlx_name and np_weight.shape == (512, 1, 11):
                # PyTorch: (512, 1, 11) -> MLX: (512, 11, 1)
                np_weight = np_weight.transpose(0, 2, 1)
                print(f"   ✨ FSMN shape converted: (512, 1, 11) -> {np_weight.shape}")
            
            # 使用MLX名称作为key保存
            converted_weights[mlx_name] = torch.from_numpy(np_weight)
            
        except Exception as e:
            print(f"❌ Error converting {pt_name}: {e}")
            return False
    
    # 保存转换后的权重
    try:
        save_file(converted_weights, output_path)
        print(f"✅ Weights successfully saved to '{output_path}'")
        print(f"   Converted {len(converted_weights)} parameters")
        return True
    except Exception as e:
        print(f"❌ Error saving weights: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch SenseVoice weights to MLX format')
    parser.add_argument('--model', default='iic/SenseVoiceSmall', help='PyTorch model path')
    parser.add_argument('--output', default='sensevoice_mlx_fixed.safetensors', help='Output path for MLX weights')
    
    args = parser.parse_args()
    
    print("🚀 Starting PyTorch to MLX weight conversion...")
    print(f"   Model path: {args.model}")
    print(f"   Output path: {args.output}")
    
    try:
        # 加载模型
        pytorch_model = load_pytorch_model(args.model)
        mlx_model = SenseVoiceMLX()
        
        # 执行转换
        success = convert_weights(pytorch_model, mlx_model, args.output)
        
        if success:
            print(f"\n🎉 Conversion completed successfully!")
            print(f"   Output file: {args.output}")
        else:
            print(f"\n❌ Conversion failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Conversion script failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())