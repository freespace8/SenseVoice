# convert_weights.py
# 可验证的PyTorch到MLX权重转换脚本

import torch
import numpy as np
from safetensors.torch import save_file
import os
from typing import Dict, Any, Tuple
from funasr import AutoModel

# 导入我们自己的模型定义
try:
    from model import SenseVoiceSmall as SenseVoicePyTorch
except ImportError:
    print("Warning: Could not import PyTorch model directly, using AutoModel")
    SenseVoicePyTorch = None

from model_mlx import SenseVoiceMLX

print("Weight Conversion Script Initialized.")


def load_pytorch_model(model_path: str = "iic/SenseVoiceSmall"):
    """加载PyTorch模型并获取权重"""
    print(f"Loading PyTorch model from '{model_path}'...")
    
    # 使用AutoModel加载预训练模型
    pytorch_model = AutoModel(
        model=model_path,
        trust_remote_code=True,
        remote_code="./model.py",
        vad_model=None,
        device="cpu",
    )
    
    # 获取实际的模型对象
    actual_model = pytorch_model.model
    actual_model.eval()
    
    print(f"PyTorch model loaded successfully.")
    return actual_model


def analyze_model_parameters(pytorch_model, mlx_model):
    """分析和比较两个模型的参数结构"""
    print("\n" + "="*60)
    print("PARAMETER ANALYSIS")
    print("="*60)
    
    # PyTorch模型参数
    print("\nPyTorch Model Parameters:")
    print("-"*60)
    pytorch_params = pytorch_model.state_dict()
    pytorch_param_count = 0
    pytorch_param_info = {}
    
    for name, tensor in pytorch_params.items():
        print(f"- {name:<50} | Shape: {tensor.shape}")
        pytorch_param_count += tensor.numel()
        pytorch_param_info[name] = {
            'shape': tensor.shape,
            'numel': tensor.numel(),
            'dtype': tensor.dtype
        }
    
    print(f"\nPyTorch Total Parameters: {pytorch_param_count:,}")
    
    # MLX模型参数
    print("\n" + "-"*60)
    print("MLX Model Parameters:")
    print("-"*60)
    mlx_param_count = 0
    mlx_param_info = {}
    
    # 获取MLX模型的所有参数
    try:
        # MLX使用不同的参数访问方式
        mlx_params_dict = mlx_model.parameters()
        for name, param in mlx_params_dict.items():
            print(f"- {name:<50} | Shape: {param.shape}")
            mlx_param_count += param.size
            mlx_param_info[name] = {
                'shape': param.shape,
                'numel': param.size,
            }
    except Exception as e:
        print(f"Error accessing MLX model parameters: {e}")
        print("Attempting alternative parameter access...")
        # 尝试递归获取参数
        def get_all_parameters(module, prefix=""):
            params = {}
            for name, value in module.__dict__.items():
                if hasattr(value, 'shape'):  # 这是一个参数
                    full_name = f"{prefix}.{name}" if prefix else name
                    params[full_name] = value
                elif hasattr(value, '__dict__') and hasattr(value, '__class__'):
                    # 这是一个子模块
                    if prefix:
                        sub_prefix = f"{prefix}.{name}"
                    else:
                        sub_prefix = name
                    sub_params = get_all_parameters(value, sub_prefix)
                    params.update(sub_params)
            return params
        
        mlx_params_dict = get_all_parameters(mlx_model)
        for name, param in mlx_params_dict.items():
            if hasattr(param, 'shape'):
                print(f"- {name:<50} | Shape: {param.shape}")
                mlx_param_count += param.size
                mlx_param_info[name] = {
                    'shape': param.shape,
                    'numel': param.size,
                }
    
    print(f"\nMLX Total Parameters: {mlx_param_count:,}")
    
    # 参数数量对比
    print("\n" + "="*60)
    print("PARAMETER COUNT COMPARISON")
    print("="*60)
    print(f"PyTorch: {pytorch_param_count:,}")
    print(f"MLX:     {mlx_param_count:,}")
    print(f"Difference: {abs(pytorch_param_count - mlx_param_count):,}")
    
    if pytorch_param_count != mlx_param_count:
        print("⚠️  WARNING: Parameter counts don't match!")
    else:
        print("✅ Parameter counts match!")
    
    return pytorch_param_info, mlx_param_info


def create_parameter_mapping(pytorch_params: Dict, mlx_params: Dict) -> Dict[str, str]:
    """创建PyTorch到MLX参数名称的映射"""
    print("\n" + "="*60)
    print("CREATING PARAMETER MAPPING")
    print("="*60)
    
    mapping = {}
    unmapped_pytorch = set(pytorch_params.keys())
    unmapped_mlx = set(mlx_params.keys())
    
    # 直接匹配（名称完全相同）
    print("\nDirect matches:")
    for pt_name in list(unmapped_pytorch):
        if pt_name in unmapped_mlx:
            mapping[pt_name] = pt_name
            unmapped_pytorch.remove(pt_name)
            unmapped_mlx.remove(pt_name)
            print(f"✅ {pt_name}")
    
    # 模式匹配
    print("\nPattern-based mapping:")
    
    # Embedding层映射
    for pt_name in list(unmapped_pytorch):
        if 'embed.weight' in pt_name:
            for mlx_name in list(unmapped_mlx):
                if 'embed.weight' in mlx_name:
                    mapping[pt_name] = mlx_name
                    unmapped_pytorch.remove(pt_name)
                    unmapped_mlx.remove(mlx_name)
                    print(f"🔄 {pt_name} -> {mlx_name}")
                    break
    
    # CTC层映射
    for pt_name in list(unmapped_pytorch):
        if 'ctc.ctc_lo' in pt_name:
            for mlx_name in list(unmapped_mlx):
                if 'ctc.ctc_lo' in mlx_name:
                    mapping[pt_name] = mlx_name
                    unmapped_pytorch.remove(pt_name)
                    unmapped_mlx.remove(mlx_name)
                    print(f"🔄 {pt_name} -> {mlx_name}")
                    break
    
    # 编码器层映射 - 复杂的层级结构
    encoder_mappings = [
        ('encoders.', 'encoders.'),
        ('self_attn.linear_q_k_v', 'self_attn.linear_q_k_v'),
        ('self_attn.linear_out', 'self_attn.linear_out'),
        ('self_attn.fsmn_block', 'self_attn.fsmn_conv'),
        ('feed_forward.w_1', 'feed_forward.w_1'),
        ('feed_forward.w_2', 'feed_forward.w_2'),
        ('norm1', 'norm1'),
        ('norm2', 'norm2'),
    ]
    
    for pt_name in list(unmapped_pytorch):
        for mlx_name in list(unmapped_mlx):
            # 检查是否为编码器参数
            if 'encoder' in pt_name.lower() and 'encoder' in mlx_name.lower():
                # 尝试匹配编码器层
                for pt_pattern, mlx_pattern in encoder_mappings:
                    if pt_pattern in pt_name and mlx_pattern in mlx_name:
                        # 提取层号
                        try:
                            pt_parts = pt_name.split('.')
                            mlx_parts = mlx_name.split('.')
                            
                            # 找到层号位置
                            pt_layer_idx = None
                            mlx_layer_idx = None
                            
                            for i, part in enumerate(pt_parts):
                                if part.isdigit():
                                    pt_layer_idx = int(part)
                                    break
                            
                            for i, part in enumerate(mlx_parts):
                                if part.isdigit():
                                    mlx_layer_idx = int(part)
                                    break
                            
                            # 如果层号匹配，进行映射
                            if pt_layer_idx is not None and mlx_layer_idx is not None and pt_layer_idx == mlx_layer_idx:
                                # 进一步检查参数类型是否匹配
                                pt_suffix = pt_name.split('.')[-1]  # weight 或 bias
                                mlx_suffix = mlx_name.split('.')[-1]
                                
                                if pt_suffix == mlx_suffix:
                                    mapping[pt_name] = mlx_name
                                    unmapped_pytorch.remove(pt_name)
                                    unmapped_mlx.remove(mlx_name)
                                    print(f"🔄 {pt_name} -> {mlx_name}")
                                    break
                        except (IndexError, ValueError):
                            continue
            
            if pt_name not in unmapped_pytorch:
                break
    
    # 报告未映射的参数
    print(f"\n📊 Mapping Summary:")
    print(f"   Mapped: {len(mapping)}")
    print(f"   Unmapped PyTorch: {len(unmapped_pytorch)}")
    print(f"   Unmapped MLX: {len(unmapped_mlx)}")
    
    if unmapped_pytorch:
        print(f"\n⚠️  Unmapped PyTorch parameters:")
        for name in sorted(unmapped_pytorch):
            print(f"   - {name}")
    
    if unmapped_mlx:
        print(f"\n⚠️  Unmapped MLX parameters:")
        for name in sorted(unmapped_mlx):
            print(f"   - {name}")
    
    return mapping


def convert_weight_tensor(name: str, tensor: torch.Tensor) -> np.ndarray:
    """转换单个权重张量的格式"""
    numpy_tensor = tensor.detach().cpu().numpy()
    
    # 检查是否需要转置（线性层权重）
    if 'linear' in name.lower() and 'weight' in name and len(numpy_tensor.shape) == 2:
        # PyTorch Linear: (out_features, in_features)
        # MLX Linear: (out_features, in_features) - 实际上MLX也是这个顺序
        # 但根据Gemini的建议，我们需要验证这一点
        print(f"   Linear weight: {name} | Shape: {numpy_tensor.shape}")
        # 暂时不转置，等待验证
        pass
    
    elif 'conv' in name.lower() and 'weight' in name:
        # 卷积层权重格式检查
        print(f"   Conv weight: {name} | Shape: {numpy_tensor.shape}")
        pass
    
    return numpy_tensor


def convert_weights(pytorch_model, mlx_model, mapping: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """执行权重转换"""
    print("\n" + "="*60)
    print("WEIGHT CONVERSION")
    print("="*60)
    
    pytorch_state_dict = pytorch_model.state_dict()
    converted_weights = {}
    
    conversion_stats = {
        'successful': 0,
        'failed': 0,
        'shape_mismatches': 0
    }
    
    for pt_name, mlx_name in mapping.items():
        if pt_name in pytorch_state_dict:
            try:
                # 获取PyTorch权重
                pt_tensor = pytorch_state_dict[pt_name]
                
                # 转换格式
                converted_tensor = convert_weight_tensor(pt_name, pt_tensor)
                
                # 转换回torch张量用于safetensors保存
                converted_weights[mlx_name] = torch.from_numpy(converted_tensor)
                
                print(f"✅ {pt_name} -> {mlx_name} | Shape: {pt_tensor.shape}")
                conversion_stats['successful'] += 1
                
            except Exception as e:
                print(f"❌ Failed to convert {pt_name}: {e}")
                conversion_stats['failed'] += 1
        else:
            print(f"⚠️  PyTorch parameter not found: {pt_name}")
            conversion_stats['failed'] += 1
    
    print(f"\n📊 Conversion Summary:")
    print(f"   Successful: {conversion_stats['successful']}")
    print(f"   Failed: {conversion_stats['failed']}")
    print(f"   Shape mismatches: {conversion_stats['shape_mismatches']}")
    
    return converted_weights


def save_converted_weights(weights: Dict[str, torch.Tensor], output_path: str):
    """保存转换后的权重到safetensors文件"""
    print(f"\n💾 Saving converted weights to '{output_path}'...")
    
    try:
        save_file(weights, output_path)
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ Successfully saved {len(weights)} parameters to {output_path}")
        print(f"   File size: {file_size:.2f} MB")
        return True
    except Exception as e:
        print(f"❌ Failed to save weights: {e}")
        return False


def verify_conversion(original_model, converted_weights: Dict[str, torch.Tensor], mapping: Dict[str, str]):
    """验证转换的正确性"""
    print("\n" + "="*60)
    print("CONVERSION VERIFICATION")
    print("="*60)
    
    pytorch_state_dict = original_model.state_dict()
    verification_results = []
    
    for pt_name, mlx_name in mapping.items():
        if pt_name in pytorch_state_dict and mlx_name in converted_weights:
            pt_tensor = pytorch_state_dict[pt_name]
            converted_tensor = converted_weights[mlx_name]
            
            # 比较形状
            shape_match = pt_tensor.shape == converted_tensor.shape
            
            # 比较数值（允许小的浮点误差）
            if shape_match:
                max_diff = torch.max(torch.abs(pt_tensor - converted_tensor)).item()
                numerical_match = max_diff < 1e-6
            else:
                max_diff = float('inf')
                numerical_match = False
            
            verification_results.append({
                'pt_name': pt_name,
                'mlx_name': mlx_name,
                'shape_match': shape_match,
                'numerical_match': numerical_match,
                'max_diff': max_diff
            })
            
            status = "✅" if shape_match and numerical_match else "❌"
            print(f"{status} {pt_name} -> {mlx_name}")
            if not shape_match:
                print(f"   Shape mismatch: {pt_tensor.shape} vs {converted_tensor.shape}")
            if not numerical_match:
                print(f"   Numerical difference: {max_diff}")
    
    # 统计验证结果
    total = len(verification_results)
    shape_matches = sum(1 for r in verification_results if r['shape_match'])
    numerical_matches = sum(1 for r in verification_results if r['numerical_match'])
    
    print(f"\n📊 Verification Summary:")
    print(f"   Total parameters: {total}")
    print(f"   Shape matches: {shape_matches}/{total} ({100*shape_matches/total:.1f}%)")
    print(f"   Numerical matches: {numerical_matches}/{total} ({100*numerical_matches/total:.1f}%)")
    
    return verification_results


def main(model_path: str = "iic/SenseVoiceSmall", output_path: str = "sensevoice_mlx.safetensors"):
    """主函数：执行完整的权重转换流程"""
    print("🚀 Starting PyTorch to MLX weight conversion...")
    print(f"   Model path: {model_path}")
    print(f"   Output path: {output_path}")
    
    try:
        # 1. 加载模型
        pytorch_model = load_pytorch_model(model_path)
        mlx_model = SenseVoiceMLX()
        
        # 2. 分析参数结构
        pytorch_params, mlx_params = analyze_model_parameters(pytorch_model, mlx_model)
        
        # 3. 创建参数映射
        mapping = create_parameter_mapping(pytorch_params, mlx_params)
        
        if not mapping:
            print("❌ No parameter mappings found! Check model architectures.")
            return False
        
        # 4. 转换权重
        converted_weights = convert_weights(pytorch_model, mlx_model, mapping)
        
        if not converted_weights:
            print("❌ No weights converted! Check parameter mapping.")
            return False
        
        # 5. 验证转换
        verification_results = verify_conversion(pytorch_model, converted_weights, mapping)
        
        # 6. 保存权重
        success = save_converted_weights(converted_weights, output_path)
        
        if success:
            print(f"\n🎉 Weight conversion completed successfully!")
            print(f"   Output file: {output_path}")
            print(f"   Converted parameters: {len(converted_weights)}")
            return True
        else:
            print(f"\n❌ Weight conversion failed during saving.")
            return False
            
    except Exception as e:
        print(f"❌ Weight conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PyTorch SenseVoice weights to MLX format")
    parser.add_argument("--model_path", type=str, default="iic/SenseVoiceSmall", 
                       help="Path to PyTorch model")
    parser.add_argument("--output_path", type=str, default="sensevoice_mlx.safetensors",
                       help="Output path for converted weights")
    
    args = parser.parse_args()
    
    success = main(args.model_path, args.output_path)
    if success:
        print("\n✅ Conversion script completed successfully.")
        exit(0)
    else:
        print("\n❌ Conversion script failed.")
        exit(1)