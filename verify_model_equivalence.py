# verify_model_equivalence.py

import mlx.core as mx
import numpy as np
import os

from model_mlx import SenseVoiceMLX

print("Model Equivalence Verification Script Initialized.")

# --- Artifact Loading ---

# 1. 加载配置和验证环境
try:
    from verification_config import get_verification_config
    config = get_verification_config()
    
    if not config.validate():
        print("❌ Configuration validation failed!")
        exit(1)
        
except ImportError:
    print("Warning: Using fallback configuration")
    config = type('Config', (), {
        'pytorch_output_dir': "verification_output_pytorch_internal",
        'mlx_weights_path': "sensevoice_mlx.safetensors", 
        'tolerance': 1e-5,
        'verbose': True
    })()

# 2. 安全的文件存在性检查
def safe_file_check(file_path: str, description: str) -> bool:
    """Safely check if file exists and is readable."""
    try:
        if not os.path.exists(file_path):
            print(f"❌ {description} not found: {file_path}")
            return False
        
        if not os.access(file_path, os.R_OK):
            print(f"❌ {description} not readable: {file_path}")
            return False
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"❌ {description} is empty: {file_path}")
            return False
            
        if config.verbose:
            print(f"✅ {description} verified: {file_path} ({file_size} bytes)")
        return True
        
    except (OSError, IOError) as e:
        print(f"❌ Error accessing {description}: {e}")
        return False

print(f"Loading PyTorch verification data from '{config.pytorch_output_dir}'...")

# 检查必需的文件是否存在且可读
required_files = ['ctc_logits.npy', 'encoder_output.npy']
missing_files = []

for file_name in required_files:
    file_path = os.path.join(config.pytorch_output_dir, file_name)
    if not safe_file_check(file_path, f"PyTorch baseline file '{file_name}'"):
        missing_files.append(file_name)

if missing_files:
    print(f"❌ Missing required files: {missing_files}")
    print("Please run verify_pytorch_internal.py first to generate PyTorch baseline data.")
    exit(1)

# 3. 安全加载PyTorch基准数据
def safe_load_numpy(file_path: str, description: str) -> np.ndarray:
    """Safely load numpy array with error handling."""
    try:
        data = np.load(file_path)
        if config.verbose:
            print(f"✅ Loaded {description}: {data.shape}, dtype: {data.dtype}")
        return data
    except (IOError, ValueError) as e:
        print(f"❌ Failed to load {description} from {file_path}: {e}")
        exit(1)

pt_ctc_logits = safe_load_numpy(
    os.path.join(config.pytorch_output_dir, 'ctc_logits.npy'),
    "PyTorch CTC logits"
)

pt_encoder_output = safe_load_numpy(
    os.path.join(config.pytorch_output_dir, 'encoder_output.npy'), 
    "PyTorch encoder output"
)

# 检查是否有输入特征文件
fbank_file = os.path.join(config.pytorch_output_dir, 'fbank_features.npy')
if safe_file_check(fbank_file, "Input features file"):
    pt_input_features = safe_load_numpy(fbank_file, "PyTorch input features")
else:
    print("⚠️  Warning: fbank_features.npy not found. Generating synthetic input...")
    # 使用配置中的合成输入形状，或从编码器输出推断
    if hasattr(config, 'synthetic_input_shape'):
        shape = config.synthetic_input_shape
    else:
        # 从encoder_output的形状推断输入特征的大小
        batch_size, seq_len, _ = pt_encoder_output.shape
        shape = (batch_size, seq_len, 80)
    
    np.random.seed(42)  # 确保可重复性
    pt_input_features = np.random.randn(*shape).astype(np.float32)
    print(f"📊 Generated synthetic input features: {pt_input_features.shape}")

# 4. 实例化MLX模型
print(f"\n🔧 Initializing MLX model...")
try:
    mlx_model = SenseVoiceMLX()
    print("✅ MLX model instantiated successfully")
except Exception as e:
    print(f"❌ Failed to instantiate MLX model: {e}")
    exit(1)

# 5. 安全加载MLX权重
if not safe_file_check(config.mlx_weights_path, "MLX weights file"):
    print("Please run the weight conversion script first to create MLX weights.")
    print(f"Expected file: {config.mlx_weights_path}")
    exit(1)

print(f"🔄 Loading MLX weights from '{config.mlx_weights_path}'...")
try:
    mlx_model.load_weights(config.mlx_weights_path)
    print("✅ MLX weights loaded successfully")
except Exception as e:
    print(f"❌ Error loading MLX weights: {e}")
    print("Please check if the weights file is valid and compatible.")
    exit(1)

# 将MLX模型设置为评估模式
mlx_model.eval()

print("All artifacts loaded successfully.")

# --- MLX Model Execution ---

print("\nExecuting MLX model in debug mode...")
# 将NumPy输入转换为MLX张量
mlx_input_features = mx.array(pt_input_features)

# 使用 debug=True 参数调用 forward 方法
try:
    mlx_outputs = mlx_model(mlx_input_features, debug=True)
    print("MLX forward pass complete. Outputs captured.")
    print(f"MLX debug outputs keys: {list(mlx_outputs.keys())}")
    
    # 打印各个输出的形状
    for key, value in mlx_outputs.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
            
except Exception as e:
    print(f"Error during MLX model execution: {e}")
    print("This could indicate an issue with the model definition or weights.")
    exit(1)

# --- Verification and Assertion ---

def verify_tensors(name: str, pt_tensor: np.ndarray, mlx_tensor: mx.array, atol: float = 1e-5) -> bool:
    """
    Robust tensor comparison function with comprehensive validation.
    
    Args:
        name: Descriptive name for the tensor being verified
        pt_tensor: PyTorch tensor as NumPy array
        mlx_tensor: MLX tensor to compare against
        atol: Absolute tolerance for numerical comparison
        
    Returns:
        bool: True if tensors match within tolerance, False otherwise
    """
    print(f"\nVerifying: {name} ...")

    # 1. Shape validation with detailed error reporting
    mlx_shape = mlx_tensor.shape
    if pt_tensor.shape != mlx_shape:
        print(f"  ❌ Shape mismatch! PyTorch: {pt_tensor.shape}, MLX: {mlx_shape}")
        return False
    print(f"  ✅ Shape check PASSED: {pt_tensor.shape}")

    # 2. Safe tensor conversion with error handling
    try:
        mlx_tensor_np = np.array(mlx_tensor)
    except Exception as e:
        print(f"  ❌ Failed to convert MLX tensor to NumPy: {e}")
        return False
    
    # 3. Data integrity validation
    if np.any(np.isnan(pt_tensor)) or np.any(np.isnan(mlx_tensor_np)):
        print(f"  ❌ NaN values detected!")
        return False
    
    if np.any(np.isinf(pt_tensor)) or np.any(np.isinf(mlx_tensor_np)):
        print(f"  ❌ Infinite values detected!")
        return False
    
    # 4. Statistical analysis for debugging insight
    abs_diff = np.abs(pt_tensor - mlx_tensor_np)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    std_abs_diff = np.std(abs_diff)
    
    print(f"  📊 Max absolute difference: {max_abs_diff:.8f}")
    print(f"  📊 Mean absolute difference: {mean_abs_diff:.8f}")
    print(f"  📊 Std absolute difference: {std_abs_diff:.8f}")
    
    # 5. Tolerance-based validation with context
    if not np.allclose(pt_tensor, mlx_tensor_np, atol=atol):
        print(f"  ❌ Numerical mismatch! (tolerance={atol})")
        print(f"     Max difference: {max_abs_diff}")
        print(f"     Relative error: {max_abs_diff / (np.max(np.abs(pt_tensor)) + 1e-8):.8f}")
        return False
    
    print(f"  ✅ Numerical check PASSED (atol={atol})")
    return True

print("\n" + "="*50)
print("Starting Final Verification")
print("="*50)

verification_results = []

try:
    # 验证编码器输出
    if "encoder_out" in mlx_outputs:
        result = verify_tensors("encoder_output", pt_encoder_output, mlx_outputs["encoder_out"], config.tolerance)
        verification_results.append(("encoder_output", result))
    else:
        print("⚠️  Warning: encoder_out not found in MLX debug outputs")
    
    # 验证最终输出 (必须验证)
    if "ctc_logits" in mlx_outputs:
        result = verify_tensors("ctc_logits", pt_ctc_logits, mlx_outputs["ctc_logits"], config.tolerance)
        verification_results.append(("ctc_logits", result))
    else:
        print("❌ Error: ctc_logits not found in MLX outputs!")
        verification_results.append(("ctc_logits", False))

    # 验证结果总结
    all_passed = all(result for _, result in verification_results)
    
    print("\n" + "="*50)
    print("Verification Summary:")
    print("="*50)
    
    for test_name, result in verification_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    if all_passed:
        print("\n" + "🎉"*20)
        print("✅ SUCCESS: All checks passed!")
        print("The MLX model is numerically equivalent to the PyTorch model.")
        print("🎉"*20)
    else:
        print("\n" + "❌"*20)
        print("❌ FAILURE: Some verifications failed!")
        print("The MLX model does NOT match the PyTorch model.")
        print("Please review the model implementation and weight conversion.")
        print("❌"*20)
        exit(1)

except Exception as e:
    print("\n" + "💥"*20)
    print("❌ CRITICAL ERROR during verification!")
    print(f"Error: {e}")
    print("💥"*20)
    exit(1)