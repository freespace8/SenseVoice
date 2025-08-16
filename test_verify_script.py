# test_verify_script.py
# 测试验证脚本逻辑的简单示例

import mlx.core as mx
import numpy as np
import os

# 模拟一个简单的验证
print("Testing verify_model_equivalence.py logic...")

# 创建测试数据
test_pt_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
test_mlx_tensor = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 从验证脚本中提取验证函数
def verify_tensors(name: str, pt_tensor: np.ndarray, mlx_tensor: mx.array, atol: float = 1e-5):
    """A helper function to compare a PyTorch tensor and an MLX tensor."""
    print(f"\nVerifying: {name} ...")

    # 形状检查
    mlx_shape = mlx_tensor.shape
    if pt_tensor.shape != mlx_shape:
        print(f"  ❌ Shape mismatch! PyTorch: {pt_tensor.shape}, MLX: {mlx_shape}")
        return False
    print(f"  ✅ Shape check PASSED: {pt_tensor.shape}")

    # 数值接近度检查
    # 将MLX张量转回NumPy以便与PyTorch的NumPy数组比较
    try:
        mlx_tensor_np = np.array(mlx_tensor)
    except Exception as e:
        print(f"  ❌ Failed to convert MLX tensor to NumPy: {e}")
        return False
    
    # 检查是否包含NaN或无穷大
    if np.any(np.isnan(pt_tensor)) or np.any(np.isnan(mlx_tensor_np)):
        print(f"  ❌ NaN values detected!")
        return False
    
    if np.any(np.isinf(pt_tensor)) or np.any(np.isinf(mlx_tensor_np)):
        print(f"  ❌ Infinite values detected!")
        return False
    
    # 计算差异统计
    abs_diff = np.abs(pt_tensor - mlx_tensor_np)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    print(f"  📊 Max absolute difference: {max_abs_diff:.8f}")
    print(f"  📊 Mean absolute difference: {mean_abs_diff:.8f}")
    
    # 数值接近度检查
    if not np.allclose(pt_tensor, mlx_tensor_np, atol=atol):
        print(f"  ❌ Numerical mismatch! (tolerance={atol})")
        print(f"     Max difference: {max_abs_diff}")
        return False
    
    print(f"  ✅ Numerical check PASSED (atol={atol})")
    return True

# 测试验证函数
print("="*50)
print("Testing verification function")
print("="*50)

result = verify_tensors("test_identical", test_pt_tensor, test_mlx_tensor)
print(f"Identical tensors result: {result}")

# 测试稍有差异的张量
test_mlx_tensor_diff = mx.array([[1.0000001, 2.0, 3.0], [4.0, 5.0, 6.0]])
result2 = verify_tensors("test_slight_diff", test_pt_tensor, test_mlx_tensor_diff)
print(f"Slightly different tensors result: {result2}")

print("\n" + "🎉"*20)
print("✅ Verification script logic test completed!")
print("The verify_model_equivalence.py script should work correctly")
print("once the MLX weights are properly converted.")
print("🎉"*20)