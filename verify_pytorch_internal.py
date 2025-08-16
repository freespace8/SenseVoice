#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# verify_pytorch_internal.py
# 捕获内部数据的PyTorch验证脚本

import os
import torch
import numpy as np
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

print("Internal PyTorch Verification Script Initialized.")

# --- Determinism Setup ---
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("Determinism enforced.")

# --- Model Loading ---
model_path = "iic/SenseVoiceSmall"
device = "cpu"

print(f"Loading SenseVoice model from '{model_path}' on {device}...")

model_wrapper = AutoModel(
    model=model_path,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model=None,
    device=device,
)

print("Model loaded successfully.")

# --- Input Data ---
input_audio_path = 'verification_data/en.mp3'
output_dir = "verification_output_pytorch_internal"
os.makedirs(output_dir, exist_ok=True)

print(f"Processing: {input_audio_path}")
print(f"Output directory: {output_dir}")

# --- Hook定义 - 捕获中间数据 ---
captured_data = {}

def capture_fbank_hook(module, input, output):
    """捕获Fbank特征"""
    if isinstance(output, tuple):
        fbank_features = output[0] if output[0] is not None else output[1]
    else:
        fbank_features = output
    
    if isinstance(fbank_features, torch.Tensor):
        captured_data['fbank_features'] = fbank_features.detach().cpu().numpy()
        print(f"Captured Fbank features: {fbank_features.shape}")

def capture_encoder_hook(module, input, output):
    """捕获编码器输出"""
    if isinstance(output, tuple):
        encoder_out = output[0]
    else:
        encoder_out = output
    
    if isinstance(encoder_out, torch.Tensor):
        captured_data['encoder_output'] = encoder_out.detach().cpu().numpy()
        print(f"Captured encoder output: {encoder_out.shape}")

def capture_ctc_hook(module, input, output):
    """捕获CTC logits"""
    if isinstance(output, torch.Tensor):
        captured_data['ctc_logits'] = output.detach().cpu().numpy()
        print(f"Captured CTC logits: {output.shape}")

# --- 注册hooks ---
def register_hooks(model):
    """递归注册hooks到模型的所有子模块"""
    hooks = []
    
    for name, module in model.named_modules():
        # 根据模块名称或类型注册不同的hook
        if 'frontend' in name.lower() or 'fbank' in name.lower():
            hook = module.register_forward_hook(capture_fbank_hook)
            hooks.append(hook)
            print(f"Registered Fbank hook on: {name}")
        elif 'encoder' in name.lower() and 'layer' not in name.lower():
            hook = module.register_forward_hook(capture_encoder_hook)
            hooks.append(hook)
            print(f"Registered encoder hook on: {name}")
        elif 'ctc' in name.lower() and 'lo' in name.lower():
            hook = module.register_forward_hook(capture_ctc_hook)
            hooks.append(hook)
            print(f"Registered CTC hook on: {name}")
    
    return hooks

# 获取实际模型并注册hooks
actual_model = model_wrapper.model
print(f"Model type: {type(actual_model)}")
print("Available modules:")
for name, module in actual_model.named_modules():
    print(f"  {name}: {type(module)}")

hooks = register_hooks(actual_model)

# --- 执行推理 ---
print("\nStarting inference with data capture...")

with torch.no_grad():
    # 使用generate方法执行推理，hooks会自动捕获数据
    result = model_wrapper.generate(
        input=input_audio_path,
        cache={},
        language="auto",
        use_itn=False,
        batch_size_s=60,
        merge_vad=False,
    )

# 清理hooks
for hook in hooks:
    hook.remove()

print("Inference completed.")

# --- 保存捕获的数据 ---
print("\nSaving captured internal data...")

for data_name, data_array in captured_data.items():
    file_path = os.path.join(output_dir, f'{data_name}.npy')
    np.save(file_path, data_array)
    print(f"Saved {data_name}: {data_array.shape} -> {file_path}")

# 保存文本结果
if result and len(result) > 0:
    raw_text = result[0]["text"]
    processed_text = rich_transcription_postprocess(raw_text)
    
    # 保存文本输出
    with open(os.path.join(output_dir, 'raw_text.txt'), 'w', encoding='utf-8') as f:
        f.write(raw_text + '\n')
    
    with open(os.path.join(output_dir, 'processed_text.txt'), 'w', encoding='utf-8') as f:
        f.write(processed_text + '\n')
    
    with open(os.path.join(output_dir, 'full_result.txt'), 'w', encoding='utf-8') as f:
        f.write(str(result) + '\n')
    
    print(f"Raw text: {raw_text}")
    print(f"Processed text: {processed_text}")

# 保存捕获数据的摘要
summary_path = os.path.join(output_dir, 'capture_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("Captured Internal Data Summary\n")
    f.write("=" * 40 + "\n")
    f.write(f"Audio file: {input_audio_path}\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Device: {device}\n\n")
    
    for data_name, data_array in captured_data.items():
        f.write(f"{data_name}:\n")
        f.write(f"  Shape: {data_array.shape}\n")
        f.write(f"  Dtype: {data_array.dtype}\n")
        f.write(f"  Min: {data_array.min():.6f}\n")
        f.write(f"  Max: {data_array.max():.6f}\n")
        f.write(f"  Mean: {data_array.mean():.6f}\n\n")

print(f"\nInternal PyTorch Verification completed!")
print(f"Captured {len(captured_data)} internal data arrays.")
print(f"All artifacts saved to: {output_dir}")