#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# verify_pytorch.py
# PyTorch验证脚本 - 为MLX迁移建立确定性基准

import os
import torch
import numpy as np
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

print("PyTorch Verification Script Initialized.")

# --- Determinism Setup ---
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("Determinism enforced.")

# --- Component Loading ---
# 使用固定的、已知的模型配置
model_path = "iic/SenseVoiceSmall"
device = "cpu"  # 为了最大程度保证跨设备的一致性，我们首先在CPU上生成基准

print(f"Loading SenseVoice model from '{model_path}' on {device}...")

# 加载SenseVoice模型 - 使用FunASR的AutoModel
model = AutoModel(
    model=model_path,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model=None,  # 不使用VAD以确保确定性
    device=device,
)

print(f"SenseVoice model loaded and set to eval mode on {device}.")

# --- Input Data Preparation ---
input_audio_path = 'verification_data/en.mp3'  # 使用英语测试音频作为主要基准
if not os.path.exists(input_audio_path):
    raise FileNotFoundError(f"Verification audio file not found at: {input_audio_path}")

print(f"Using verification audio: {input_audio_path}")
# 注意：我们将直接使用AutoModel的generate方法，它内部会处理音频加载

# --- Core Inference and Output Capture ---
# 创建一个目录来存储我们的基准输出
output_dir = "verification_output_pytorch"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving verification artifacts to: {output_dir}")

print("Starting deterministic inference...")

# 1. 执行端到端推理
# 使用模型的generate方法进行推理
result = model.generate(
    input=input_audio_path,
    cache={},
    language="auto",  # 自动检测语言
    use_itn=False,  # 关闭逆文本规范化以确保确定性
    batch_size_s=60,
    merge_vad=False,  # 关闭VAD合并
)

print(f"Step 1/3: Model inference complete.")

# 2. 获取原始输出
raw_text = result[0]["text"] if result and len(result) > 0 else ""
print(f"Step 2/3: Raw text extracted: {raw_text}")

# 3. 后处理文本
processed_text = rich_transcription_postprocess(raw_text)
print(f"Step 3/3: Text post-processing complete.")

# 保存中间结果
# 保存原始模型输出
raw_output_path = os.path.join(output_dir, 'raw_output.txt')
with open(raw_output_path, 'w', encoding='utf-8') as f:
    f.write(str(result) + '\n')

# 保存原始文本
raw_text_path = os.path.join(output_dir, 'raw_text.txt')
with open(raw_text_path, 'w', encoding='utf-8') as f:
    f.write(raw_text + '\n')

# 保存最终处理后的文本
final_text_output_path = os.path.join(output_dir, 'decoded_text.txt')
with open(final_text_output_path, 'w', encoding='utf-8') as f:
    f.write(processed_text + '\n')

# 保存音频元信息
audio_info_path = os.path.join(output_dir, 'audio_info.txt')
with open(audio_info_path, 'w', encoding='utf-8') as f:
    f.write(f"Audio file: {input_audio_path}\n")
    f.write(f"File size: {os.path.getsize(input_audio_path)} bytes\n")
    f.write(f"Model configuration: {model_path}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Language detection: auto\n")
    f.write(f"ITN disabled: True\n")
    f.write(f"VAD disabled: True\n")

print("-" * 50)
print("VERIFICATION RESULTS:")
print("-" * 50)
print(f"Raw Text: {raw_text}")
print(f"Final Processed Text: {processed_text}")
print("-" * 50)

print(f"Verification artifacts saved to: {output_dir}")
print("Files created:")
print(f"  - raw_output.txt: Complete model output")
print(f"  - raw_text.txt: Raw transcription text")
print(f"  - decoded_text.txt: Final processed text")
print(f"  - audio_info.txt: Audio file metadata")

print("\nPyTorch Verification Script Finished Successfully.")
print("These files constitute the immutable verification baseline for MLX development.")