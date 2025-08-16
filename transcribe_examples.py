#!/usr/bin/env python3
"""
使用 MLX 实现的 SenseVoice 模型批量转录音频文件
"""

import os
import time
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np

# 导入 MLX 模型和统一前端
from model_mlx import SenseVoiceMLX
from utils.frontend_mlx import create_frontend_mlx


def extract_features_with_unified_frontend(audio_path, frontend):
    """
    使用统一前端提取特征
    
    Args:
        audio_path: 音频文件路径
        frontend: UnifiedFrontend 实例
    
    Returns:
        MLX 格式的特征 (1, T, 560)
    """
    # 读取音频文件
    waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    
    # 使用统一前端提取特征，直接返回 MLX 格式
    features_mlx = frontend(waveform, output_format="mlx")
    
    return features_mlx


def load_model(model_path):
    """加载 MLX 模型权重"""
    print(f"📥 加载模型权重: {model_path}")
    
    # 初始化模型 - 使用 560 维输入（LFR处理后的特征维度）
    model = SenseVoiceMLX(
        input_size=560,  # 560维 LFR特征 (80 * 7)
        vocab_size=25055,
        encoder_conf={
            "output_size": 512,
            "attention_heads": 4,
            "linear_units": 2048,
            "num_blocks": 50,
            "tp_blocks": 20,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "attention_dropout_rate": 0.0,
            "normalize_before": True,
            "kernel_size": 11,
            "sanm_shift": 0,
        }
    )
    
    # 加载权重
    weights = mx.load(model_path)
    model.load_weights(weights)
    
    print("✅ 模型加载成功")
    return model


def decode_tokens(token_ids, token_list_path="tokens.txt"):
    """
    将 token ID 解码为文本
    
    Args:
        token_ids: Token ID 数组
        token_list_path: Token 列表文件路径
    
    Returns:
        解码后的文本
    """
    # 如果 token 列表文件不存在，使用简单的 ID 映射
    if not os.path.exists(token_list_path):
        print(f"⚠️  Token 列表文件不存在: {token_list_path}")
        print("   使用简单 ID 显示")
        return f"[Tokens: {token_ids.tolist()}]"
    
    # 加载 token 列表
    with open(token_list_path, 'r', encoding='utf-8') as f:
        tokens = [line.strip() for line in f]
    
    # 解码
    text_tokens = []
    for tid in token_ids:
        if 0 <= tid < len(tokens):
            token = tokens[tid]
            # 跳过特殊标记
            if not token.startswith('<') and not token.startswith('['):
                text_tokens.append(token)
    
    # 合并文本
    text = ''.join(text_tokens)
    
    # 清理文本
    text = text.replace('▁', ' ')  # 替换空格标记
    text = text.strip()
    
    return text


def transcribe_audio(model, frontend, audio_path, language="auto"):
    """
    转录单个音频文件
    
    Args:
        model: SenseVoiceMLX 模型
        frontend: UnifiedFrontend 实例
        audio_path: 音频文件路径
        language: 语言设置
    
    Returns:
        转录文本
    """
    print(f"\n🎤 处理音频: {audio_path}")
    print(f"   语言设置: {language}")
    
    # 提取特征
    print("   提取音频特征...")
    features_mx = extract_features_with_unified_frontend(audio_path, frontend)
    print(f"   特征形状: {features_mx.shape}")
    
    # 获取序列长度
    speech_lengths = mx.array([features_mx.shape[1]], dtype=mx.int32)
    
    # 推理
    print("   运行模型推理...")
    start_time = time.time()
    
    # MLX 不需要 no_grad 上下文，默认就是推理模式
    outputs = model(features_mx, speech_lengths)
    ctc_logits = outputs['ctc_logits']
    
    inference_time = time.time() - start_time
    print(f"   推理时间: {inference_time:.2f}秒")
    
    # CTC 解码 - 简单的 argmax 解码
    token_ids = mx.argmax(ctc_logits[0], axis=-1)  # (T,)
    
    # 去除重复和空白标记
    token_ids_np = np.array(token_ids)
    
    # 去除连续重复
    decoded_tokens = []
    prev_token = -1
    for token in token_ids_np:
        if token != prev_token and token != 0:  # 0 是 blank token
            decoded_tokens.append(token)
        prev_token = token
    
    # 解码为文本
    if len(decoded_tokens) > 0:
        text = decode_tokens(np.array(decoded_tokens))
    else:
        text = "[无法识别]"
    
    return text


def main():
    """主函数"""
    print("=" * 60)
    print("SenseVoice MLX 音频转录工具")
    print("=" * 60)
    
    # 模型路径
    model_path = "/Users/taylor/Documents/code/SenseVoice/model/model_mlx.safetensors"
    
    # 音频文件夹
    examples_dir = "/Users/taylor/Documents/code/SenseVoice/examples"
    
    # 检查路径
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(examples_dir):
        print(f"❌ 示例文件夹不存在: {examples_dir}")
        return
    
    # 加载模型
    model = load_model(model_path)
    
    # 创建统一前端
    print("\n📐 初始化统一前端...")
    # 检查 CMVN 文件是否存在
    cmvn_file = "/Users/taylor/.cache/modelscope/hub/models/iic/SenseVoiceSmall/am.mvn"
    if not os.path.exists(cmvn_file):
        print(f"   ⚠️  CMVN 文件不存在: {cmvn_file}")
        print("   使用无 CMVN 的前端")
        frontend = create_frontend_mlx(cmvn_file=None)
    else:
        print(f"   ✅ 使用 CMVN 文件: {cmvn_file}")
        frontend = create_frontend_mlx(cmvn_file=cmvn_file)
    
    # 获取所有音频文件
    audio_files = sorted([f for f in os.listdir(examples_dir) if f.endswith('.mp3')])
    
    if not audio_files:
        print(f"❌ 在 {examples_dir} 中没有找到音频文件")
        return
    
    print(f"\n📂 找到 {len(audio_files)} 个音频文件")
    print("-" * 60)
    
    # 处理每个音频文件
    results = []
    total_start = time.time()
    
    for audio_file in audio_files:
        audio_path = os.path.join(examples_dir, audio_file)
        
        # 从文件名推断语言
        if audio_file.startswith('zh'):
            language = 'zh'
        elif audio_file.startswith('en'):
            language = 'en'
        elif audio_file.startswith('ja'):
            language = 'ja'
        elif audio_file.startswith('ko'):
            language = 'ko'
        elif audio_file.startswith('yue'):
            language = 'yue'
        else:
            language = 'auto'
        
        # 转录音频
        try:
            text = transcribe_audio(model, frontend, audio_path, language)
            results.append({
                'file': audio_file,
                'language': language,
                'text': text
            })
            print(f"   ✅ 转录结果: {text}")
        except Exception as e:
            print(f"   ❌ 转录失败: {e}")
            results.append({
                'file': audio_file,
                'language': language,
                'text': f"[错误: {str(e)}]"
            })
    
    total_time = time.time() - total_start
    
    # 显示汇总结果
    print("\n" + "=" * 60)
    print("转录结果汇总")
    print("=" * 60)
    
    for result in results:
        print(f"\n📄 文件: {result['file']}")
        print(f"   语言: {result['language']}")
        print(f"   文本: {result['text']}")
    
    print(f"\n⏱️  总用时: {total_time:.2f}秒")
    print(f"   平均每个文件: {total_time/len(audio_files):.2f}秒")
    
    # 保存结果到文件
    output_file = "transcription_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("SenseVoice MLX 转录结果\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            f.write(f"文件: {result['file']}\n")
            f.write(f"语言: {result['language']}\n")
            f.write(f"文本: {result['text']}\n")
            f.write("-" * 40 + "\n\n")
        
        f.write(f"总用时: {total_time:.2f}秒\n")
        f.write(f"平均每个文件: {total_time/len(audio_files):.2f}秒\n")
    
    print(f"\n💾 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()