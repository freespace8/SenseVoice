#!/usr/bin/env python3
"""
SenseVoice 模型对比工具 - PyTorch vs MLX
比较两个模型的推理速度和输出结果
"""

import os
import sys
import time
import torch
import numpy as np
import librosa
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.align import Align

# 设置 CUDA 设备为 CPU (Mac 环境)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_default_tensor_type('torch.FloatTensor')

# 导入 MLX 相关
import mlx.core as mx

# 导入模型
from model import SenseVoiceSmall
from model_mlx import SenseVoiceMLX
from utils.frontend_mlx import create_frontend_mlx

# 导入 FunASR
from funasr import AutoModel


def load_pytorch_model(model_dir="/Users/taylor/.cache/modelscope/hub/models/iic/SenseVoiceSmall"):
    """加载 PyTorch 模型"""
    print("\n📦 加载 PyTorch 模型...")
    start_time = time.time()
    
    # 使用 from_pretrained 方法加载
    try:
        # 强制使用 CPU
        model, kwargs = SenseVoiceSmall.from_pretrained(
            model=model_dir, 
            device="cpu",
            disable_update=True,
            disable_log=True
        )
        model.eval()
        load_time = time.time() - start_time
        print(f"   ✅ PyTorch 模型加载成功 (耗时: {load_time:.2f}秒)")
        return model, kwargs
    except Exception as e:
        print(f"   ❌ PyTorch 模型加载失败: {e}")
        return None, None


def load_mlx_model(model_path="/Users/taylor/Documents/code/SenseVoice/model/model_mlx.safetensors"):
    """加载 MLX 模型"""
    print("\n📦 加载 MLX 模型...")
    start_time = time.time()
    
    try:
        # 初始化模型
        model = SenseVoiceMLX(
            input_size=560,  # LFR 特征维度
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
        
        load_time = time.time() - start_time
        print(f"   ✅ MLX 模型加载成功 (耗时: {load_time:.2f}秒)")
        return model
    except Exception as e:
        print(f"   ❌ MLX 模型加载失败: {e}")
        return None


def extract_features_for_pytorch(audio_path, frontend):
    """为 PyTorch 模型提取特征"""
    # 读取音频
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # 使用统一前端提取特征
    features = frontend(waveform, output_format="numpy")
    
    # 转换为 PyTorch 张量
    features_pt = torch.from_numpy(features).float()
    
    # 添加 batch 维度
    if len(features_pt.shape) == 2:
        features_pt = features_pt.unsqueeze(0)
    
    return features_pt, features_pt.shape[1]


def extract_features_for_mlx(audio_path, frontend):
    """为 MLX 模型提取特征"""
    # 读取音频
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # 使用统一前端提取特征，直接返回 MLX 格式
    features_mx = frontend(waveform, output_format="mlx")
    
    return features_mx, features_mx.shape[1]


def inference_pytorch(model, kwargs, audio_path, language="auto"):
    """使用 PyTorch 模型进行推理"""
    print(f"\n🔥 PyTorch 推理: {audio_path}")
    
    try:
        # 使用模型的 inference 方法
        start_time = time.time()
        
        res = model.inference(
            data_in=audio_path,
            language=language,
            use_itn=False,
            ban_emo_unk=False,
            **kwargs
        )
        
        inference_time = time.time() - start_time
        
        if res and len(res) > 0:
            text = res[0][0]['text'] if isinstance(res[0], list) else res[0]['text']
            print(f"   ⏱️  推理时间: {inference_time:.3f}秒")
            print(f"   📝 识别结果: {text}")
            return text, inference_time
        else:
            print(f"   ❌ 无识别结果")
            return "[无识别结果]", inference_time
            
    except Exception as e:
        print(f"   ❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return f"[错误: {str(e)}]", 0


def load_tokenizer():
    """加载 tokenizer 用于解码"""
    try:
        # 首先尝试加载 sentencepiece tokenizer
        from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
        model_dir = "/Users/taylor/.cache/modelscope/hub/models/iic/SenseVoiceSmall"
        tokenizer_conf = {"sentencepiece_model": f"{model_dir}/chn_jpn_yue_eng_ko_spectok.bpe.model"}
        tokenizer = SentencepiecesTokenizer(**tokenizer_conf)
        return tokenizer
    except:
        # 如果失败，尝试使用 tokens.json
        try:
            import json
            tokens_file = "/Users/taylor/.cache/modelscope/hub/models/iic/SenseVoiceSmall/tokens.json"
            with open(tokens_file, 'r', encoding='utf-8') as f:
                tokens = json.load(f)
            
            # 创建一个简单的 tokenizer 对象
            class SimpleTokenizer:
                def __init__(self, tokens):
                    self.tokens = tokens
                    self.id2token = {i: token for i, token in enumerate(tokens)}
                
                def decode(self, token_ids, keep_special_tokens=False):
                    """将 token IDs 解码为文本
                    
                    Args:
                        token_ids: Token ID 列表
                        keep_special_tokens: 是否保留特殊标记
                    """
                    text_tokens = []
                    for tid in token_ids:
                        if tid in self.id2token:
                            token = self.id2token[tid]
                            # 根据设置决定是否跳过特殊标记
                            if keep_special_tokens or not (token.startswith('<') and token.endswith('>')):
                                text_tokens.append(token)
                    
                    # 合并文本
                    text = ''.join(text_tokens)
                    # 清理文本
                    text = text.replace('▁', ' ')  # 替换空格标记
                    text = text.strip()
                    return text
            
            return SimpleTokenizer(tokens)
        except:
            return None


def inference_mlx(model, frontend, audio_path, language="auto", tokenizer=None, keep_special_tokens=False):
    """使用 MLX 模型进行推理
    
    Args:
        model: MLX 模型
        frontend: 前端处理器
        audio_path: 音频文件路径
        language: 语言设置
        tokenizer: 分词器
        keep_special_tokens: 是否保留特殊标记（如语言、情感标记）
    """
    print(f"\n⚡ MLX 推理: {audio_path}")
    
    try:
        # 提取特征
        features_mx = extract_features_for_mlx(audio_path, frontend)
        speech_lengths = mx.array([features_mx[0].shape[1]], dtype=mx.int32)
        
        # 推理
        start_time = time.time()
        outputs = model(features_mx[0], speech_lengths)
        ctc_logits = outputs['ctc_logits']
        
        # CTC 解码
        token_ids = mx.argmax(ctc_logits[0], axis=-1)
        
        # 去除重复
        token_ids_np = np.array(token_ids)
        decoded_tokens = []
        prev_token = -1
        for token in token_ids_np:
            if token != prev_token and token != 0:  # 0 是 blank token
                decoded_tokens.append(token)
            prev_token = token
        
        inference_time = time.time() - start_time
        
        # 使用 tokenizer 解码文本
        if tokenizer and len(decoded_tokens) > 0:
            try:
                text = tokenizer.decode(decoded_tokens, keep_special_tokens=keep_special_tokens)
            except:
                text = f"[Tokens: {decoded_tokens[:20]}...]" if len(decoded_tokens) > 20 else f"[Tokens: {decoded_tokens}]"
        else:
            text = f"[Tokens: {decoded_tokens[:20]}...]" if len(decoded_tokens) > 20 else f"[Tokens: {decoded_tokens}]"
        
        print(f"   ⏱️  推理时间: {inference_time:.3f}秒")
        print(f"   📝 识别结果: {text}")
        return text, inference_time, decoded_tokens
        
    except Exception as e:
        print(f"   ❌ 推理失败: {e}")
        return f"[错误: {str(e)}]", 0, []


def calculate_similarity(text1, text2):
    """计算两个文本的相似度"""
    try:
        from difflib import SequenceMatcher
        
        # 清理文本，去除特殊标记
        def clean_text(text):
            # 去除特殊标记
            import re
            text = re.sub(r'<\|[^|]+\|>', '', text)
            text = text.strip()
            return text
        
        clean1 = clean_text(text1)
        clean2 = clean_text(text2)
        
        # 如果其中一个是 token 列表，返回 0
        if "[Tokens:" in text2 or "[错误:" in text1 or "[错误:" in text2:
            return 0.0
        
        # 计算相似度
        similarity = SequenceMatcher(None, clean1, clean2).ratio()
        return similarity * 100  # 转换为百分比
    except:
        return 0.0


def compare_models(audio_files):
    """对比两个模型的性能"""
    print("\n" + "=" * 80)
    print("🔬 SenseVoice 模型对比测试")
    print("=" * 80)
    
    # 加载模型
    pytorch_model, pytorch_kwargs = load_pytorch_model()
    mlx_model = load_mlx_model()
    
    if not pytorch_model or not mlx_model:
        print("❌ 模型加载失败，无法进行对比")
        return
    
    # 创建统一前端
    print("\n📐 初始化统一前端...")
    cmvn_file = "/Users/taylor/.cache/modelscope/hub/models/iic/SenseVoiceSmall/am.mvn"
    frontend = create_frontend_mlx(cmvn_file=cmvn_file if os.path.exists(cmvn_file) else None)
    
    # 加载 tokenizer
    print("📖 加载 Tokenizer...")
    tokenizer = load_tokenizer()
    if tokenizer:
        print("   ✅ Tokenizer 加载成功")
    else:
        print("   ⚠️  Tokenizer 加载失败，MLX 将只输出 Token IDs")
    
    # 对比结果
    results = []
    
    for audio_file in audio_files:
        print("\n" + "-" * 80)
        print(f"📄 处理文件: {os.path.basename(audio_file)}")
        
        # 从文件名推断语言
        filename = os.path.basename(audio_file)
        if filename.startswith('zh'):
            language = 'zh'
        elif filename.startswith('en'):
            language = 'en'
        elif filename.startswith('ja'):
            language = 'ja'
        elif filename.startswith('ko'):
            language = 'ko'
        elif filename.startswith('yue'):
            language = 'yue'
        else:
            language = 'auto'
        
        print(f"   🌐 语言: {language}")
        
        # PyTorch 推理
        pytorch_text, pytorch_time = inference_pytorch(
            pytorch_model, pytorch_kwargs, audio_file, language
        )
        
        # MLX 推理（保留特殊标记以匹配 PyTorch 输出格式）
        mlx_text, mlx_time, mlx_tokens = inference_mlx(
            mlx_model, frontend, audio_file, language, tokenizer, keep_special_tokens=True
        )
        
        # 计算加速比
        if pytorch_time > 0 and mlx_time > 0:
            speedup = pytorch_time / mlx_time
            print(f"\n   ⚡ 加速比: {speedup:.2f}x")
        else:
            speedup = 0
        
        # 计算相似度
        similarity = calculate_similarity(pytorch_text, mlx_text)
        print(f"   📊 文本相似度: {similarity:.1f}%")
        
        results.append({
            'file': os.path.basename(audio_file),
            'language': language,
            'pytorch_text': pytorch_text,
            'pytorch_time': pytorch_time,
            'mlx_text': mlx_text,
            'mlx_time': mlx_time,
            'speedup': speedup,
            'similarity': similarity,
            'mlx_tokens': mlx_tokens
        })
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("📊 对比结果汇总")
    print("=" * 80)
    
    total_pytorch_time = sum(r['pytorch_time'] for r in results)
    total_mlx_time = sum(r['mlx_time'] for r in results)
    avg_similarity = sum(r['similarity'] for r in results) / len(results) if results else 0
    avg_speedup = total_pytorch_time/total_mlx_time if total_mlx_time > 0 else 0
    
    # 创建 Rich 表格
    console = Console()
    table = Table(title="", show_header=True, header_style="bold cyan")
    
    # 添加列
    table.add_column("文件", style="yellow", no_wrap=True, width=12)
    table.add_column("PyTorch(s)", justify="right", style="red", width=12)
    table.add_column("MLX(s)", justify="right", style="green", width=10)
    table.add_column("加速比", justify="right", style="magenta", width=10)
    table.add_column("相似度", justify="right", style="blue", width=10)
    
    # 添加数据行
    for r in results:
        table.add_row(
            r['file'],
            f"{r['pytorch_time']:.3f}",
            f"{r['mlx_time']:.3f}",
            f"{r['speedup']:.2f}x",
            f"{r['similarity']:.1f}%"
        )
    
    # 添加分隔线
    table.add_section()
    
    # 添加汇总行
    table.add_row(
        "总计/平均",
        f"{total_pytorch_time:.3f}",
        f"{total_mlx_time:.3f}",
        f"{avg_speedup:.2f}x",
        f"{avg_similarity:.1f}%",
        style="bold"
    )
    
    # 打印表格
    print()
    console.print(table)
    
    # 保存详细结果
    output_file = "model_comparison_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("SenseVoice 模型对比结果\n")
        f.write("=" * 80 + "\n\n")
        
        for r in results:
            f.write(f"文件: {r['file']}\n")
            f.write(f"语言: {r['language']}\n")
            f.write(f"PyTorch 结果: {r['pytorch_text']}\n")
            f.write(f"PyTorch 时间: {r['pytorch_time']:.3f}秒\n")
            f.write(f"MLX 结果: {r['mlx_text']}\n")
            f.write(f"MLX 时间: {r['mlx_time']:.3f}秒\n")
            f.write(f"加速比: {r['speedup']:.2f}x\n")
            f.write(f"文本相似度: {r['similarity']:.1f}%\n")
            if 'mlx_tokens' in r and r['mlx_tokens']:
                f.write(f"MLX Tokens (前20个): {r['mlx_tokens'][:20]}\n")
            f.write("-" * 40 + "\n\n")
        
        f.write(f"\n总计:\n")
        f.write(f"PyTorch 总时间: {total_pytorch_time:.3f}秒\n")
        f.write(f"MLX 总时间: {total_mlx_time:.3f}秒\n")
        f.write(f"平均加速比: {total_pytorch_time/total_mlx_time if total_mlx_time > 0 else 0:.2f}x\n")
        f.write(f"平均文本相似度: {avg_similarity:.1f}%\n")
    
    print(f"\n💾 详细结果已保存到: {output_file}")


def main():
    """主函数"""
    # 示例音频文件
    examples_dir = "/Users/taylor/Documents/code/SenseVoice/examples"
    
    if not os.path.exists(examples_dir):
        print(f"❌ 示例文件夹不存在: {examples_dir}")
        return
    
    # 获取所有音频文件
    audio_files = sorted([
        os.path.join(examples_dir, f) 
        for f in os.listdir(examples_dir) 
        if f.endswith('.mp3')
    ])
    
    if not audio_files:
        print(f"❌ 在 {examples_dir} 中没有找到音频文件")
        return
    
    print(f"📂 找到 {len(audio_files)} 个音频文件")
    
    # 进行对比
    compare_models(audio_files)


if __name__ == "__main__":
    main()