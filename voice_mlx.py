#!/usr/bin/env python3
"""
VoiceMLX - SenseVoice MLX 模型封装类
提供简洁的 API 接口进行语音识别
"""

import os
import time
import json
import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple

# MLX 相关导入
import mlx.core as mx

# 导入模型和前端处理
from model_mlx import SenseVoiceMLX
from utils.frontend_mlx import create_frontend_mlx


class VoiceMLX:
    """SenseVoice MLX 模型封装类
    
    提供简洁的接口进行语音识别，支持多语言和多种音频格式。
    
    Example:
        >>> voice = VoiceMLX()
        >>> result = voice.transcribe("audio.mp3")
        >>> print(result['text'])
    """
    
    def __init__(
        self,
        model_path: str = "/Users/taylor/Documents/code/SenseVoice/model/model_mlx.safetensors",
        model_dir: str = "/Users/taylor/.cache/modelscope/hub/models/iic/SenseVoiceSmall",
        device: str = "auto",
        verbose: bool = True
    ):
        """初始化 VoiceMLX
        
        Args:
            model_path: MLX 模型权重文件路径
            model_dir: 模型目录，包含 tokenizer 和配置文件
            device: 设备类型（auto/cpu/gpu）
            verbose: 是否输出详细信息
        """
        self.model_path = model_path
        self.model_dir = model_dir
        self.device = device
        self.verbose = verbose
        
        # 初始化组件
        self.model = None
        self.frontend = None
        self.tokenizer = None
        
        # 模型配置
        self.model_config = {
            "input_size": 560,  # LFR 特征维度
            "vocab_size": 25055,
            "encoder_conf": {
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
        }
        
        # 语言映射
        self.language_map = {
            'zh': 'Chinese',
            'en': 'English',
            'ja': 'Japanese',
            'ko': 'Korean',
            'yue': 'Cantonese',
            'auto': 'Auto-detect'
        }
        
        # 初始化模型
        self._initialize()
    
    def _initialize(self):
        """初始化所有组件"""
        if self.verbose:
            print("🚀 初始化 VoiceMLX...")
        
        # 加载模型
        self._load_model()
        
        # 初始化前端处理器
        self._init_frontend()
        
        # 加载 tokenizer
        self._load_tokenizer()
        
        if self.verbose:
            print("✅ VoiceMLX 初始化完成")
    
    def _load_model(self):
        """加载 MLX 模型"""
        if self.verbose:
            print("📦 加载 MLX 模型...")
        
        start_time = time.time()
        
        try:
            # 初始化模型架构
            self.model = SenseVoiceMLX(
                input_size=self.model_config["input_size"],
                vocab_size=self.model_config["vocab_size"],
                encoder_conf=self.model_config["encoder_conf"]
            )
            
            # 加载权重
            if os.path.exists(self.model_path):
                weights = mx.load(self.model_path)
                self.model.load_weights(weights)
                
                if self.verbose:
                    load_time = time.time() - start_time
                    print(f"   ✅ 模型加载成功 (耗时: {load_time:.2f}秒)")
            else:
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
                
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def _init_frontend(self):
        """初始化前端处理器"""
        if self.verbose:
            print("📐 初始化前端处理器...")
        
        cmvn_file = os.path.join(self.model_dir, "am.mvn")
        self.frontend = create_frontend_mlx(
            cmvn_file=cmvn_file if os.path.exists(cmvn_file) else None
        )
        
        if self.verbose:
            print("   ✅ 前端处理器初始化成功")
    
    def _load_tokenizer(self):
        """加载 tokenizer"""
        if self.verbose:
            print("📖 加载 Tokenizer...")
        
        try:
            # 尝试加载 sentencepiece tokenizer
            from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
            tokenizer_conf = {
                "sentencepiece_model": os.path.join(self.model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")
            }
            self.tokenizer = SentencepiecesTokenizer(**tokenizer_conf)
            
            if self.verbose:
                print("   ✅ SentencePiece Tokenizer 加载成功")
        except:
            # 备用方案：使用 tokens.json
            try:
                tokens_file = os.path.join(self.model_dir, "tokens.json")
                with open(tokens_file, 'r', encoding='utf-8') as f:
                    tokens = json.load(f)
                
                self.tokenizer = SimpleTokenizer(tokens)
                
                if self.verbose:
                    print("   ✅ Simple Tokenizer 加载成功")
            except:
                if self.verbose:
                    print("   ⚠️  Tokenizer 加载失败，将返回 Token IDs")
                self.tokenizer = None
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: str = "auto",
        return_tokens: bool = False,
        keep_special_tokens: bool = False,
        sample_rate: int = 16000
    ) -> Dict:
        """转录音频
        
        Args:
            audio: 音频文件路径或音频数组
            language: 语言设置 (zh/en/ja/ko/yue/auto)
            return_tokens: 是否返回 token IDs
            keep_special_tokens: 是否保留特殊标记
            sample_rate: 采样率
            
        Returns:
            包含识别结果的字典：
            {
                'text': 识别文本,
                'language': 检测到的语言,
                'tokens': token IDs (如果 return_tokens=True),
                'time': 推理时间,
                'confidence': 置信度 (预留)
            }
        """
        # 加载音频
        if isinstance(audio, str):
            if not os.path.exists(audio):
                raise FileNotFoundError(f"音频文件不存在: {audio}")
            waveform, sr = librosa.load(audio, sr=sample_rate, mono=True)
        else:
            waveform = audio
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=0)  # 转换为单声道
        
        # 提取特征
        features = self._extract_features(waveform)
        
        # 执行推理
        start_time = time.time()
        text, tokens = self._inference(features, language, keep_special_tokens)
        inference_time = time.time() - start_time
        
        # 构建返回结果
        result = {
            'text': text,
            'language': self._detect_language(text),
            'time': inference_time
        }
        
        if return_tokens:
            result['tokens'] = tokens
        
        return result
    
    def transcribe_batch(
        self,
        audio_files: List[str],
        language: str = "auto",
        **kwargs
    ) -> List[Dict]:
        """批量转录音频文件
        
        Args:
            audio_files: 音频文件路径列表
            language: 语言设置
            **kwargs: 其他参数传递给 transcribe
            
        Returns:
            识别结果列表
        """
        results = []
        
        for audio_file in audio_files:
            try:
                result = self.transcribe(audio_file, language, **kwargs)
                result['file'] = os.path.basename(audio_file)
                results.append(result)
            except Exception as e:
                results.append({
                    'file': os.path.basename(audio_file),
                    'text': f"[错误: {str(e)}]",
                    'error': str(e)
                })
        
        return results
    
    def _extract_features(self, waveform: np.ndarray) -> mx.array:
        """提取音频特征
        
        Args:
            waveform: 音频波形数组
            
        Returns:
            MLX 格式的特征数组
        """
        # 使用前端处理器提取特征
        features = self.frontend(waveform, output_format="mlx")
        return features
    
    def _inference(
        self,
        features: mx.array,
        language: str = "auto",
        keep_special_tokens: bool = False
    ) -> Tuple[str, List[int]]:
        """执行模型推理
        
        Args:
            features: 音频特征
            language: 语言设置
            keep_special_tokens: 是否保留特殊标记
            
        Returns:
            (识别文本, token IDs)
        """
        # 准备输入
        speech_lengths = mx.array([features.shape[1]], dtype=mx.int32)
        
        # 执行推理
        outputs = self.model(features, speech_lengths)
        ctc_logits = outputs['ctc_logits']
        
        # CTC 解码
        token_ids = mx.argmax(ctc_logits[0], axis=-1)
        
        # 去除重复和空白标记
        token_ids_np = np.array(token_ids)
        decoded_tokens = []
        prev_token = -1
        
        for token in token_ids_np:
            if token != prev_token and token != 0:  # 0 是 blank token
                decoded_tokens.append(int(token))
            prev_token = token
        
        # 解码为文本
        if self.tokenizer and len(decoded_tokens) > 0:
            try:
                text = self.tokenizer.decode(decoded_tokens, keep_special_tokens=keep_special_tokens)
            except:
                text = self._tokens_to_string(decoded_tokens)
        else:
            text = self._tokens_to_string(decoded_tokens)
        
        return text, decoded_tokens
    
    def _tokens_to_string(self, tokens: List[int]) -> str:
        """将 token IDs 转换为字符串表示"""
        if len(tokens) > 20:
            return f"[Tokens: {tokens[:20]}...]"
        else:
            return f"[Tokens: {tokens}]"
    
    def _detect_language(self, text: str) -> str:
        """从文本中检测语言
        
        Args:
            text: 识别文本
            
        Returns:
            检测到的语言代码
        """
        # 检查特殊标记
        language_tags = {
            '<|zh|>': 'zh',
            '<|en|>': 'en',
            '<|ja|>': 'ja',
            '<|ko|>': 'ko',
            '<|yue|>': 'yue'
        }
        
        for tag, lang in language_tags.items():
            if tag in text:
                return lang
        
        return 'auto'
    
    def benchmark(self, audio_file: str, iterations: int = 10) -> Dict:
        """性能基准测试
        
        Args:
            audio_file: 测试音频文件
            iterations: 测试迭代次数
            
        Returns:
            性能统计信息
        """
        times = []
        
        for i in range(iterations):
            result = self.transcribe(audio_file, keep_special_tokens=False)
            times.append(result['time'])
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'iterations': iterations
        }


class SimpleTokenizer:
    """简单的 Tokenizer 实现"""
    
    def __init__(self, tokens: List[str]):
        """初始化
        
        Args:
            tokens: token 列表
        """
        self.tokens = tokens
        self.id2token = {i: token for i, token in enumerate(tokens)}
    
    def decode(self, token_ids: List[int], keep_special_tokens: bool = False) -> str:
        """将 token IDs 解码为文本
        
        Args:
            token_ids: Token ID 列表
            keep_special_tokens: 是否保留特殊标记
            
        Returns:
            解码后的文本
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


def main():
    """示例用法"""
    # 创建 VoiceMLX 实例
    voice = VoiceMLX(verbose=True)
    
    # 示例音频文件
    examples_dir = "/Users/taylor/Documents/code/SenseVoice/examples"
    
    if os.path.exists(examples_dir):
        audio_files = [
            os.path.join(examples_dir, f)
            for f in os.listdir(examples_dir)
            if f.endswith('.mp3')
        ]
        
        if audio_files:
            print("\n" + "=" * 60)
            print("📝 转录示例")
            print("=" * 60)
            
            for audio_file in audio_files[:2]:  # 只测试前两个文件
                print(f"\n🎵 音频文件: {os.path.basename(audio_file)}")
                
                # 转录
                result = voice.transcribe(
                    audio_file,
                    language="auto",
                    keep_special_tokens=True
                )
                
                print(f"📝 识别结果: {result['text']}")
                print(f"⏱️  推理时间: {result['time']:.3f}秒")
                print(f"🌐 检测语言: {result['language']}")
            
            # 性能基准测试
            if audio_files:
                print("\n" + "=" * 60)
                print("⚡ 性能基准测试")
                print("=" * 60)
                
                benchmark = voice.benchmark(audio_files[0], iterations=5)
                print(f"平均时间: {benchmark['mean_time']:.3f}秒")
                print(f"标准差: {benchmark['std_time']:.3f}秒")
                print(f"最快: {benchmark['min_time']:.3f}秒")
                print(f"最慢: {benchmark['max_time']:.3f}秒")


if __name__ == "__main__":
    main()