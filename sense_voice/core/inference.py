#!/usr/bin/env python3
"""
SenseVoice 统一推理器
实现MLX优先，自动回退到PyTorch的智能推理系统
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

# 尝试导入MLX
try:
    import mlx.core as mx
    from safetensors import safe_open
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️ MLX 未安装，将使用 PyTorch 后端")

# 尝试导入PyTorch
try:
    from funasr import AutoModel
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    if not MLX_AVAILABLE:
        print("❌ MLX 和 PyTorch 都未安装，请至少安装一个推理后端")

# 音频处理库
try:
    import soundfile as sf
    import librosa
except ImportError:
    print("⚠️ 音频处理库未完整安装")

from .config import config


class UnifiedTranscriber:
    """
    统一转录器 - MLX优先，自动回退到PyTorch
    提供统一的接口，屏蔽底层实现细节
    """
    
    def __init__(self, force_backend: Optional[str] = None):
        """
        初始化统一转录器
        
        Args:
            force_backend: 强制使用指定后端 ('mlx', 'pytorch', None为自动)
        """
        self.backend = None
        self.model = None
        self.frontend = None
        self.id2token = None
        
        # 决定使用哪个后端
        if force_backend == 'mlx' and MLX_AVAILABLE:
            self._init_mlx()
        elif force_backend == 'pytorch' and PYTORCH_AVAILABLE:
            self._init_pytorch()
        else:
            # 自动选择：MLX优先
            if MLX_AVAILABLE and config.mlx_weights_path and config.mlx_weights_path.exists():
                print("✅ 检测到 MLX 可用，使用 MLX 后端进行加速推理")
                self._init_mlx()
            elif PYTORCH_AVAILABLE:
                print("⚠️ MLX 不可用或权重文件未找到，回退到 PyTorch 后端")
                self._init_pytorch()
            else:
                print("❌ 没有可用的推理后端")
    
    def _init_mlx(self):
        """初始化MLX后端"""
        try:
            print("🍎 初始化 MLX 模型...")
            
            # 延迟导入MLX特定模块
            from .model_mlx import SenseVoiceMLX
            from .frontend_unified import create_unified_frontend
            
            # 初始化模型
            self.model = SenseVoiceMLX(input_size=560)
            
            # 加载权重
            self._load_mlx_weights()
            
            # 创建前端处理器
            self.frontend = create_unified_frontend(
                cmvn_file=str(config.cmvn_file) if config.cmvn_file else None,
                dither=config.dither
            )
            
            # 加载词汇表
            self._load_vocab()
            
            self.backend = 'mlx'
            print("✅ MLX 模型初始化完成")
            
        except Exception as e:
            print(f"❌ MLX 初始化失败: {e}")
            if PYTORCH_AVAILABLE:
                print("⚠️ 回退到 PyTorch 后端")
                self._init_pytorch()
    
    def _init_pytorch(self):
        """初始化PyTorch后端"""
        if not PYTORCH_AVAILABLE:
            print("❌ PyTorch 后端不可用")
            return
        
        try:
            print("🔥 初始化 PyTorch 模型...")
            
            # 使用配置中的模型路径
            model_path = str(config.model_dir) if config.model_dir.exists() else config.model_name
            
            self.model = AutoModel(
                model=model_path,
                trust_remote_code=True,
                remote_code=str(config.model_dir / "model.py") if (config.model_dir / "model.py").exists() else None,
                vad_model=None,
                device="cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu",
            )
            
            self.backend = 'pytorch'
            print("✅ PyTorch 模型初始化完成")
            
        except Exception as e:
            print(f"❌ PyTorch 初始化失败: {e}")
    
    def _load_mlx_weights(self):
        """加载MLX模型权重"""
        if not config.mlx_weights_path or not config.mlx_weights_path.exists():
            raise FileNotFoundError(f"MLX权重文件未找到: {config.mlx_weights_path}")
        
        weights = {}
        with safe_open(str(config.mlx_weights_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                weights[key] = mx.array(tensor.numpy())
        
        # 灵活加载权重
        for key, value in weights.items():
            try:
                parts = key.split('.')
                current = self.model
                for part in parts[:-1]:
                    if hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        break
                else:
                    if hasattr(current, parts[-1]):
                        setattr(current, parts[-1], value)
            except:
                pass
    
    def _load_vocab(self):
        """加载词汇表"""
        if config.vocab_path and config.vocab_path.exists():
            with open(config.vocab_path, 'r', encoding='utf-8') as f:
                vocab_list = json.load(f)
            self.id2token = {i: token for i, token in enumerate(vocab_list)}
        else:
            self.id2token = None
    
    def transcribe(self, audio_input: Union[str, Path, np.ndarray], language: str = "auto") -> dict:
        """
        统一的转录接口
        
        Args:
            audio_input: 音频文件路径或numpy数组
            language: 语言代码 (auto, zh, en, ja等)
        
        Returns:
            dict: 包含转录结果和元信息的字典
        """
        start_time = time.time()
        
        if self.backend == 'mlx':
            result = self._transcribe_mlx(audio_input, language)
        elif self.backend == 'pytorch':
            result = self._transcribe_pytorch(audio_input, language)
        else:
            return {'text': '', 'error': '没有初始化的推理后端'}
        
        # 添加公共元信息
        result['backend'] = self.backend
        result['total_time'] = time.time() - start_time
        
        if result.get('duration', 0) > 0:
            result['rtf'] = result['duration'] / result['total_time']
        
        return result
    
    def _transcribe_mlx(self, audio_input: Union[str, Path, np.ndarray], language: str) -> dict:
        """MLX后端转录"""
        # 处理音频
        if isinstance(audio_input, (str, Path)):
            audio_path = str(audio_input)
            if audio_path.endswith('.mp3'):
                audio, sr = librosa.load(audio_path, sr=None, mono=True)
            else:
                audio, sr = sf.read(audio_path)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
        else:
            audio = audio_input
            sr = config.sample_rate
        
        # 重采样到16kHz
        if sr != config.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=config.sample_rate)
        
        audio = audio.astype(np.float32)
        duration = len(audio) / config.sample_rate
        
        # 特征提取
        features = self.frontend(audio, output_format="mlx")
        
        # 准备输入
        batch_size, seq_len, _ = features.shape
        x_lens = mx.array([seq_len] * batch_size, dtype=mx.int32)
        
        # 推理
        if language == "auto":
            # 从音频中自动检测语言（这里简化处理）
            language = "zh"  # 默认中文
        
        encoder_out, encoder_out_lens = self.model.encode(features, x_lens, language=language)
        ctc_out = self.model.ctc.get_logits(encoder_out)
        
        # CTC解码
        predictions = mx.argmax(ctc_out, axis=-1)
        
        # 去除重复和空白
        decoded = []
        prev_token = -1
        for token in predictions[0].tolist():
            if token != 0 and token != prev_token:
                decoded.append(token)
            prev_token = token
        
        # 解码为文本
        text = self._decode_tokens(decoded)
        
        return {
            'text': text,
            'duration': duration,
            'language': language,
            'tokens': decoded
        }
    
    def _transcribe_pytorch(self, audio_input: Union[str, Path, np.ndarray], language: str) -> dict:
        """PyTorch后端转录"""
        # 如果是numpy数组，先保存为临时文件
        temp_file = None
        if isinstance(audio_input, np.ndarray):
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, audio_input, config.sample_rate)
            audio_path = temp_file.name
        else:
            audio_path = str(audio_input)
        
        try:
            # 使用FunASR推理
            res = self.model.generate(
                input=audio_path,
                cache={},
                language=language if language != "auto" else "auto",
                use_itn=False,
            )
            
            # 获取音频时长
            if audio_path.endswith('.mp3'):
                audio, sr = librosa.load(audio_path, sr=None)
            else:
                audio, sr = sf.read(audio_path)
            duration = len(audio) / sr
            
            # 提取结果
            if res and len(res) > 0:
                text = res[0].get("text", "")
                return {
                    'text': text,
                    'duration': duration,
                    'language': res[0].get("language", language)
                }
            
            return {
                'text': "",
                'duration': duration,
                'language': language
            }
            
        finally:
            # 清理临时文件
            if temp_file:
                os.unlink(temp_file.name)
    
    def _decode_tokens(self, token_ids: list) -> str:
        """解码token ID为文本"""
        if self.id2token is None:
            return ' '.join([f"[{t}]" for t in token_ids])
        
        text_parts = []
        for token_id in token_ids:
            token_id = int(token_id)
            
            if token_id == 0:  # blank
                continue
            elif token_id == 1:  # <sos>
                text_parts.append("<S>")
            elif token_id == 2:  # <eos>
                text_parts.append("<E>")
            elif token_id in self.id2token:
                token = self.id2token[token_id]
                text_parts.append(token)
            else:
                text_parts.append(f"[{token_id}]")
        
        text = ''.join(text_parts)
        text = text.replace("▁", " ")
        text = text.strip()
        
        return text
    
    def get_info(self) -> dict:
        """获取模型信息"""
        return {
            'backend': self.backend,
            'mlx_available': MLX_AVAILABLE,
            'pytorch_available': PYTORCH_AVAILABLE,
            'config': str(config)
        }


def create_transcriber(**kwargs) -> UnifiedTranscriber:
    """工厂函数：创建统一转录器实例"""
    return UnifiedTranscriber(**kwargs)