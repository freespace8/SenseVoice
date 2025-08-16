#!/usr/bin/env python3
"""
SenseVoice 统一配置模块
移除硬编码路径，提供灵活的配置管理
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """统一配置管理器"""
    
    def __init__(self):
        # 基础路径配置
        self.project_root = Path(__file__).parent.parent.parent
        self.cache_dir = self._get_cache_dir()
        
        # 模型配置
        self.model_name = os.getenv("SENSEVOICE_MODEL", "iic/SenseVoiceSmall")
        self.model_dir = self._get_model_dir()
        
        # MLX权重文件
        self.mlx_weights_path = self._get_mlx_weights()
        
        # 资源文件路径
        self.vocab_path = self._get_vocab_path()
        self.cmvn_file = self._get_cmvn_file()
        
        # 音频处理配置
        self.sample_rate = 16000
        self.dither = 0.0  # 确保确定性输出
        
        # 推理配置
        self.device = os.getenv("SENSEVOICE_DEVICE", "auto")  # auto, cpu, cuda, mlx
        self.batch_size = int(os.getenv("SENSEVOICE_BATCH_SIZE", "1"))
        
    def _get_cache_dir(self) -> Path:
        """获取缓存目录"""
        # 优先使用环境变量
        if cache_dir := os.getenv("SENSEVOICE_CACHE_DIR"):
            return Path(cache_dir)
        
        # 尝试标准缓存位置
        home = Path.home()
        cache_locations = [
            home / ".cache" / "modelscope" / "hub" / "models",
            home / ".cache" / "sensevoice",
            self.project_root / "models"
        ]
        
        for location in cache_locations:
            if location.exists():
                return location
        
        # 默认创建项目本地缓存
        default_cache = self.project_root / "models"
        default_cache.mkdir(parents=True, exist_ok=True)
        return default_cache
    
    def _get_model_dir(self) -> Path:
        """获取模型目录"""
        model_dir = self.cache_dir / self.model_name.replace("/", os.sep)
        if not model_dir.exists():
            # 尝试查找其他可能的位置
            alt_locations = [
                self.project_root / "models" / self.model_name.replace("/", "_"),
                self.project_root / self.model_name.split("/")[-1]
            ]
            for alt in alt_locations:
                if alt.exists():
                    return alt
        return model_dir
    
    def _get_mlx_weights(self) -> Optional[Path]:
        """获取MLX权重文件路径"""
        # 优先使用环境变量
        if weights_path := os.getenv("SENSEVOICE_MLX_WEIGHTS"):
            return Path(weights_path)
        
        # 查找可能的权重文件
        possible_names = [
            "sensevoice_mlx_converted_fixed.safetensors",
            "sensevoice_mlx.safetensors",
            "model_mlx.safetensors",
            "weights.safetensors"
        ]
        
        search_dirs = [
            self.project_root,
            self.project_root / "models",
            self.model_dir
        ]
        
        for dir_path in search_dirs:
            if dir_path.exists():
                for name in possible_names:
                    weight_file = dir_path / name
                    if weight_file.exists():
                        return weight_file
        
        return None
    
    def _get_vocab_path(self) -> Optional[Path]:
        """获取词汇表路径"""
        possible_paths = [
            self.model_dir / "tokens.json",
            self.model_dir / "vocab.json",
            self.project_root / "tokens.json"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _get_cmvn_file(self) -> Optional[Path]:
        """获取CMVN文件路径"""
        possible_paths = [
            self.model_dir / "am.mvn",
            self.model_dir / "cmvn.ark",
            self.project_root / "am.mvn"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def validate(self) -> tuple[bool, list[str]]:
        """验证配置完整性"""
        errors = []
        
        if not self.model_dir.exists():
            errors.append(f"模型目录不存在: {self.model_dir}")
        
        if self.mlx_weights_path and not self.mlx_weights_path.exists():
            errors.append(f"MLX权重文件不存在: {self.mlx_weights_path}")
        
        if not self.vocab_path or not self.vocab_path.exists():
            errors.append("词汇表文件未找到")
        
        if not self.cmvn_file or not self.cmvn_file.exists():
            errors.append("CMVN文件未找到")
        
        return len(errors) == 0, errors
    
    def __str__(self) -> str:
        """配置信息字符串表示"""
        info = [
            "SenseVoice 配置信息:",
            f"  项目根目录: {self.project_root}",
            f"  缓存目录: {self.cache_dir}",
            f"  模型目录: {self.model_dir}",
            f"  MLX权重: {self.mlx_weights_path}",
            f"  词汇表: {self.vocab_path}",
            f"  CMVN文件: {self.cmvn_file}",
            f"  设备: {self.device}",
        ]
        return "\n".join(info)


# 全局配置实例
config = Config()