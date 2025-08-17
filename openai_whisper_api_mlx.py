#!/usr/bin/env python3
"""
SenseVoice MLX - OpenAI Whisper Compatible API
使用 MLX 加速的 SenseVoice 模型，提供与 OpenAI Whisper API 兼容的接口
"""

import os
import time
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import librosa
import soundfile as sf

# 导入 VoiceMLX
from voice_mlx import VoiceMLX


# 配置管理类
class Config:
    """MLX API 配置管理"""
    
    def __init__(self):
        # 服务器配置
        self.HOST = os.getenv("SENSEVOICE_MLX_HOST", "0.0.0.0")
        self.PORT = int(os.getenv("SENSEVOICE_MLX_PORT", "6209"))
        
        # 模型配置
        self.MODEL_PATH = os.getenv(
            "SENSEVOICE_MLX_MODEL_PATH",
            "/Users/taylor/Documents/code/SenseVoice/model/model_mlx.safetensors"
        )
        self.MODEL_DIR = os.getenv(
            "SENSEVOICE_MODEL_DIR",
            os.path.expanduser("~/.cache/modelscope/hub/models/iic/SenseVoiceSmall")
        )
        
        # 文件处理配置
        self.TMP_DIR = os.getenv("SENSEVOICE_MLX_TMP_DIR", "./tmp_mlx")
        self.MAX_FILE_SIZE = int(os.getenv("SENSEVOICE_MAX_FILE_SIZE", str(25 * 1024 * 1024)))  # 25MB
        self.SUPPORTED_FORMATS = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm', '.flac', '.ogg'}
        
        # 音频处理配置
        self.TARGET_SAMPLE_RATE = int(os.getenv("SENSEVOICE_SAMPLE_RATE", "16000"))
        
        # 日志配置
        self.LOG_LEVEL = os.getenv("SENSEVOICE_MLX_LOG_LEVEL", "INFO")
        
        # API配置
        self.API_TITLE = "SenseVoice MLX - OpenAI Compatible API"
        self.API_VERSION = "1.0.0"
        self.API_DESCRIPTION = "高性能 MLX 加速的语音识别服务，兼容 OpenAI Whisper API"
        
        # 确保临时目录存在
        os.makedirs(self.TMP_DIR, exist_ok=True)
        
        # 配置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志系统"""
        log_file = os.path.join(self.TMP_DIR, 'api_mlx.log')
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def print_config(self):
        """打印当前配置"""
        print("\n" + "=" * 60)
        print("🚀 SenseVoice MLX API 配置")
        print("=" * 60)
        print(f"📍 服务地址: http://{self.HOST}:{self.PORT}")
        print(f"💾 模型路径: {self.MODEL_PATH}")
        print(f"📂 模型目录: {self.MODEL_DIR}")
        print(f"🗂️  临时目录: {self.TMP_DIR}")
        print(f"📦 最大文件: {self.MAX_FILE_SIZE // (1024*1024)}MB")
        print(f"📝 日志级别: {self.LOG_LEVEL}")
        print("=" * 60 + "\n")


# 初始化配置
config = Config()
config.print_config()

# FastAPI 应用初始化
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION
)

# 记录应用启动时间
app_start_time = time.time()

# 全局模型实例
model: Optional[VoiceMLX] = None


def initialize_model():
    """初始化 MLX 模型"""
    global model
    
    try:
        print("⏳ 正在初始化 MLX 模型...")
        start_time = time.time()
        
        model = VoiceMLX(
            model_path=config.MODEL_PATH,
            model_dir=config.MODEL_DIR,
            verbose=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ MLX 模型初始化成功 (耗时: {load_time:.2f}秒)")
        
        # 预热模型
        print("🔥 预热模型...")
        warmup_audio = np.zeros(16000, dtype=np.float32)  # 1秒静音
        _ = model.transcribe(warmup_audio, language="auto")
        print("✅ 模型预热完成")
        
        return model
        
    except Exception as e:
        config.logger.error(f"模型初始化失败: {e}")
        print(f"❌ 模型初始化失败: {e}")
        print("💡 请检查：")
        print("   1. 模型文件是否存在")
        print("   2. 模型路径是否正确")
        print("   3. Python环境和依赖是否正确")
        raise RuntimeError(f"模型初始化失败: {e}")


# 应用启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    try:
        initialize_model()
        print(f"🎉 服务启动成功！访问 http://{config.HOST}:{config.PORT}/docs 查看 API 文档")
    except Exception as e:
        config.logger.error(f"启动失败: {e}")
        print(f"💥 启动失败: {e}")


# API 响应模型
class TranscriptionResponse(BaseModel):
    """转录响应模型"""
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None
    segments: Optional[List[Dict[str, Any]]] = None


class TranslationResponse(BaseModel):
    """翻译响应模型"""
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None


# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - app_start_time,
        "model_loaded": model is not None,
        "config": {
            "api_version": config.API_VERSION,
            "max_file_size_mb": config.MAX_FILE_SIZE // (1024 * 1024),
            "supported_formats": list(config.SUPPORTED_FORMATS)
        }
    }
    
    if model is None:
        health_status["status"] = "unhealthy"
        health_status["error"] = "Model not loaded"
        return JSONResponse(content=health_status, status_code=503)
    
    return health_status


# 统计端点
@app.get("/stats")
async def stats():
    """API 使用统计"""
    return {
        "uptime_seconds": time.time() - app_start_time,
        "model_loaded": model is not None,
        "model_type": "MLX",
        "acceleration": "Apple Silicon",
        "config": {
            "model_path": config.MODEL_PATH,
            "sample_rate": config.TARGET_SAMPLE_RATE,
            "max_file_size_mb": config.MAX_FILE_SIZE // (1024 * 1024),
            "supported_formats": list(config.SUPPORTED_FORMATS)
        }
    }


def format_text(text: str, response_format: str = "json") -> str:
    """格式化输出文本，去除特殊标记"""
    if response_format == "json" or response_format == "verbose_json":
        # 去除特殊标记
        import re
        # 移除语言标记
        text = re.sub(r'<\|[a-z]+\|>', '', text)
        # 移除情感标记
        text = re.sub(r'<\|[A-Z]+\|>', '', text)
        # 移除事件标记
        text = re.sub(r'<\|[A-Za-z_]+\|>', '', text)
        text = text.strip()
    
    return text


def process_audio_file(file_path: str, language: str = "auto") -> Dict[str, Any]:
    """处理音频文件"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not initialized"
        )
    
    try:
        # 加载音频
        audio, sr = librosa.load(file_path, sr=config.TARGET_SAMPLE_RATE, mono=True)
        duration = len(audio) / sr
        
        # 转录
        result = model.transcribe(
            audio,
            language=language,
            return_tokens=False,
            keep_special_tokens=False  # 不保留特殊标记
        )
        
        # 构建响应
        response = {
            "text": result["text"],
            "language": result.get("language", language),
            "duration": duration,
            "inference_time": result.get("time", 0)
        }
        
        return response
        
    except Exception as e:
        config.logger.error(f"音频处理失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(e)}"
        )


# 主要 API 端点

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: Optional[List[str]] = Form(default=None)
):
    """
    转录音频文件（OpenAI Whisper API 兼容）
    
    Args:
        file: 音频文件
        model: 模型名称（兼容性参数，实际使用 MLX 模型）
        language: 语言代码 (zh/en/ja/ko/yue/auto)
        prompt: 提示文本（暂不支持）
        response_format: 响应格式 (json/text/srt/verbose_json/vtt)
        temperature: 温度参数（暂不支持）
        timestamp_granularities: 时间戳粒度（暂不支持）
    """
    
    # 检查文件大小
    if file.size and file.size > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum limit of {config.MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # 检查文件格式
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format. Supported formats: {', '.join(config.SUPPORTED_FORMATS)}"
        )
    
    # 保存临时文件
    temp_file = None
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
            dir=config.TMP_DIR
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name
        
        # 处理音频
        result = process_audio_file(
            temp_file,
            language=language or "auto"
        )
        
        # 格式化响应
        text = format_text(result["text"], response_format)
        
        if response_format == "text":
            return text
        
        elif response_format == "srt":
            # 简单的 SRT 格式
            srt_content = f"1\n00:00:00,000 --> 00:00:{int(result['duration']):02d},000\n{text}\n"
            return srt_content
        
        elif response_format == "vtt":
            # 简单的 WebVTT 格式
            vtt_content = f"WEBVTT\n\n00:00:00.000 --> 00:00:{int(result['duration']):02d}.000\n{text}\n"
            return vtt_content
        
        else:  # json 或 verbose_json
            response = {
                "text": text,
                "language": result.get("language", "auto"),
                "duration": result.get("duration", 0)
            }
            
            if response_format == "verbose_json":
                # 添加更多详细信息
                response["task"] = "transcribe"
                response["model"] = "sensevoice-mlx"
                response["inference_time"] = result.get("inference_time", 0)
                
                # 创建简单的段落信息
                response["segments"] = [{
                    "id": 0,
                    "start": 0.0,
                    "end": result.get("duration", 0),
                    "text": text,
                    "temperature": temperature
                }]
            
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        config.logger.error(f"转录失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


@app.post("/v1/audio/translations")
async def translate_audio(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0)
):
    """
    翻译音频文件到英文（OpenAI Whisper API 兼容）
    
    注意：SenseVoice 不直接支持翻译，此端点返回转录结果
    """
    
    # 使用转录功能
    result = await transcribe_audio(
        file=file,
        model=model,
        language="auto",  # 自动检测语言
        prompt=prompt,
        response_format=response_format,
        temperature=temperature
    )
    
    # 如果是 JSON 格式，添加翻译说明
    if isinstance(result, dict):
        result["note"] = "Translation not supported, returning transcription"
    
    return result


# 批量处理端点（扩展功能）
@app.post("/v1/audio/batch")
async def batch_transcribe(
    files: List[UploadFile] = File(...),
    language: str = Form(default="auto"),
    response_format: str = Form(default="json")
):
    """
    批量转录音频文件（扩展功能）
    
    Args:
        files: 音频文件列表
        language: 语言代码
        response_format: 响应格式
    """
    
    results = []
    
    for file in files:
        try:
            # 处理每个文件
            result = await transcribe_audio(
                file=file,
                language=language,
                response_format=response_format
            )
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "result": result
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}


# 性能基准测试端点（扩展功能）
@app.post("/v1/benchmark")
async def benchmark(
    file: UploadFile = File(...),
    iterations: int = Form(default=5)
):
    """
    性能基准测试
    
    Args:
        file: 测试音频文件
        iterations: 测试迭代次数
    """
    
    # 保存临时文件
    temp_file = None
    try:
        # 创建临时文件
        file_ext = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
            dir=config.TMP_DIR
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name
        
        # 运行基准测试
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not initialized"
            )
        
        benchmark_result = model.benchmark(temp_file, iterations=iterations)
        
        return {
            "filename": file.filename,
            "iterations": iterations,
            "mean_time": benchmark_result["mean_time"],
            "std_time": benchmark_result["std_time"],
            "min_time": benchmark_result["min_time"],
            "max_time": benchmark_result["max_time"],
            "throughput": 1.0 / benchmark_result["mean_time"] if benchmark_result["mean_time"] > 0 else 0
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}"
        )
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


# 根路径
@app.get("/")
async def root():
    """API 根路径"""
    return {
        "name": config.API_TITLE,
        "version": config.API_VERSION,
        "description": config.API_DESCRIPTION,
        "endpoints": {
            "transcribe": "/v1/audio/transcriptions",
            "translate": "/v1/audio/translations",
            "batch": "/v1/audio/batch",
            "benchmark": "/v1/benchmark",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        },
        "model": "SenseVoice MLX",
        "acceleration": "Apple Silicon Optimized"
    }


if __name__ == "__main__":
    import uvicorn
    
    # 运行服务器
    uvicorn.run(
        "openai_whisper_api_mlx:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,  # 生产环境设为 False
        log_level=config.LOG_LEVEL.lower()
    )