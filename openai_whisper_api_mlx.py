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
from pydantic import BaseModel, Field
from pycorrector import MacBertCorrector
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
        
        # 创建专用的logger
        self.logger = logging.getLogger('sensevoice_mlx')
        self.logger.setLevel(getattr(logging, self.LOG_LEVEL.upper()))
        
        # 防止日志传播到root logger，避免重复输出
        self.logger.propagate = False
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            console_handler.setFormatter(console_formatter)
            
            # 文件handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            
            # 添加handlers
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
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
corrector_model: Optional[MacBertCorrector] = None

def get_corrector() -> MacBertCorrector:
    """
    获取 MacBertCorrector 模型实例（懒加载）。
    模型在首次调用时初始化并缓存。
    """
    global corrector_model
    if corrector_model is None:
        config.logger.info("🔧 正在首次初始化 MacBertCorrector 文本纠错模型...")
        config.logger.info("📦 模型来源: shibing624/macbert4csc-base-chinese")
        start_time = time.time()
        # 注意：MacBertCorrector 默认会从 huggingface 下载模型，
        # 也可以指定本地模型路径 model_name_or_path
        corrector_model = MacBertCorrector()
        load_time = time.time() - start_time
        config.logger.info(f"✅ MacBertCorrector 模型初始化成功 (耗时: {load_time:.2f}秒)")
        config.logger.info("🎯 文本纠错功能已就绪")
    return corrector_model


def initialize_model():
    """初始化 MLX 模型"""
    global model
    
    try:
        print("⏳ 正在初始化 MLX 模型...")
        start_time = time.time()
        
        # 从环境变量读取是否启用标点恢复（默认启用）
        enable_punctuation = os.getenv("SENSEVOICE_ENABLE_PUNCTUATION", "true").lower() == "true"
        
        model = VoiceMLX(
            model_path=config.MODEL_PATH,
            model_dir=config.MODEL_DIR,
            verbose=True,
            enable_punctuation=enable_punctuation
        )
        
        if enable_punctuation:
            config.logger.info("✅ 标点恢复功能已启用")
        else:
            config.logger.info("ℹ️ 标点恢复功能未启用（设置 SENSEVOICE_ENABLE_PUNCTUATION=true 以启用）")
        
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
class CorrectionResult(BaseModel):
    """文本纠错结果模型"""
    source: str
    target: str
    errors: List[tuple] = Field(default_factory=list)

class TranscriptionResponse(BaseModel):
    """转录响应模型"""
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None
    segments: Optional[List[Dict[str, Any]]] = None
    correction: Optional[CorrectionResult] = None # 新增字段


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


def correct_text_with_macbert(text: str) -> Dict[str, Any]:
    """
    使用 MacBert 模型对文本进行纠错。

    Args:
        text: 待纠错的原始文本。

    Returns:
        一个包含纠错结果的字典。
    """
    config.logger.info(f"🔍 开始文本纠错处理...")
    config.logger.info(f"📝 原始文本: {text}")
    config.logger.info(f"📏 文本长度: {len(text)} 字符")
    
    # 检查文本长度，对于5个字符以下的短文本跳过纠错
    if len(text.strip()) < 5:
        config.logger.info("🛡️ 文本过短（<5字符），跳过纠错以避免误判")
        return {
            "source": text,
            "target": text,
            "errors": [],
            "inference_time": 0.0,
            "skip_reason": "文本过短（<5字符）"
        }
    
    try:
        corrector = get_corrector()
        config.logger.info("⚡ 正在执行 MacBert 文本纠错...")
        start_time = time.time()
        result = corrector.correct(text)
        inference_time = time.time() - start_time
        
        corrected_text = result.get('target', text)
        errors = result.get('errors', [])
        
        # 对纠错结果进行质量检查
        if _should_accept_correction(text, corrected_text, errors):
            config.logger.info(f"✅ 纠错处理完成 (耗时: {inference_time:.3f}秒)")
            config.logger.info(f"📝 纠错结果: {corrected_text}")
            config.logger.info(f"🔧 发现错误: {len(errors)} 个")
            
            if errors:
                config.logger.info("📋 错误详情:")
                for i, error in enumerate(errors, 1):
                    if len(error) >= 3:
                        wrong_char, correct_char, position = error[0], error[1], error[2]
                        config.logger.info(f"   {i}. 位置{position}: '{wrong_char}' → '{correct_char}'")
                    else:
                        config.logger.info(f"   {i}. {error}")
            else:
                config.logger.info("✨ 文本无需纠错，质量良好")

            # 将 pycorrector 的输出格式化
            return {
                "source": result.get('source', text),
                "target": corrected_text,
                "errors": errors,
                "inference_time": inference_time
            }
        else:
            config.logger.info(f"🚫 纠错结果质量检查未通过，保持原文")
            config.logger.info(f"📝 原始文本: {text}")
            config.logger.info(f"📝 模型建议: {corrected_text}")
            config.logger.info("✨ 保留原始文本以避免错误纠正")
            
            return {
                "source": text,
                "target": text,
                "errors": [],
                "inference_time": inference_time,
                "rejected_suggestion": corrected_text
            }
            
    except Exception as e:
        config.logger.error(f"❌ MacBert 文本纠错失败: {e}")
        config.logger.info("🔄 启用容错模式，返回原始文本")
        # 在纠错失败时，优雅地返回原始文本，不中断主流程
        return {
            "source": text,
            "target": text,
            "errors": [],
            "error_message": str(e)
        }


def _should_accept_correction(original: str, corrected: str, errors: List) -> bool:
    """
    检查纠错结果是否应该被接受
    
    Args:
        original: 原始文本
        corrected: 纠错后文本
        errors: 错误列表
        
    Returns:
        是否接受纠错结果
    """
    # 如果没有修改，直接接受
    if original == corrected:
        return True
    
    # 检查修改的合理性 - 如果修改过多，可能是误判
    if len(errors) > len(original) // 2:
        return False
    
    # 检查是否为全文替换（可能是模型错误）
    if len(errors) == len(original) and len(original) <= 8:
        return False
    
    return True


def process_audio_file(file_path: str, language: str = "auto", enable_punctuation: Optional[bool] = None) -> Dict[str, Any]:
    """处理音频文件
    
    Args:
        file_path: 音频文件路径
        language: 语言设置
        enable_punctuation: 是否启用标点恢复（None 时使用模型默认设置）
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not initialized"
        )
    
    try:
        # 加载音频
        audio, sr = librosa.load(file_path, sr=config.TARGET_SAMPLE_RATE, mono=True)
        duration = len(audio) / sr
        
        # 构建转录参数
        transcribe_kwargs = {
            "audio": audio,
            "language": language,
            "return_tokens": False,
            "keep_special_tokens": False  # 不保留特殊标记
        }
        
        # 如果指定了标点恢复设置，添加到参数中
        if enable_punctuation is not None:
            transcribe_kwargs["enable_punctuation"] = enable_punctuation
        
        # 转录
        result = model.transcribe(**transcribe_kwargs)
        
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
    timestamp_granularities: Optional[List[str]] = Form(default=None),
    enable_punctuation: Optional[bool] = Form(default=None),
    enable_correction: bool = Form(default=True) # 新增参数
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
        enable_punctuation: 是否启用标点恢复（None 时使用默认设置）
        enable_correction: 是否启用MacBert文本纠错（默认启用）
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
            language=language or "auto",
            enable_punctuation=enable_punctuation
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

            # 如果启用了文本纠错，则执行并添加到响应中
            if enable_correction:
                config.logger.info("🎯 启用文本纠错功能")
                correction_result = correct_text_with_macbert(text)
                response["correction"] = correction_result
                config.logger.info("📊 纠错结果已添加到响应中")
                # 可选：用纠错后的文本覆盖原始文本
                # response["text"] = correction_result["target"]
            else:
                config.logger.info("⚠️  文本纠错功能已禁用")
            
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
        app,
        host=config.HOST,
        port=config.PORT,
        reload=False,  # 生产环境设为 False
        log_level=config.LOG_LEVEL.lower()
    )