# @Author: Bi Ying
# @Date:   2024-07-10 17:22:55
import shutil
import os
from pathlib import Path
from typing import Union
import logging

import torch
import torchaudio
import numpy as np
from funasr import AutoModel
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, status

# 配置管理类
class Config:
    """应用配置管理"""
    
    def __init__(self):
        # 服务器配置
        self.HOST = os.getenv("SENSEVOICE_HOST", "0.0.0.0")
        self.PORT = int(os.getenv("SENSEVOICE_PORT", "8000"))
        
        # 模型配置
        self.DEVICE = os.getenv("SENSEVOICE_DEVICE", "cpu")
        self.CACHE_DIR = os.getenv("SENSEVOICE_CACHE_DIR") or os.path.expanduser("~/.cache/modelscope/hub/models")
        self.MAIN_MODEL = os.getenv("SENSEVOICE_MAIN_MODEL", "iic/SenseVoiceSmall")
        self.VAD_MODEL = os.getenv("SENSEVOICE_VAD_MODEL", "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
        
        # 文件处理配置
        self.TMP_DIR = os.getenv("SENSEVOICE_TMP_DIR", "./tmp")
        self.MAX_FILE_SIZE = int(os.getenv("SENSEVOICE_MAX_FILE_SIZE", str(25 * 1024 * 1024)))  # 25MB
        self.SUPPORTED_FORMATS = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm', '.flac', '.ogg'}
        
        # 音频处理配置
        self.TARGET_SAMPLE_RATE = int(os.getenv("SENSEVOICE_SAMPLE_RATE", "16000"))
        self.VAD_MAX_SEGMENT_TIME = int(os.getenv("SENSEVOICE_VAD_MAX_SEGMENT", "30000"))
        
        # 日志配置
        self.LOG_LEVEL = os.getenv("SENSEVOICE_LOG_LEVEL", "INFO")
        
        # API配置
        self.API_TITLE = os.getenv("SENSEVOICE_API_TITLE", "SenseVoice OpenAI Compatible API")
        self.API_VERSION = os.getenv("SENSEVOICE_API_VERSION", "1.0.1")
        
        # 确保临时目录存在
        os.makedirs(self.TMP_DIR, exist_ok=True)
        
        # 配置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.TMP_DIR, 'api.log'))
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_model_paths(self):
        """获取模型路径配置"""
        # 主模型路径
        if self.MAIN_MODEL.startswith("/") or self.MAIN_MODEL.startswith("~"):
            main_model_path = os.path.expanduser(self.MAIN_MODEL)
        else:
            main_model_path = str(Path(self.CACHE_DIR) / self.MAIN_MODEL)
        
        # VAD模型路径  
        if self.VAD_MODEL.startswith("/") or self.VAD_MODEL.startswith("~"):
            vad_model_path = os.path.expanduser(self.VAD_MODEL)
        else:
            vad_model_path = str(Path(self.CACHE_DIR) / self.VAD_MODEL)
        
        return main_model_path, vad_model_path
    
    def print_config(self):
        """打印当前配置"""
        print("🔧 SenseVoice API 配置：")
        print(f"   服务地址: http://{self.HOST}:{self.PORT}")
        print(f"   计算设备: {self.DEVICE}")
        print(f"   临时目录: {self.TMP_DIR}")
        print(f"   最大文件: {self.MAX_FILE_SIZE // (1024*1024)}MB")
        print(f"   日志级别: {self.LOG_LEVEL}")
        main_path, vad_path = self.get_model_paths()
        print(f"   主模型: {main_path}")
        print(f"   VAD模型: {vad_path}")

# 初始化配置
config = Config()
config.print_config()

# FastAPI应用初始化
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="SenseVoice语音转文字服务，兼容OpenAI Whisper API"
)

# 临时目录变量 (向后兼容)
TMP_DIR = config.TMP_DIR


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "message": config.API_TITLE, 
        "version": config.API_VERSION,
        "compatibility": "OpenAI Whisper API v1",
        "endpoints": {
            "transcriptions": "/v1/audio/transcriptions",
            "models": "/v1/models",
            "health": "/health",
            "docs": "/docs"
        },
        "supported_formats": sorted(list(config.SUPPORTED_FORMATS)),
        "max_file_size_mb": config.MAX_FILE_SIZE // (1024 * 1024),
        "device": config.DEVICE
    }


@app.get("/health")
async def health():
    """增强的健康检查端点"""
    import psutil
    import time
    
    start_time = time.time()
    
    # 基础状态检查
    health_status = {
        "status": "healthy",
        "timestamp": int(time.time()),
        "uptime": time.time() - app_start_time if 'app_start_time' in globals() else 0,
        "version": config.API_VERSION
    }
    
    # 模型状态检查
    if model is None:
        health_status["status"] = "unhealthy"
        health_status["model_status"] = "unavailable"
        health_status["issues"] = ["Model not loaded"]
    else:
        health_status["model_status"] = "available"
    
    # 系统资源检查
    try:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 内存使用
        memory = psutil.virtual_memory()
        
        # 磁盘空间（临时目录）
        disk = psutil.disk_usage(config.TMP_DIR)
        
        health_status["system"] = {
            "cpu_percent": cpu_percent,
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": round((disk.used / disk.total) * 100, 1)
            }
        }
        
        # 资源警告检查
        issues = []
        if cpu_percent > 90:
            issues.append("High CPU usage")
        if memory.percent > 90:
            issues.append("High memory usage")
        if disk.free < 1024**3:  # 小于1GB可用空间
            issues.append("Low disk space")
            
        if issues:
            health_status["status"] = "degraded"
            health_status["issues"] = issues
    
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["issues"] = [f"System monitoring error: {str(e)}"]
    
    # 配置信息
    health_status["config"] = {
        "device": config.DEVICE,
        "tmp_dir": config.TMP_DIR,
        "max_file_size_mb": config.MAX_FILE_SIZE // (1024 * 1024)
    }
    
    # 响应时间
    health_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
    
    # 设置HTTP状态码
    status_code = 200
    if health_status["status"] == "unhealthy":
        status_code = 503
    elif health_status["status"] == "degraded":
        status_code = 200  # 仍然可用，但有警告
        
    from fastapi import status as http_status
    from fastapi.responses import JSONResponse
    
    return JSONResponse(content=health_status, status_code=status_code)

# 添加统计端点
@app.get("/stats")
async def stats():
    """API使用统计"""
    return {
        "message": "Statistics endpoint",
        "note": "Basic statistics - extend as needed",
        "uptime_seconds": time.time() - app_start_time if 'app_start_time' in globals() else 0,
        "model_loaded": model is not None,
        "config": {
            "device": config.DEVICE,
            "max_file_size_mb": config.MAX_FILE_SIZE // (1024 * 1024),
            "supported_formats": list(config.SUPPORTED_FORMATS)
        }
    }

# 模型初始化函数，使用配置系统
def initialize_model():
    """
    初始化SenseVoice模型，包含完善的错误处理和回退机制
    """
    local_model_path, local_vad_model_path = config.get_model_paths()
    
    vad_kwargs = {"max_single_segment_time": config.VAD_MAX_SEGMENT_TIME}
    model_kwargs = {
        "vad_kwargs": vad_kwargs,
        "trust_remote_code": True,
        "disable_update": True,
        "device": config.DEVICE
    }
    
    try:
        # 策略1: 尝试使用本地主模型和VAD模型
        if os.path.exists(local_model_path):
            print(f"✅ 找到本地主模型: {local_model_path}")
            
            if os.path.exists(local_vad_model_path):
                print(f"✅ 找到本地VAD模型: {local_vad_model_path}")
                try:
                    model = AutoModel(
                        model=local_model_path,
                        vad_model=local_vad_model_path,
                        **model_kwargs
                    )
                    print("🎯 成功加载：本地主模型 + 本地VAD模型")
                    return model
                except Exception as e:
                    print(f"⚠️ 本地模型加载失败: {e}")
            
            # 策略2: 本地主模型 + 在线VAD模型
            try:
                print("🔄 尝试：本地主模型 + 在线VAD模型")
                model = AutoModel(
                    model=local_model_path,
                    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    **model_kwargs
                )
                print("🎯 成功加载：本地主模型 + 在线VAD模型")
                return model
            except Exception as e:
                print(f"⚠️ 混合模式加载失败: {e}")
        
        # 策略3: 完全在线模式（回退选项）
        print("🔄 回退到完全在线模式")
        model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            **model_kwargs
        )
        print("🎯 成功加载：在线主模型 + 在线VAD模型")
        return model
        
    except Exception as e:
        print(f"❌ 模型初始化完全失败: {e}")
        print("💡 请检查：")
        print("   1. 网络连接是否正常")
        print("   2. 模型文件是否完整")
        print("   3. Python环境和依赖是否正确")
        print(f"   4. 设备设置: {config.DEVICE}")
        raise RuntimeError(f"模型初始化失败: {e}")

# 记录应用启动时间
import time
app_start_time = time.time()

# 初始化模型
try:
    model = initialize_model()
    print(f"🚀 模型初始化成功，服务启动完成")
except Exception as e:
    print(f"💥 致命错误：无法初始化模型 - {e}")
    model = None  # 设置为None，在API调用时进行检查

emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "🤧",
}

emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

lang_dict = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {
    "🎼",
    "👏",
    "😀",
    "😭",
    "🤧",
    "😷",
}


def format_str_v2(text: str, show_emo=True, show_event=True):
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = text.count(sptk)
        text = text.replace(sptk, "")

    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    if show_emo:
        text = text + emo_dict[emo]

    for e in event_dict:
        if sptk_dict[e] > 0 and show_event:
            text = event_dict[e] + text

    for emoji in emo_set.union(event_set):
        text = text.replace(" " + emoji, emoji)
        text = text.replace(emoji + " ", emoji)

    return text.strip()


def format_str_v3(text: str, show_emo=True, show_event=True):
    def get_emo(s):
        return s[-1] if s[-1] in emo_set else None

    def get_event(s):
        return s[0] if s[0] in event_set else None

    text = text.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        text = text.replace(lang, "<|lang|>")
    parts = [format_str_v2(part, show_emo, show_event).strip(" ") for part in text.split("<|lang|>")]
    new_s = " " + parts[0]
    cur_ent_event = get_event(new_s)
    for i in range(1, len(parts)):
        if len(parts[i]) == 0:
            continue
        if get_event(parts[i]) == cur_ent_event and get_event(parts[i]) is not None:
            parts[i] = parts[i][1:]
        cur_ent_event = get_event(parts[i])
        if get_emo(parts[i]) is not None and get_emo(parts[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += parts[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()


def process_audio_simple(waveform, sample_rate, target_fs=None):
    """
    简化的音频处理函数，参考原始api.py的实现
    
    Args:
        waveform: torch.Tensor 音频波形数据
        sample_rate: int 原始采样率
        target_fs: int 目标采样率
        
    Returns:
        torch.Tensor: 处理后的音频数据
    """
    if target_fs is None:
        target_fs = config.TARGET_SAMPLE_RATE
    
    # 重采样到目标采样率（如果需要）
    if sample_rate != target_fs:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_fs)
        waveform = resampler(waveform)
    
    # 多通道转单通道（使用mean(0)，与原始api.py一致）
    if len(waveform.shape) > 1:
        waveform = waveform.mean(0)
    
    return waveform

def model_inference(processed_audio, language, show_emo=True, show_event=True):
    """
    模型推理函数，接受已处理的音频数据
    
    Args:
        processed_audio: torch.Tensor 已处理的音频数据
        language: str 语言代码
        show_emo: bool 是否显示情感
        show_event: bool 是否显示事件
    """
    language = "auto" if len(language) < 1 else language

    if len(processed_audio) == 0:
        raise ValueError("The provided audio is empty.")

    merge_vad = True
    text = model.generate(
        input=processed_audio,
        cache={},
        language=language,
        use_itn=True,
        batch_size_s=0,
        merge_vad=merge_vad,
    )

    text = text[0]["text"]
    text = format_str_v3(text, show_emo, show_event)

    return text


@app.get("/v1/models")
async def models():
    """返回可用模型列表，兼容OpenAI API格式"""
    import time
    timestamp = int(time.time())
    
    # 模型状态检查
    model_status = "available" if model is not None else "unavailable"
    
    return {
        "object": "list",
        "data": [
            {
                "id": "sensevoice-small",
                "object": "model", 
                "created": timestamp,
                "owned_by": "sensevoice",
                "root": "sensevoice-small",
                "parent": None,
                "status": model_status,
                "description": "SenseVoice Small - 多语言语音识别模型",
                "context_length": 30000,  # VAD最大段长度
                "languages": ["auto", "zh", "en", "yue", "ja", "ko"],
                "permission": [
                    {
                        "id": "modelperm-sensevoice-001",
                        "object": "model_permission",
                        "created": timestamp,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ]
            },
            # 添加兼容别名，与原始ID保持一致
            {
                "id": "iic/SenseVoiceSmall",
                "object": "model",
                "created": timestamp,
                "owned_by": "sensevoice",
                "root": "iic/SenseVoiceSmall", 
                "parent": "sensevoice-small",
                "status": model_status,
                "description": "SenseVoice Small (兼容别名)",
                "context_length": 30000,
                "languages": ["auto", "zh", "en", "yue", "ja", "ko"],
                "permission": [
                    {
                        "id": "modelperm-sensevoice-002",
                        "object": "model_permission",
                        "created": timestamp,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ]
            }
        ]
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: Union[UploadFile, None] = File(default=None), 
    model: str = Form(default="sensevoice-small"),
    language: str = Form(default="auto"),
    prompt: str = Form(default=""),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0)
):
    """转录音频文件，兼容OpenAI Whisper API格式"""
    
    # 基础参数验证
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail={"error": {"type": "invalid_request_error", "message": "No audio file provided"}}
        )
    
    # 模型可用性检查
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"type": "service_unavailable", "message": "Model not available, please check server logs"}}
        )

    # 文件格式验证
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"type": "invalid_request_error", "message": "Invalid filename"}}
        )
    
    # 支持的音频格式检查
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"type": "invalid_request_error", "message": f"Unsupported file format: {file_ext}"}}
        )
    
    # 文件大小检查
    file_content = await file.read()
    if len(file_content) > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error": {"type": "invalid_request_error", "message": f"File size exceeds {config.MAX_FILE_SIZE // (1024*1024)}MB limit"}}
        )
    
    # 重置文件指针
    await file.seek(0)

    # 确保临时目录存在
    os.makedirs(config.TMP_DIR, exist_ok=True)
    
    filename = file.filename
    fileobj = file.file
    # 使用时间戳和进程ID生成唯一文件名，避免冲突
    import time
    timestamp = int(time.time() * 1000)  # 毫秒时间戳
    tmp_file = Path(config.TMP_DIR) / f"upload_{timestamp}_{os.getpid()}_{filename}"

    try:
        # 保存上传的文件
        with open(tmp_file, "wb+") as upload_file:
            shutil.copyfileobj(fileobj, upload_file)
        
        # 验证文件是否有效
        if tmp_file.stat().st_size == 0:
            raise ValueError("Uploaded file is empty")
        
        try:
            # 简化的音频处理流程，参考原始api.py
            import gc
            
            # 加载音频文件
            waveform, sample_rate = torchaudio.load(tmp_file)
            
            # 简单验证音频数据
            if waveform.numel() == 0:
                raise ValueError("Audio file contains no data")
            
            # 使用简化的音频处理函数
            processed_audio = process_audio_simple(waveform, sample_rate)
            
            # 清理原始音频数据
            del waveform
            gc.collect()

            # 执行推理
            start_time = time.time()
            result = model_inference(processed_audio, language=language, show_emo=False)
            inference_time = time.time() - start_time
            
            # 记录性能信息
            config.logger.info(f"推理完成 - 文件: {filename}, 时长: {inference_time:.3f}s, 文本长度: {len(result)}")
            
            # 根据response_format返回不同格式
            if response_format.lower() == "text":
                return result  # 纯文本
            elif response_format.lower() == "srt":
                # 简单的SRT格式 (暂未实现时间戳)
                return f"1\n00:00:00,000 --> 00:00:30,000\n{result}\n"
            elif response_format.lower() == "vtt":
                # WebVTT格式
                return f"WEBVTT\n\n1\n00:00:00.000 --> 00:00:30.000\n{result}\n"
            else:
                # 默认JSON格式，兼容OpenAI
                response = {
                    "text": result,
                    "task": "transcribe",
                    "language": language if language != "auto" else "detected",
                    "duration": inference_time
                }
                return response
            
        except Exception as audio_error:
            # 音频处理相关错误
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"type": "invalid_request_error", "message": f"Audio processing failed: {str(audio_error)}"}}
            )
        
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"type": "server_error", "message": "Failed to save uploaded file"}}
        )
    except Exception as e:
        # 其他未预期的错误
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail={"error": {"type": "server_error", "message": f"Internal server error: {str(e)}"}}
        )
    finally:
        # 确保临时文件被删除
        try:
            if tmp_file.exists():
                tmp_file.unlink()
        except Exception as cleanup_error:
            print(f"⚠️ 临时文件清理失败: {cleanup_error}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
