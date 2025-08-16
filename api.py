"""
SenseVoice API Server with Unified Backend Support
Supports both PyTorch and MLX backends through unified interface
"""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
from io import BytesIO

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated

import numpy as np
import soundfile as sf
import torchaudio

from engine import get_engine, list_available_backends, get_backend_info
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TARGET_FS = 16000
REGEX_CLEAN = r"<\|.*\|>"


class Language(str, Enum):
    """Supported languages for transcription"""
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


class APIConfig:
    """Configuration for the API server"""
    def __init__(self):
        self.backend: str = "pytorch"
        self.model_path: str = None
        self.tokenizer_path: Optional[str] = None
        self.device: Optional[str] = None
        self.host: str = "0.0.0.0"
        self.port: int = 50000
        self.max_file_size: int = 100 * 1024 * 1024  # 100MB
        self.supported_formats: List[str] = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']


# Global variables
app = FastAPI(title="SenseVoice API", version="2.0.0")
config = APIConfig()
inference_engine = None


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SenseVoice API Server")
    
    # Backend selection
    parser.add_argument(
        "--backend", 
        type=str, 
        default="pytorch",
        choices=["pytorch", "mlx"],
        help="Inference backend to use"
    )
    
    # Model paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="iic/SenseVoiceSmall",
        help="Path to model (directory for PyTorch, .safetensors for MLX)"
    )
    
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer.json (required for MLX backend)"
    )
    
    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for PyTorch backend (cuda:0, cpu, etc.)"
    )
    
    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=50000,
        help="Port to bind the server to"
    )
    
    return parser.parse_args()


def initialize_engine(args):
    """Initialize the inference engine based on configuration"""
    global inference_engine, config
    
    # Update config from arguments
    config.backend = args.backend
    config.model_path = args.model_path
    config.tokenizer_path = args.tokenizer_path
    config.device = args.device or os.getenv("SENSEVOICE_DEVICE", None)
    config.host = args.host
    config.port = args.port
    
    # Log available backends
    available_backends = list_available_backends()
    logger.info(f"Available backends: {available_backends}")
    
    # Validate backend availability
    if config.backend not in available_backends:
        raise ValueError(f"Backend '{config.backend}' is not available. Available: {available_backends}")
    
    # Prepare engine kwargs
    engine_kwargs = {
        'model_path': config.model_path
    }
    
    if config.backend == 'pytorch':
        if config.device:
            engine_kwargs['device'] = config.device
        # Add frontend configuration if needed
        engine_kwargs['frontend_conf'] = {}
    
    elif config.backend == 'mlx':
        if not config.tokenizer_path:
            # Try to auto-detect tokenizer path
            model_dir = Path(config.model_path).parent
            tokenizer_candidates = [
                model_dir / "tokenizer.json",
                Path("SenseVoice-Small") / "tokenizer.json",
                Path("iic/SenseVoiceSmall") / "tokenizer.json"
            ]
            
            for candidate in tokenizer_candidates:
                if candidate.exists():
                    config.tokenizer_path = str(candidate)
                    logger.info(f"Auto-detected tokenizer at: {config.tokenizer_path}")
                    break
            
            if not config.tokenizer_path:
                raise ValueError("MLX backend requires tokenizer_path. Please specify with --tokenizer_path")
        
        engine_kwargs['tokenizer_path'] = config.tokenizer_path
    
    # Initialize the engine
    logger.info(f"Initializing {config.backend} engine...")
    logger.info(f"Model path: {config.model_path}")
    
    try:
        inference_engine = get_engine(config.backend, **engine_kwargs)
        logger.info(f"Successfully initialized {config.backend} engine")
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        raise


async def process_audio_file(file: UploadFile) -> np.ndarray:
    """
    Process uploaded audio file and convert to numpy array
    
    Args:
        file: Uploaded audio file
    
    Returns:
        Numpy array of audio samples at target sample rate
    """
    # Read file content
    file_content = await file.read()
    file_io = BytesIO(file_content)
    
    # Load audio based on file extension
    file_ext = Path(file.filename).suffix.lower()
    
    try:
        if file_ext in ['.wav', '.flac', '.ogg']:
            # Use soundfile for these formats
            audio_data, sample_rate = sf.read(file_io)
        else:
            # Use torchaudio for other formats
            audio_data, sample_rate = torchaudio.load(file_io)
            audio_data = audio_data.numpy()
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=0 if audio_data.shape[0] > audio_data.shape[1] else 1)
        
        # Resample if necessary
        if sample_rate != TARGET_FS:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=TARGET_FS)
        
        return audio_data
    
    except Exception as e:
        logger.error(f"Error processing audio file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process audio file: {str(e)}")


def postprocess_transcription(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Postprocess transcription result
    
    Args:
        result: Raw transcription result from engine
    
    Returns:
        Processed result with additional fields
    """
    text = result.get('text', '')
    
    # Clean text
    clean_text = re.sub(REGEX_CLEAN, "", text, 0, re.MULTILINE)
    
    # Rich text processing
    rich_text = rich_transcription_postprocess(text)
    
    return {
        'raw_text': text,
        'clean_text': clean_text,
        'text': rich_text,
        'language': result.get('language', 'unknown'),
        'timestamps': result.get('timestamps', None)
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    backend_info = get_backend_info(config.backend) if config.backend else "Not initialized"
    
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>SenseVoice API Server</title>
        </head>
        <body>
            <h1>SenseVoice API Server v2.0</h1>
            <p><strong>Current Backend:</strong> {config.backend}</p>
            <p><strong>Model Path:</strong> {config.model_path}</p>
            <p><strong>Status:</strong> {'Ready' if inference_engine else 'Not initialized'}</p>
            <hr>
            <a href='./docs'>API Documentation</a>
        </body>
    </html>
    """


@app.get("/api/v1/info")
async def get_info():
    """Get server information and status"""
    return {
        "version": "2.0.0",
        "backend": config.backend,
        "model_path": config.model_path,
        "available_backends": list_available_backends(),
        "supported_languages": [lang.value for lang in Language],
        "supported_formats": config.supported_formats,
        "max_file_size": config.max_file_size,
        "status": "ready" if inference_engine else "not_initialized"
    }


@app.post("/api/v1/asr")
async def transcribe_audio(
    files: Annotated[List[UploadFile], File(description="Audio files (wav, mp3, m4a, flac, ogg)")],
    keys: Annotated[Optional[str], Form(description="Names for each audio, comma-separated")] = None,
    lang: Annotated[Language, Form(description="Language of audio content")] = Language.auto,
):
    """
    Transcribe audio files
    
    Args:
        files: List of audio files to transcribe
        keys: Optional names for each audio file
        lang: Language hint for transcription
    
    Returns:
        Transcription results for each file
    """
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    # Validate file count
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Parse keys
    if keys:
        key_list = keys.split(",")
        if len(key_list) != len(files):
            raise HTTPException(status_code=400, detail="Number of keys must match number of files")
    else:
        key_list = [f.filename for f in files]
    
    # Process each file
    results = []
    for idx, file in enumerate(files):
        try:
            # Validate file size
            file.file.seek(0, 2)  # Seek to end
            file_size = file.file.tell()
            file.file.seek(0)  # Reset to beginning
            
            if file_size > config.max_file_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} exceeds maximum size of {config.max_file_size} bytes"
                )
            
            # Validate file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in config.supported_formats:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {file_ext}. Supported: {config.supported_formats}"
                )
            
            # Process audio
            logger.info(f"Processing file {idx + 1}/{len(files)}: {file.filename}")
            audio_data = await process_audio_file(file)
            
            # Transcribe
            result = inference_engine.transcribe(
                audio_waveform=audio_data,
                sample_rate=TARGET_FS,
                language=lang.value if lang != Language.auto else None
            )
            
            # Postprocess
            processed_result = postprocess_transcription(result)
            processed_result['key'] = key_list[idx]
            processed_result['filename'] = file.filename
            
            results.append(processed_result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            results.append({
                'key': key_list[idx],
                'filename': file.filename,
                'error': str(e),
                'text': '',
                'clean_text': '',
                'raw_text': ''
            })
    
    return {"result": results}


@app.post("/api/v1/detect_language")
async def detect_language(
    file: Annotated[UploadFile, File(description="Audio file for language detection")]
):
    """
    Detect the language of an audio file
    
    Args:
        file: Audio file to analyze
    
    Returns:
        Detected language code
    """
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Process audio
        audio_data = await process_audio_file(file)
        
        # Detect language
        detected_lang = inference_engine.detect_language(
            audio_waveform=audio_data,
            sample_rate=TARGET_FS
        )
        
        return {
            "filename": file.filename,
            "detected_language": detected_lang
        }
    
    except Exception as e:
        logger.error(f"Error detecting language for file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup if running as module"""
    logger.info("SenseVoice API Server starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("SenseVoice API Server shutting down...")


def main():
    """Main entry point for the API server"""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize engine
    initialize_engine(args)
    
    # Start server
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()