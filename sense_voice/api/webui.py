"""
SenseVoice Web UI with Unified Backend Support
Interactive web interface supporting both PyTorch and MLX backends
"""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import gradio as gr
import numpy as np
import torch
import torchaudio
import librosa

from engine import get_engine, list_available_backends, get_backend_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TARGET_FS = 16000

# Emoji mappings for rich text output
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
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
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
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷"}


class WebUIConfig:
    """Configuration for the Web UI"""
    def __init__(self):
        self.backend: str = "pytorch"
        self.model_path: str = None
        self.tokenizer_path: Optional[str] = None
        self.device: Optional[str] = None
        self.use_vad: bool = True
        self.vad_model: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        self.max_single_segment_time: int = 30000
        self.share: bool = False
        self.server_name: str = "0.0.0.0"
        self.server_port: int = 7860


# Global variables
config = WebUIConfig()
inference_engine = None
vad_model = None


def format_str(s):
    """Format string with emoji replacements"""
    for sptk in emoji_dict:
        s = s.replace(sptk, emoji_dict[sptk])
    return s


def format_str_v2(s):
    """Advanced formatting with emotion and event detection"""
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = s.count(sptk)
        s = s.replace(sptk, "")
    
    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    
    for e in event_dict:
        if sptk_dict[e] > 0:
            s = event_dict[e] + s
    
    s = s + emo_dict[emo]
    
    for emoji in emo_set.union(event_set):
        s = s.replace(" " + emoji, emoji)
        s = s.replace(emoji + " ", emoji)
    
    return s.strip()


def format_str_v3(s):
    """Ultimate formatting with language handling"""
    def get_emo(s):
        return s[-1] if s[-1] in emo_set else None
    
    def get_event(s):
        return s[0] if s[0] in event_set else None
    
    s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        s = s.replace(lang, "<|lang|>")
    
    s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
    new_s = " " + s_list[0]
    cur_ent_event = get_event(new_s)
    
    for i in range(1, len(s_list)):
        if len(s_list[i]) == 0:
            continue
        if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
            s_list[i] = s_list[i][1:]
        cur_ent_event = get_event(s_list[i])
        if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += s_list[i].strip().lstrip()
    
    new_s = new_s.replace("The.", " ")
    return new_s.strip()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SenseVoice Web UI")
    
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
    
    # VAD configuration
    parser.add_argument(
        "--no_vad",
        action="store_true",
        help="Disable VAD (Voice Activity Detection)"
    )
    
    parser.add_argument(
        "--vad_model",
        type=str,
        default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        help="VAD model to use"
    )
    
    # Server configuration
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link"
    )
    
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server name for Gradio"
    )
    
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port for Gradio"
    )
    
    return parser.parse_args()


def initialize_engine(args):
    """Initialize the inference engine based on configuration"""
    global inference_engine, vad_model, config
    
    # Update config from arguments
    config.backend = args.backend
    config.model_path = args.model_path
    config.tokenizer_path = args.tokenizer_path
    config.device = args.device or os.getenv("SENSEVOICE_DEVICE", None)
    config.use_vad = not args.no_vad
    config.vad_model = args.vad_model
    config.share = args.share
    config.server_name = args.server_name
    config.server_port = args.server_port
    
    # Log available backends
    available_backends = list_available_backends()
    logger.info(f"Available backends: {available_backends}")
    
    # Validate backend availability
    if config.backend not in available_backends:
        raise ValueError(f"Backend '{config.backend}' is not available. Available: {available_backends}")
    
    # Initialize VAD if enabled and backend is PyTorch
    if config.use_vad and config.backend == "pytorch":
        try:
            from funasr import AutoModel
            logger.info(f"Initializing VAD model: {config.vad_model}")
            vad_model = AutoModel(
                model=config.vad_model,
                trust_remote_code=True
            )
            logger.info("VAD model initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize VAD model: {e}")
            config.use_vad = False
    
    # Prepare engine kwargs
    engine_kwargs = {
        'model_path': config.model_path
    }
    
    if config.backend == 'pytorch':
        if config.device:
            engine_kwargs['device'] = config.device
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


def process_audio_for_vad(audio_data: np.ndarray, sample_rate: int) -> list:
    """
    Process audio through VAD to get segments
    
    Args:
        audio_data: Audio waveform
        sample_rate: Sample rate
    
    Returns:
        List of audio segments
    """
    if not vad_model or not config.use_vad:
        return [audio_data]
    
    try:
        # VAD expects specific format
        segments = vad_model.generate(
            input=audio_data,
            cache={},
            batch_size_s=config.max_single_segment_time,
            merge_vad=True
        )
        
        # Extract audio segments
        audio_segments = []
        for seg in segments:
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            audio_segments.append(audio_data[start_sample:end_sample])
        
        return audio_segments if audio_segments else [audio_data]
    
    except Exception as e:
        logger.warning(f"VAD processing failed: {e}")
        return [audio_data]


def model_inference(input_wav: Union[Tuple[int, np.ndarray], str], language: str, fs: int = 16000) -> str:
    """
    Main inference function for the Web UI
    
    Args:
        input_wav: Audio input (tuple from Gradio or file path)
        language: Language selection
        fs: Target sample rate
    
    Returns:
        Transcribed and formatted text
    """
    if not inference_engine:
        return "Error: Inference engine not initialized. Please restart the application."
    
    language_abbr = {
        "auto": "auto",
        "zh": "zh",
        "en": "en",
        "yue": "yue",
        "ja": "ja",
        "ko": "ko",
        "nospeech": "nospeech"
    }
    
    language = "auto" if not language or len(language) < 1 else language
    selected_language = language_abbr.get(language, "auto")
    
    try:
        # Process input audio
        if isinstance(input_wav, tuple):
            fs, input_wav = input_wav
            input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
            if len(input_wav.shape) > 1:
                input_wav = input_wav.mean(-1)
            if fs != TARGET_FS:
                logger.info(f"Resampling audio from {fs}Hz to {TARGET_FS}Hz")
                resampler = torchaudio.transforms.Resample(fs, TARGET_FS)
                input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
                input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
        elif isinstance(input_wav, str):
            # Load from file path
            input_wav, fs = librosa.load(input_wav, sr=TARGET_FS)
        
        # Apply VAD if enabled (PyTorch backend only)
        if config.use_vad and config.backend == "pytorch" and vad_model:
            audio_segments = process_audio_for_vad(input_wav, TARGET_FS)
            
            # Process each segment and combine results
            all_texts = []
            for segment in audio_segments:
                if len(segment) < TARGET_FS * 0.1:  # Skip very short segments
                    continue
                
                result = inference_engine.transcribe(
                    audio_waveform=segment,
                    sample_rate=TARGET_FS,
                    language=selected_language if selected_language != "auto" else None
                )
                
                if result and result.get('text'):
                    all_texts.append(result['text'])
            
            # Combine all segment texts
            text = " ".join(all_texts) if all_texts else ""
        else:
            # Direct inference without VAD
            result = inference_engine.transcribe(
                audio_waveform=input_wav,
                sample_rate=TARGET_FS,
                language=selected_language if selected_language != "auto" else None
            )
            text = result.get('text', '') if result else ""
        
        # Format the output text with emojis
        text = format_str_v3(text)
        
        logger.info(f"Transcription completed: {text[:100]}...")
        return text
    
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return f"Error during transcription: {str(e)}"


# Audio examples for the UI
audio_examples = [
    ["example/zh.mp3", "zh"],
    ["example/yue.mp3", "yue"],
    ["example/en.mp3", "en"],
    ["example/ja.mp3", "ja"],
    ["example/ko.mp3", "ko"],
    ["example/emo_1.wav", "auto"],
    ["example/emo_2.wav", "auto"],
    ["example/emo_3.wav", "auto"],
    ["example/rich_1.wav", "auto"],
    ["example/rich_2.wav", "auto"],
    ["example/longwav_1.wav", "auto"],
    ["example/longwav_2.wav", "auto"],
    ["example/longwav_3.wav", "auto"],
]

# HTML content for the UI
html_content = f"""
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">Voice Understanding Model: SenseVoice-Small</h2>
    <p style="font-size: 18px;margin-left: 20px;">SenseVoice-Small is an encoder-only speech foundation model designed for rapid voice understanding. It encompasses a variety of features including automatic speech recognition (ASR), spoken language identification (LID), speech emotion recognition (SER), and acoustic event detection (AED).</p>
    <p style="font-size: 18px;margin-left: 20px;"><strong>Current Backend:</strong> <span id="backend-info">Loading...</span></p>
    <h2 style="font-size: 22px;margin-left: 0px;">Features</h2>
    <ul style="font-size: 18px;margin-left: 20px;">
        <li>Multilingual recognition: Chinese, English, Cantonese, Japanese, and Korean</li>
        <li>Emotion detection: 😊 happy, 😡 angry/exciting, 😔 sad</li>
        <li>Event detection: 😀 laughter, 🎼 music, 👏 applause, 🤧 cough&sneeze, 😭 cry</li>
        <li>Ultra-low latency: 7x faster than Whisper-small, 17x faster than Whisper-large</li>
        <li>Dual backend support: PyTorch (CUDA/CPU) and MLX (Apple Silicon optimized)</li>
    </ul>
    <h2 style="font-size: 22px;margin-left: 0px;">Usage</h2>
    <p style="font-size: 18px;margin-left: 20px;">Upload an audio file or input through a microphone, then select the language. The audio will be transcribed with associated emotions and sound events. Event labels appear at the front of text, emotions at the back.</p>
    <p style="font-size: 18px;margin-left: 20px;">Recommended audio duration: &lt; 30 seconds. For longer audio, local deployment is recommended.</p>
    <h2 style="font-size: 22px;margin-left: 0px;">Repository</h2>
    <p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">SenseVoice</a>: Multilingual speech understanding model</p>
    <p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/modelscope/FunASR" target="_blank">FunASR</a>: Fundamental speech recognition toolkit</p>
</div>
"""


def launch():
    """Launch the Gradio web interface"""
    with gr.Blocks(theme=gr.themes.Soft(), title="SenseVoice Web UI") as demo:
        # Update HTML with backend info
        backend_html = html_content.replace(
            '<span id="backend-info">Loading...</span>',
            f'<span id="backend-info">{config.backend.upper()} {"(VAD enabled)" if config.use_vad else ""}</span>'
        )
        gr.HTML(backend_html)
        
        with gr.Row():
            with gr.Column():
                audio_inputs = gr.Audio(label="Upload audio or use the microphone")
                
                with gr.Accordion("Configuration", open=True):
                    language_inputs = gr.Dropdown(
                        choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
                        value="auto",
                        label="Language"
                    )
                    
                    # Show backend info
                    gr.Markdown(f"**Backend:** {config.backend} | **Model:** {config.model_path}")
                
                fn_button = gr.Button("Start Transcription", variant="primary")
                text_outputs = gr.Textbox(label="Results", lines=5)
            
            with gr.Column():
                gr.Examples(
                    examples=audio_examples,
                    inputs=[audio_inputs, language_inputs],
                    examples_per_page=10
                )
        
        fn_button.click(
            model_inference,
            inputs=[audio_inputs, language_inputs],
            outputs=text_outputs
        )
    
    # Launch the interface
    demo.launch(
        share=config.share,
        server_name=config.server_name,
        server_port=config.server_port
    )


def main():
    """Main entry point for the Web UI"""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize engine
    initialize_engine(args)
    
    # Launch the web interface
    logger.info(f"Launching SenseVoice Web UI with {config.backend} backend...")
    launch()


if __name__ == "__main__":
    main()