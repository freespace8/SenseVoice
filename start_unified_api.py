#!/usr/bin/env python3
"""
SenseVoice 统一API启动脚本
支持MLX优先，自动回退到PyTorch
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from sense_voice.core.inference import create_transcriber
from sense_voice.core.config import config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SenseVoice 统一推理系统")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "mlx", "pytorch"],
        default="auto",
        help="推理后端 (auto=MLX优先，自动回退)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="运行测试"
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="要转录的音频文件"
    )
    
    args = parser.parse_args()
    
    # 创建转录器
    force_backend = None if args.backend == "auto" else args.backend
    transcriber = create_transcriber(force_backend=force_backend)
    
    # 显示信息
    info = transcriber.get_info()
    print("\n" + "="*60)
    print("🚀 SenseVoice 统一推理系统")
    print("="*60)
    print(f"✅ 当前后端: {info['backend']}")
    print(f"📍 MLX可用: {info['mlx_available']}")
    print(f"📍 PyTorch可用: {info['pytorch_available']}")
    print("="*60)
    
    # 如果指定了音频文件，进行转录
    if args.audio:
        print(f"\n🎵 转录音频: {args.audio}")
        result = transcriber.transcribe(args.audio)
        print(f"📝 结果: {result['text']}")
        print(f"⏱️ 耗时: {result.get('total_time', 0):.2f}秒")
        if 'rtf' in result:
            print(f"⚡ RTF: {result['rtf']:.2f}x")
    
    # 如果是测试模式
    if args.test:
        print("\n🧪 运行测试...")
        test_audio = Path(__file__).parent / "examples" / "zh.mp3"
        if test_audio.exists():
            result = transcriber.transcribe(str(test_audio))
            print(f"✅ 测试成功!")
            print(f"   文本: {result['text'][:50]}...")
        else:
            print("⚠️ 测试音频不存在")

if __name__ == "__main__":
    main()