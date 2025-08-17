#!/usr/bin/env python3
"""
测试 SenseVoice MLX API
兼容 OpenAI Whisper API 的客户端测试
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Optional


class MLXAPIClient:
    """SenseVoice MLX API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """初始化客户端
        
        Args:
            base_url: API 基础 URL
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> dict:
        """健康检查"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def stats(self) -> dict:
        """获取统计信息"""
        response = self.session.get(f"{self.base_url}/stats")
        return response.json()
    
    def transcribe(
        self,
        audio_file: str,
        language: str = "auto",
        response_format: str = "json"
    ) -> dict:
        """转录音频文件
        
        Args:
            audio_file: 音频文件路径
            language: 语言代码
            response_format: 响应格式
            
        Returns:
            转录结果
        """
        with open(audio_file, 'rb') as f:
            files = {'file': (os.path.basename(audio_file), f)}
            data = {
                'language': language,
                'response_format': response_format
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/audio/transcriptions",
                files=files,
                data=data
            )
            
            if response.status_code != 200:
                raise Exception(f"API Error: {response.status_code} - {response.text}")
            
            if response_format == "text":
                return {"text": response.text}
            else:
                return response.json()
    
    def benchmark(self, audio_file: str, iterations: int = 5) -> dict:
        """运行基准测试
        
        Args:
            audio_file: 测试音频文件
            iterations: 迭代次数
            
        Returns:
            基准测试结果
        """
        with open(audio_file, 'rb') as f:
            files = {'file': (os.path.basename(audio_file), f)}
            data = {'iterations': iterations}
            
            response = self.session.post(
                f"{self.base_url}/v1/benchmark",
                files=files,
                data=data
            )
            
            if response.status_code != 200:
                raise Exception(f"API Error: {response.status_code} - {response.text}")
            
            return response.json()


def test_api(base_url: str = "http://localhost:8001"):
    """测试 API 功能"""
    
    print("\n" + "=" * 60)
    print("🧪 SenseVoice MLX API 测试")
    print("=" * 60)
    
    # 创建客户端
    client = MLXAPIClient(base_url)
    
    # 1. 健康检查
    print("\n📋 健康检查...")
    try:
        health = client.health_check()
        print(f"   状态: {health['status']}")
        print(f"   模型加载: {health['model_loaded']}")
        print(f"   运行时间: {health['uptime_seconds']:.1f}秒")
    except Exception as e:
        print(f"   ❌ 健康检查失败: {e}")
        print("   请确保 API 服务器正在运行")
        return
    
    # 2. 统计信息
    print("\n📊 统计信息...")
    try:
        stats = client.stats()
        print(f"   模型类型: {stats['model_type']}")
        print(f"   加速类型: {stats['acceleration']}")
    except Exception as e:
        print(f"   ⚠️  统计信息获取失败: {e}")
    
    # 3. 测试转录
    examples_dir = "/Users/taylor/Documents/code/SenseVoice/examples"
    if os.path.exists(examples_dir):
        audio_files = [
            f for f in os.listdir(examples_dir)
            if f.endswith('.mp3')
        ]
        
        if audio_files:
            print("\n🎙️  测试转录功能...")
            
            # 测试不同格式
            test_file = os.path.join(examples_dir, audio_files[0])
            
            # JSON 格式
            print(f"\n   测试文件: {audio_files[0]}")
            print("   格式: JSON")
            try:
                start_time = time.time()
                result = client.transcribe(test_file, response_format="json")
                elapsed = time.time() - start_time
                
                print(f"   ✅ 转录成功 (耗时: {elapsed:.3f}秒)")
                print(f"   文本: {result['text'][:100]}...")
                print(f"   语言: {result.get('language', 'unknown')}")
            except Exception as e:
                print(f"   ❌ 转录失败: {e}")
            
            # 纯文本格式
            print("\n   格式: TEXT")
            try:
                result = client.transcribe(test_file, response_format="text")
                print(f"   ✅ 文本: {result['text'][:100]}...")
            except Exception as e:
                print(f"   ❌ 转录失败: {e}")
            
            # Verbose JSON 格式
            print("\n   格式: VERBOSE_JSON")
            try:
                result = client.transcribe(test_file, response_format="verbose_json")
                print(f"   ✅ 段落数: {len(result.get('segments', []))}")
                print(f"   推理时间: {result.get('inference_time', 0):.3f}秒")
            except Exception as e:
                print(f"   ❌ 转录失败: {e}")
            
            # 4. 基准测试
            print("\n⚡ 性能基准测试...")
            try:
                benchmark = client.benchmark(test_file, iterations=3)
                print(f"   平均时间: {benchmark['mean_time']:.3f}秒")
                print(f"   最快时间: {benchmark['min_time']:.3f}秒")
                print(f"   最慢时间: {benchmark['max_time']:.3f}秒")
                print(f"   吞吐量: {benchmark['throughput']:.2f} 文件/秒")
            except Exception as e:
                print(f"   ⚠️  基准测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)


def compare_with_openai():
    """与 OpenAI API 格式对比测试"""
    
    print("\n" + "=" * 60)
    print("🔄 OpenAI API 兼容性测试")
    print("=" * 60)
    
    # 测试与 OpenAI 客户端的兼容性
    try:
        from openai import OpenAI
        
        # 创建客户端，指向我们的 API
        client = OpenAI(
            api_key="not-needed",  # 我们的 API 不需要密钥
            base_url="http://localhost:8001/v1"
        )
        
        # 测试转录
        examples_dir = "/Users/taylor/Documents/code/SenseVoice/examples"
        if os.path.exists(examples_dir):
            audio_files = [
                f for f in os.listdir(examples_dir)
                if f.endswith('.mp3')
            ]
            
            if audio_files:
                test_file = os.path.join(examples_dir, audio_files[0])
                
                print(f"\n使用 OpenAI 客户端测试...")
                print(f"文件: {audio_files[0]}")
                
                with open(test_file, 'rb') as audio:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",  # 兼容性参数
                        file=audio
                    )
                
                print(f"✅ 兼容性测试成功")
                print(f"文本: {transcript.text[:100]}...")
    
    except ImportError:
        print("⚠️  OpenAI 客户端未安装，跳过兼容性测试")
        print("   运行: pip install openai 来安装")
    except Exception as e:
        print(f"❌ 兼容性测试失败: {e}")
    
    print("=" * 60)


if __name__ == "__main__":
    # 检查命令行参数
    base_url = "http://localhost:8001"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"📍 API 地址: {base_url}")
    
    # 运行测试
    test_api(base_url)
    
    # 兼容性测试
    compare_with_openai()