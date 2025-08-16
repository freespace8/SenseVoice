# utils/frontend_mlx_v3.py
# 白箱复现Kaldi策略 - 精确匹配kaldi_native_fbank行为

import numpy as np
from scipy.signal import get_window
from typing import Tuple
import soundfile as sf

print("MLX Frontend V3 (白箱Kaldi复现) Script Initialized.")

class FbankMLX_V3:
    """
    Fbank特征提取器 - 白箱复现kaldi_native_fbank.OnlineFbank
    采用纯NumPy实现，精确匹配Kaldi的每一个处理步骤
    """
    def __init__(
        self, 
        fs: int = 16000, 
        n_mels: int = 80, 
        frame_length_ms: float = 25.0, 
        frame_shift_ms: float = 10.0,
        dither: float = 1.0, 
        window_type: str = 'hamming', 
        **kwargs
    ):
        """
        初始化参数 - 精确复现frontend.py中的FbankOptions设置
        """
        print("初始化FbankMLX_V3 - 白箱Kaldi复现版本...")
        
        self.fs = fs
        self.n_mels = n_mels
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.dither_coeff = dither
        self.window_type = window_type
        
        # 精确复现Kaldi的样本数计算
        self.frame_length_samples = int(frame_length_ms * fs / 1000)
        self.frame_shift_samples = int(frame_shift_ms * fs / 1000)
        
        # n_fft 通常是大于等于 frame_length_samples 的最小2的幂
        self.n_fft = 2**int(np.ceil(np.log2(self.frame_length_samples)))
        
        # 预先生成窗函数
        self.window = get_window(window_type, self.frame_length_samples, fftbins=False)
        
        # 预先生成梅尔滤波器组 (使用librosa，参数经过精确调试)
        import librosa
        self.mel_filters = librosa.filters.mel(
            sr=fs,
            n_fft=self.n_fft,
            n_mels=n_mels,
            fmin=20.0,  # 🔑 关键参数：经调试发现Kaldi使用20Hz作为最低频率
            fmax=fs / 2,
            htk=True,   # Kaldi使用HTK公式
            norm=None   # Kaldi的默认行为是不进行Slaney归一化
        )
        
        print(f"  ✅ 参数配置:")
        print(f"     - 采样率: {fs} Hz")
        print(f"     - 帧长度: {frame_length_ms} ms ({self.frame_length_samples} 样本)")
        print(f"     - 帧移: {frame_shift_ms} ms ({self.frame_shift_samples} 样本)")
        print(f"     - N_FFT: {self.n_fft}")
        print(f"     - 抖动系数: {dither}")
        print(f"     - 窗函数: {window_type}")
        print(f"     - Mel滤波器: {n_mels} bins")

    def __call__(self, waveform: np.ndarray, debug_frames: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        核心Fbank计算 - 严格按照Kaldi的处理顺序
        
        Args:
            waveform: 输入音频波形
            debug_frames: 是否输出调试信息（特别是帧数）
            
        Returns:
            (fbank_features, feat_length)
        """
        # 确保输入是一维float32
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        waveform = waveform.astype(np.float32)
        
        if debug_frames:
            print(f"  📊 输入音频: {len(waveform)} 样本")
        
        # === Kaldi预处理流程 ===
        
        # 1. 缩放至int16范围 (关键步骤)
        waveform = waveform * 32768.0
        
        # 2. 抖动 (Dithering) - 在整数范围内进行
        if self.dither_coeff > 0:
            # 产生与输入信号相同长度的随机数
            dither_noise = np.random.normal(0, self.dither_coeff, len(waveform))
            waveform = waveform + dither_noise
        
        # 3. 预加重 (Pre-emphasis) - Kaldi的默认系数是0.97
        if len(waveform) > 1:
            waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
        
        # === 关键步骤: 分帧 (snip-edges=True) ===
        
        # 4. 精确复现 snip-edges=True 的帧提取
        num_frames = (len(waveform) - self.frame_length_samples) // self.frame_shift_samples + 1
        
        if debug_frames:
            print(f"  📊 snip-edges计算:")
            print(f"     - 预处理后样本数: {len(waveform)}")
            print(f"     - 计算帧数: ({len(waveform)} - {self.frame_length_samples}) // {self.frame_shift_samples} + 1 = {num_frames}")
        
        if num_frames < 1:
            if debug_frames:
                print("  ⚠️  样本不足，返回空特征")
            return np.empty((0, self.n_mels), dtype=np.float32), np.array(0, dtype=np.int32)
        
        # 手动分帧 - 这是与librosa的关键区别
        frames = np.zeros((num_frames, self.frame_length_samples), dtype=np.float32)
        for i in range(num_frames):
            start_index = i * self.frame_shift_samples
            frames[i] = waveform[start_index : start_index + self.frame_length_samples]
        
        if debug_frames:
            print(f"  ✅ 成功提取 {num_frames} 帧")
        
        # === 标准信号处理流程 ===
        
        # 5. 加窗 (Windowing)
        frames *= self.window
        
        # 6. 快速傅里叶变换 (FFT)
        stft_matrix = np.fft.rfft(frames, n=self.n_fft)
        
        # 7. 计算能量谱 (Power Spectrum) - 根据调试发现使用无归一化
        power_spectrum = np.abs(stft_matrix)**2
        
        # 8. 应用梅尔滤波器组
        mel_spectrum = np.dot(power_spectrum, self.mel_filters.T)
        
        # 9. 取对数 (Kaldi的方式)
        # Kaldi的energy_floor=0.0意味着直接替换0值
        log_mel_spectrum = np.log(np.maximum(mel_spectrum, 1e-10))
        
        # 返回结果
        feat_length = np.array(log_mel_spectrum.shape[0], dtype=np.int32)
        return log_mel_spectrum.astype(np.float32), feat_length


def debug_frame_extraction():
    """专门用于验证帧数一致性的调试函数"""
    print("\n" + "="*50)
    print("🔍 帧数一致性验证 (首要目标)")
    print("="*50)
    
    # 导入组件
    try:
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        from utils.frontend import WavFrontend as FbankPyTorch
        print("✅ 成功导入PyTorch组件")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return
    
    # 加载测试音频
    try:
        waveform_np, sr = sf.read('verification_data/en.mp3')
        if sr != 16000:
            import librosa
            waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=16000)
            sr = 16000
        print(f"✅ 加载音频: {len(waveform_np)} 样本, {sr} Hz")
    except Exception as e:
        print(f"❌ 音频加载失败: {e}")
        return
    
    # 配置参数 - 首先禁用抖动以消除随机性
    config = {
        'fs': 16000,
        'window': 'hamming',
        'n_mels': 80,
        'frame_length': 25,
        'frame_shift': 10,
        'dither': 0.0,  # 🔑 关键：禁用抖动
    }
    
    # 初始化两个实现
    print(f"\n🔧 初始化对比实验 (dither=0.0)...")
    fbank_pytorch = FbankPyTorch(**config)
    fbank_v3 = FbankMLX_V3(**config)
    
    # 生成特征并对比帧数
    print(f"\n⚡ 执行特征提取...")
    
    # PyTorch版本
    pt_features, pt_length = fbank_pytorch.fbank(waveform_np)
    print(f"PyTorch结果: {pt_features.shape}, 帧数={pt_length}")
    
    # V3版本 (启用调试)
    v3_features, v3_length = fbank_v3(waveform_np, debug_frames=True)
    print(f"V3结果: {v3_features.shape}, 帧数={v3_length}")
    
    # 🎯 关键验证：帧数是否一致
    print(f"\n🎯 帧数一致性验证:")
    if pt_length == v3_length:
        print(f"   ✅ SUCCESS: 帧数完全一致 ({pt_length} frames)")
        
        # 如果帧数一致，进行数值对比
        if pt_features.shape == v3_features.shape:
            diff = np.abs(pt_features - v3_features)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"\n📊 数值差异分析 (dither=0.0):")
            print(f"   最大绝对差异: {max_diff:.8f}")
            print(f"   平均绝对差异: {mean_diff:.8f}")
            
            # 对比前几个值
            print(f"\n🔬 样本对比:")
            print(f"   PyTorch前5值: {pt_features[0, :5]}")
            print(f"   V3前5值:      {v3_features[0, :5]}")
            
            if max_diff < 1e-4:
                print(f"   🎉 EXCELLENT: 数值差异在目标范围内!")
            elif max_diff < 0.01:
                print(f"   ✅ GOOD: 数值差异较小，需要微调")
            else:
                print(f"   ⚠️  NEEDS WORK: 数值差异较大，需要进一步调试")
        else:
            print(f"   ❌ 形状不匹配: PyTorch {pt_features.shape} vs V3 {v3_features.shape}")
    else:
        print(f"   ❌ CRITICAL: 帧数不一致!")
        print(f"      PyTorch: {pt_length} frames")
        print(f"      V3:      {v3_length} frames")
        print(f"   🔧 需要检查snip-edges实现")


if __name__ == '__main__':
    # 运行帧数验证
    debug_frame_extraction()