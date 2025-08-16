"""
Unified Frontend for both PyTorch and MLX versions
Ensures 100% identical feature extraction
"""

import numpy as np
import kaldi_native_fbank as knf
from typing import Optional, Tuple, Literal


class UnifiedFrontend:
    """
    统一的前端处理器，可同时用于 PyTorch 和 MLX
    保证特征提取100%一致
    """
    
    def __init__(
        self,
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        lfr_m: int = 7,
        lfr_n: int = 6,
        dither: float = 1.0,
        cmvn_file: Optional[str] = None,
        **kwargs,
    ):
        """
        初始化统一的前端处理器
        
        Args:
            fs: 采样率 (Hz)
            window: 窗函数类型
            n_mels: Mel频带数量
            frame_length: 帧长 (毫秒)
            frame_shift: 帧移 (毫秒)
            lfr_m: LFR堆叠帧数
            lfr_n: LFR帧移
            dither: 抖动因子
            cmvn_file: CMVN统计文件路径
        """
        # 创建 Kaldi fbank 选项
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = fs
        opts.frame_opts.dither = dither
        opts.frame_opts.window_type = window
        opts.frame_opts.frame_shift_ms = float(frame_shift)
        opts.frame_opts.frame_length_ms = float(frame_length)
        opts.mel_opts.num_bins = n_mels
        opts.energy_floor = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts
        
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file
        
        # 加载 CMVN
        self.cmvn = None
        if self.cmvn_file:
            self.cmvn = self._load_cmvn(cmvn_file)
    
    def _load_cmvn(self, cmvn_file: str) -> np.ndarray:
        """加载CMVN统计文件"""
        with open(cmvn_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        means_list = []
        vars_list = []
        for i in range(len(lines)):
            line_item = lines[i].split()
            if line_item[0] == "<AddShift>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    add_shift_line = line_item[3 : (len(line_item) - 1)]
                    means_list = list(add_shift_line)
                    continue
            elif line_item[0] == "<Rescale>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    rescale_line = line_item[3 : (len(line_item) - 1)]
                    vars_list = list(rescale_line)
                    continue
        
        means = np.array(means_list).astype(np.float64)
        vars = np.array(vars_list).astype(np.float64)
        cmvn = np.array([means, vars])
        return cmvn
    
    def extract_fbank(self, waveform: np.ndarray) -> np.ndarray:
        """
        提取 Fbank 特征
        
        Args:
            waveform: 音频信号 (numpy array)
            
        Returns:
            Fbank 特征 [frames, n_mels]
        """
        # 缩放波形
        waveform = waveform * (1 << 15)
        
        # 创建 OnlineFbank 实例
        fbank_fn = knf.OnlineFbank(self.opts)
        
        # 接受波形
        fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        
        # 获取帧数
        frames = fbank_fn.num_frames_ready
        
        # 提取特征
        mat = np.empty([frames, self.opts.mel_opts.num_bins])
        for i in range(frames):
            mat[i, :] = fbank_fn.get_frame(i)
        
        # 转换为 float32
        feat = mat.astype(np.float32)
        return feat
    
    def apply_lfr(self, inputs: np.ndarray) -> np.ndarray:
        """
        应用 LFR (Low Frame Rate) 处理
        
        Args:
            inputs: 输入特征 [frames, feature_dim]
            
        Returns:
            LFR 处理后的特征 [lfr_frames, feature_dim * lfr_m]
        """
        if self.lfr_m == 1 and self.lfr_n == 1:
            return inputs
        
        LFR_inputs = []
        
        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / self.lfr_n))
        left_padding = np.tile(inputs[0], ((self.lfr_m - 1) // 2, 1))
        inputs = np.vstack((left_padding, inputs))
        T = T + (self.lfr_m - 1) // 2
        
        for i in range(T_lfr):
            if self.lfr_m <= T - i * self.lfr_n:
                LFR_inputs.append((inputs[i * self.lfr_n : i * self.lfr_n + self.lfr_m]).reshape(1, -1))
            else:
                # 处理最后一个 LFR 帧
                num_padding = self.lfr_m - (T - i * self.lfr_n)
                frame = inputs[i * self.lfr_n :].reshape(-1)
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))
                LFR_inputs.append(frame)
        
        LFR_outputs = np.vstack(LFR_inputs).astype(np.float32)
        return LFR_outputs
    
    def apply_cmvn(self, inputs: np.ndarray) -> np.ndarray:
        """
        应用 CMVN 归一化
        
        Args:
            inputs: 输入特征
            
        Returns:
            归一化后的特征
        """
        if self.cmvn is None:
            return inputs
        
        frame, dim = inputs.shape
        means = np.tile(self.cmvn[0:1, :dim], (frame, 1))
        vars = np.tile(self.cmvn[1:2, :dim], (frame, 1))
        inputs = (inputs + means) * vars
        return inputs.astype(np.float32)
    
    def extract_features(
        self, 
        audio: np.ndarray,
        output_format: Literal["numpy", "pytorch", "mlx"] = "numpy"
    ) -> any:
        """
        完整的特征提取流程
        
        Args:
            audio: 原始音频信号 (numpy array)
            output_format: 输出格式 ("numpy", "pytorch", "mlx")
            
        Returns:
            处理后的特征，格式取决于 output_format:
            - "numpy": numpy array [frames, 560]
            - "pytorch": (features, feature_length) 元组
            - "mlx": mlx array [1, frames, 560]
        """
        # 1. 提取 Fbank 特征
        fbank_features = self.extract_fbank(audio)
        
        # 2. 应用 LFR
        lfr_features = self.apply_lfr(fbank_features)
        
        # 3. 应用 CMVN
        final_features = self.apply_cmvn(lfr_features)
        
        # 4. 根据需要的格式返回
        if output_format == "numpy":
            return final_features
        
        elif output_format == "pytorch":
            # PyTorch 格式: (features, feature_lengths)
            feat_len = np.array(final_features.shape[0]).astype(np.int32)
            return final_features, feat_len
        
        elif output_format == "mlx":
            # MLX 格式: [1, frames, feature_dim]
            import mlx.core as mx
            features_mx = mx.array(final_features, dtype=mx.float32)
            features_mx = mx.expand_dims(features_mx, 0)
            return features_mx
        
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
    
    def __call__(self, audio: np.ndarray, output_format: str = "numpy"):
        """快捷调用方法"""
        return self.extract_features(audio, output_format)


def create_frontend_mlx(
    cmvn_file: Optional[str] = "/Users/taylor/.cache/modelscope/hub/models/iic/SenseVoiceSmall/am.mvn",
    **kwargs
) -> UnifiedFrontend:
    """
    创建统一的前端处理器
    
    Args:
        cmvn_file: CMVN文件路径
        **kwargs: 其他配置参数
        
    Returns:
        UnifiedFrontend 实例
    """
    default_config = {
        'fs': 16000,
        'n_mels': 80,
        'frame_length': 25,
        'frame_shift': 10,
        'lfr_m': 7,
        'lfr_n': 6,
        'window': 'hamming',
        'dither': 1.0,
        'cmvn_file': cmvn_file,
    }
    default_config.update(kwargs)
    return UnifiedFrontend(**default_config)


# 测试代码
if __name__ == "__main__":
    import librosa
    
    print("🧪 测试统一前端...")
    
    # 加载测试音频
    audio, sr = librosa.load("examples/ja.mp3", sr=16000, mono=True)
    print(f"音频形状: {audio.shape}")
    
    # 创建统一前端
    frontend = create_unified_frontend()
    
    # 测试不同输出格式
    print("\n1. NumPy 格式:")
    features_np = frontend(audio, output_format="numpy")
    print(f"   输出形状: {features_np.shape}")
    print(f"   数据类型: {features_np.dtype}")
    
    print("\n2. PyTorch 格式:")
    features_pt, feat_len = frontend(audio, output_format="pytorch")
    print(f"   特征形状: {features_pt.shape}")
    print(f"   特征长度: {feat_len}")
    
    print("\n3. MLX 格式:")
    features_mlx = frontend(audio, output_format="mlx")
    print(f"   输出形状: {features_mlx.shape}")
    print(f"   数据类型: {features_mlx.dtype}")
    
    # 验证三种格式的数值一致性
    import mlx.core as mx
    features_mlx_np = np.array(features_mlx.squeeze(0))
    
    print("\n✅ 数值一致性检查:")
    # PyTorch 返回的是元组，第一个元素是特征
    print(f"   NumPy vs PyTorch 特征数组: {np.array_equal(features_np, features_pt)}")
    print(f"   NumPy vs MLX (allclose):    {np.allclose(features_np, features_mlx_np, rtol=1e-5, atol=1e-6)}")
    
    # 详细检查
    diff = np.abs(features_np - features_mlx_np)
    print(f"\n   详细差异分析:")
    print(f"   最大差异: {diff.max():.10f}")
    print(f"   平均差异: {diff.mean():.10f}")
    print(f"   差异位置: {np.unravel_index(np.argmax(diff), diff.shape)}")
    
    print("\n✨ 统一前端测试完成！")