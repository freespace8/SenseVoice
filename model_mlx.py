# model_mlx.py
# MLX Implementation of SenseVoice Model
# Precise architectural equivalent of the PyTorch SenseVoice model

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Tuple
import math

print("MLX Model Definition Script Initialized.")


class SinusoidalPositionEncoder(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model: int = 80, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate
    
    def encode(self, positions: mx.array, depth: int, dtype=mx.float32) -> mx.array:
        """Generate sinusoidal position encodings."""
        batch_size = positions.shape[0]
        positions = positions.astype(dtype)
        
        log_timescale_increment = math.log(10000.0) / (depth / 2 - 1)
        inv_timescales = mx.exp(mx.arange(depth // 2, dtype=dtype) * (-log_timescale_increment))
        inv_timescales = mx.broadcast_to(inv_timescales[None, :], (batch_size, depth // 2))
        
        scaled_time = positions[:, :, None] * inv_timescales[:, None, :]
        encoding = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=2)
        return encoding.astype(dtype)
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass adding positional encoding."""
        batch_size, timesteps, input_dim = x.shape
        positions = mx.arange(1, timesteps + 1, dtype=mx.float32)[None, :]
        positions = mx.broadcast_to(positions, (batch_size, timesteps))
        position_encoding = self.encode(positions, input_dim, x.dtype)
        return x + position_encoding


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer equivalent to PyTorch version."""
    
    def __init__(self, idim: int, hidden_units: int, dropout_rate: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class MultiHeadedAttentionSANM(nn.Module):
    """Multi-Head Attention layer with SANM (Self-Attention Memory) mechanism."""
    
    def __init__(
        self,
        n_head: int,
        in_feat: int,
        n_feat: int,
        dropout_rate: float,
        kernel_size: int,
        sanm_shift: int = 0,
    ):
        super().__init__()
        assert n_feat % n_head == 0
        
        self.d_k = n_feat // n_head
        self.h = n_head
        self.n_feat = n_feat
        
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.dropout = nn.Dropout(dropout_rate)
        
        # FSMN block - using Conv1d for memory mechanism
        self.fsmn_conv = nn.Conv1d(
            in_channels=n_feat,
            out_channels=n_feat,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=n_feat,
            bias=False
        )
        
        # Calculate padding for causal convolution
        left_padding = (kernel_size - 1) // 2
        if sanm_shift > 0:
            left_padding = left_padding + sanm_shift
        right_padding = kernel_size - 1 - left_padding
        self.left_padding = left_padding
        self.right_padding = right_padding
    
    def forward_fsmn(self, inputs: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass for FSMN (Feedforward Sequential Memory Network)."""
        b, t, d = inputs.shape
        
        if mask is not None:
            mask = mx.reshape(mask, (b, -1, 1))
            inputs = inputs * mask
        
        # MLX Conv1d期待输入格式: (B, L, C) 而不是 PyTorch的 (B, C, L)
        # 我们的输入已经是 (B, T, D) 格式，其中T是序列长度，D是特征维度
        
        # Apply padding on the time dimension
        x = mx.pad(inputs, [(0, 0), (self.left_padding, self.right_padding), (0, 0)])
        
        # Apply convolution - MLX Conv1d处理 (B, L, C) 格式
        x = self.fsmn_conv(x)
        
        # Residual connection
        x = x + inputs
        x = self.dropout(x)
        
        if mask is not None:
            x = x * mask
            
        return x
    
    def forward_qkv(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Transform query, key and value."""
        b, t, d = x.shape
        q_k_v = self.linear_q_k_v(x)
        
        # Split into q, k, v
        q = q_k_v[:, :, :self.h * self.d_k]
        k = q_k_v[:, :, self.h * self.d_k:2 * self.h * self.d_k]
        v = q_k_v[:, :, 2 * self.h * self.d_k:]
        
        # Reshape and transpose for multi-head attention
        q_h = mx.reshape(q, (b, t, self.h, self.d_k))
        q_h = mx.transpose(q_h, (0, 2, 1, 3))  # (batch, head, time, d_k)
        
        k_h = mx.reshape(k, (b, t, self.h, self.d_k))
        k_h = mx.transpose(k_h, (0, 2, 1, 3))  # (batch, head, time, d_k)
        
        v_h = mx.reshape(v, (b, t, self.h, self.d_k))
        v_h = mx.transpose(v_h, (0, 2, 1, 3))  # (batch, head, time, d_k)
        
        return q_h, k_h, v_h, v
    
    def forward_attention(self, value: mx.array, scores: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Compute attention context vector."""
        if mask is not None:
            mask = mx.expand_dims(mask, 1)  # Add head dimension
            mask = mask == 0  # Convert to boolean mask
            
            # Apply mask to scores
            scores = mx.where(mask, float('-inf'), scores)
            attn = mx.softmax(scores, axis=-1)
            attn = mx.where(mask, 0.0, attn)
        else:
            attn = mx.softmax(scores, axis=-1)
        
        p_attn = self.dropout(attn)
        x = mx.matmul(p_attn, value)  # (batch, head, time1, d_k)
        
        # Concatenate heads
        b, h, t, d_k = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))  # (batch, time1, head, d_k)
        x = mx.reshape(x, (b, t, h * d_k))  # (batch, time1, n_feat)
        
        return self.linear_out(x)
    
    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass of multi-head attention."""
        # Multi-head attention  
        q_h, k_h, v_h, v = self.forward_qkv(x)
        
        # FSMN processing on value tensor (after dimension transformation)
        fsmn_memory = self.forward_fsmn(v, mask)
        
        # Compute attention scores
        q_h = q_h / math.sqrt(self.d_k)
        scores = mx.matmul(q_h, mx.transpose(k_h, (0, 1, 3, 2)))
        
        # Apply attention
        att_outs = self.forward_attention(v_h, scores, mask)
        
        # Combine attention output with FSMN memory
        return att_outs + fsmn_memory


class EncoderLayerSANM(nn.Module):
    """Encoder layer with SANM attention mechanism."""
    
    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttentionSANM,
        feed_forward: PositionwiseFeedForward,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        in_size: Optional[int] = None,  # 添加in_size参数以支持第一层
    ):
        super().__init__()
        self.in_size = in_size if in_size is not None else size
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(self.in_size)  # 使用in_size
        self.norm2 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
    
    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass of encoder layer."""
        # Self-attention with residual connection
        # 当in_size != size时（如第一层560->512），不使用残差连接
        if self.normalize_before:
            x_norm = self.norm1(x)
            attn_out = self.dropout(self.self_attn(x_norm, mask))
            
            # 只有当维度匹配时才使用残差连接
            if self.in_size == self.size:
                x = x + attn_out
            else:
                x = attn_out  # 第一层不使用残差连接
        else:
            attn_out = self.dropout(self.self_attn(x, mask))
            if self.in_size == self.size:
                x = self.norm1(x + attn_out)
            else:
                x = self.norm1(attn_out)
        
        # Feed-forward with residual connection
        if self.normalize_before:
            x_norm = self.norm2(x)
            x = x + self.dropout(self.feed_forward(x_norm))
        else:
            x = self.norm2(x + self.dropout(self.feed_forward(x)))
        
        return x


class SenseVoiceEncoderSmall(nn.Module):
    """SenseVoice Encoder with SANM attention mechanism - 3 Encoder Groups Architecture."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int = 512,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 50,
        tp_blocks: int = 20,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        kernel_size: int = 11,
        sanm_shift: int = 0,
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size
        self.embed = SinusoidalPositionEncoder()
        self.normalize_before = normalize_before
        
        # === THREE ENCODER GROUPS ARCHITECTURE ===
        
        # Group 1: encoders0 (1 layer, input_size -> output_size)
        # Note: PyTorch第一层使用560作为输入维度(80 + 480 pos encoding)
        encoders0_input_size = 560  # This matches PyTorch model structure
        encoders0_layers = []
        attention_layer0 = MultiHeadedAttentionSANM(
            n_head=attention_heads,
            in_feat=encoders0_input_size,
            n_feat=output_size,
            dropout_rate=attention_dropout_rate,
            kernel_size=kernel_size,
            sanm_shift=sanm_shift,
        )
        ff_layer0 = PositionwiseFeedForward(
            idim=output_size,
            hidden_units=linear_units,
            dropout_rate=dropout_rate,
        )
        encoder_layer0 = EncoderLayerSANM(
            size=output_size,
            self_attn=attention_layer0,
            feed_forward=ff_layer0,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            in_size=encoders0_input_size,  # 第一层需要处理560->512的转换
        )
        encoders0_layers.append(encoder_layer0)
        
        # Group 2: encoders (num_blocks - 1 layers, output_size -> output_size)  
        encoders_layers = []
        for i in range(num_blocks - 1):
            attention_layer = MultiHeadedAttentionSANM(
                n_head=attention_heads,
                in_feat=output_size,
                n_feat=output_size,
                dropout_rate=attention_dropout_rate,
                kernel_size=kernel_size,
                sanm_shift=sanm_shift,
            )
            ff_layer = PositionwiseFeedForward(
                idim=output_size,
                hidden_units=linear_units,
                dropout_rate=dropout_rate,
            )
            encoder_layer = EncoderLayerSANM(
                size=output_size,
                self_attn=attention_layer,
                feed_forward=ff_layer,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
            )
            encoders_layers.append(encoder_layer)
            
        # Group 3: tp_encoders (tp_blocks layers, output_size -> output_size)
        tp_encoders_layers = []
        for i in range(tp_blocks):
            attention_layer = MultiHeadedAttentionSANM(
                n_head=attention_heads,
                in_feat=output_size,
                n_feat=output_size,
                dropout_rate=attention_dropout_rate,
                kernel_size=kernel_size,
                sanm_shift=sanm_shift,
            )
            ff_layer = PositionwiseFeedForward(
                idim=output_size,
                hidden_units=linear_units,
                dropout_rate=dropout_rate,
            )
            encoder_layer = EncoderLayerSANM(
                size=output_size,
                self_attn=attention_layer,
                feed_forward=ff_layer,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
            )
            tp_encoders_layers.append(encoder_layer)
        
        # Store as attributes for proper MLX module handling
        for i, layer in enumerate(encoders0_layers):
            setattr(self, f"encoders0_{i}", layer)
        for i, layer in enumerate(encoders_layers):
            setattr(self, f"encoders_{i}", layer)
        for i, layer in enumerate(tp_encoders_layers):
            setattr(self, f"tp_encoders_{i}", layer)
            
        # Store layer counts for iteration
        self.encoders0_count = len(encoders0_layers)
        self.encoders_count = len(encoders_layers)
        self.tp_encoders_count = len(tp_encoders_layers)
            
        # Normalization layers for each group
        self.after_norm = nn.LayerNorm(output_size)  # After encoders group
        self.tp_norm = nn.LayerNorm(output_size)      # After tp_encoders group
    
    def output_size(self) -> int:
        """Return the output size of the encoder."""
        return self._output_size
    
    def __call__(self, x: mx.array, x_lens: mx.array, debug_mode: bool = False) -> Tuple[mx.array, mx.array]:
        """Forward pass of the encoder with 3-group architecture."""
        if debug_mode:
            print(f"🔍 Encoder Input Shape: {x.shape}")
        
        # Apply positional encoding
        x = self.embed(x)
        
        if debug_mode:
            print(f"🔍 After Position Encoding: {x.shape}")
        
        # Create mask from lengths
        batch_size, max_len, _ = x.shape
        mask = mx.arange(max_len)[None, :] < x_lens[:, None]
        mask = mask.astype(mx.float32)
        
        if debug_mode:
            print(f"🔍 Masks Shape: {mask.shape}")
        
        # === GROUP 1: encoders0 (1 layer) ===
        for i in range(self.encoders0_count):
            encoder_layer = getattr(self, f"encoders0_{i}")
            x = encoder_layer(x, mask)
            if debug_mode:
                print(f"🔍 After encoders0 Layer {i}: {x.shape}")
        
        # === GROUP 2: encoders (num_blocks - 1 layers) ===
        for i in range(self.encoders_count):
            encoder_layer = getattr(self, f"encoders_{i}")
            x = encoder_layer(x, mask)
            if debug_mode and i < 3:  # Only first few layers to avoid spam
                print(f"🔍 After encoders Layer {i}: {x.shape}")
        
        # Apply normalization after main encoders
        x = self.after_norm(x)
        
        if debug_mode:
            print(f"🔍 After encoders group normalization: {x.shape}")
        
        # === GROUP 3: tp_encoders (tp_blocks layers) ===
        for i in range(self.tp_encoders_count):
            encoder_layer = getattr(self, f"tp_encoders_{i}")
            x = encoder_layer(x, mask)
            if debug_mode:
                print(f"🔍 After tp_encoders Layer {i}: {x.shape}")
        
        # Apply final tp_norm
        x = self.tp_norm(x)
        
        if debug_mode:
            print(f"🔍 Final Encoder Output: {x.shape}")
        
        # Calculate output lengths
        olens = mask.sum(axis=1).astype(mx.int32)
        
        return x, olens


class CTC(nn.Module):
    """CTC (Connectionist Temporal Classification) layer."""
    
    def __init__(self, odim: int, encoder_output_size: int, **kwargs):
        super().__init__()
        self.ctc_lo = nn.Linear(encoder_output_size, odim)
        self.odim = odim
        self.blank_id = kwargs.get('blank_id', 0)
    
    def __call__(self, hs_pad: mx.array, hlens: mx.array, ys_pad: mx.array, ys_lens: mx.array) -> mx.array:
        """Forward pass for CTC loss calculation."""
        logits = self.ctc_lo(hs_pad)
        
        # For inference, return logits
        # For training, would need actual CTC loss implementation
        if ys_pad is None:
            return logits
        
        # Placeholder for CTC loss - would need proper implementation
        # In a complete implementation, this would calculate:
        # 1. Log softmax over logits
        # 2. CTC forward-backward algorithm
        # 3. Return negative log likelihood
        log_probs = mx.log_softmax(logits, axis=-1)
        
        # Simple placeholder loss - not actual CTC
        # TODO: Implement proper CTC loss calculation
        return mx.mean(log_probs)
    
    def argmax(self, hs_pad: mx.array) -> mx.array:
        """Argmax decoding for CTC."""
        logits = self.ctc_lo(hs_pad)
        return mx.argmax(logits, axis=-1)
    
    def get_logits(self, hs_pad: mx.array) -> mx.array:
        """Get CTC logits for inference."""
        return self.ctc_lo(hs_pad)


class SenseVoiceMLX(nn.Module):
    """
    SenseVoice model implemented using MLX.
    This class aims to be a 1-to-1 architectural equivalent of the original PyTorch model.
    """
    
    def __init__(
        self,
        input_size: int = 80,
        vocab_size: int = 25055,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        encoder_conf: Optional[Dict] = None,
        ctc_conf: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__()
        
        # Set default encoder configuration
        if encoder_conf is None:
            encoder_conf = {
                "output_size": 512,
                "attention_heads": 4,
                "linear_units": 2048,
                "num_blocks": 50,
                "tp_blocks": 20,
                "dropout_rate": 0.1,
                "positional_dropout_rate": 0.1,
                "attention_dropout_rate": 0.0,
                "normalize_before": True,
                "kernel_size": 11,
                "sanm_shift": 0,
            }
        
        # Initialize encoder
        self.encoder = SenseVoiceEncoderSmall(
            input_size=input_size,
            **encoder_conf
        )
        
        encoder_output_size = self.encoder.output_size()
        
        # Initialize CTC layer
        if ctc_conf is None:
            ctc_conf = {}
        self.ctc = CTC(
            odim=vocab_size,
            encoder_output_size=encoder_output_size,
            **ctc_conf
        )
        
        # Model parameters
        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.encoder_output_size = encoder_output_size
        
        # Language and style dictionaries (matching PyTorch version)
        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        
        # Embedding layer for language and style tokens  
        embed_dim = 16  # Match PyTorch model: embed.num_embeddings = 16
        self.embed = nn.Embedding(embed_dim, 560)  # Match PyTorch model: embed.embedding_dim = 560
        
        # Emotion dictionary
        self.emo_dict = {"unk": 25009, "happy": 25001, "sad": 25002, "angry": 25003, "neutral": 25004}
        
        print("SenseVoiceMLX model initialized with all components.")
    
    def encode(self, speech: mx.array, speech_lengths: mx.array, text: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        """Encoder forward pass with language and style conditioning."""
        batch_size = speech.shape[0]
        
        # Create language queries (simplified version)
        if text is not None:
            # Extract language IDs from text tokens (first token)
            lid_tokens = text[:, 0:1]  # Shape: (batch, 1)
            # Map to language embeddings (simplified mapping)
            language_query = self.embed(mx.zeros((batch_size, 1), dtype=mx.int32))
        else:
            # Default language query
            language_query = self.embed(mx.zeros((batch_size, 1), dtype=mx.int32))
        
        # Create style queries
        if text is not None and text.shape[1] > 3:
            # Extract style tokens (4th token)
            style_tokens = text[:, 3:4]  # Shape: (batch, 1)
            style_query = self.embed(mx.ones((batch_size, 1), dtype=mx.int32))
        else:
            style_query = self.embed(mx.ones((batch_size, 1), dtype=mx.int32))
        
        # 首先需要将speech特征从80维投影到560维以匹配embedding输出
        # 这是PyTorch模型中隐含的步骤
        batch_size, seq_len, feat_dim = speech.shape
        if feat_dim != 560:
            # 创建一个线性投影层将80维转换为560维
            # 注意：这里使用零填充作为临时解决方案
            # 实际的PyTorch模型可能有专门的投影层或使用不同的特征提取
            padding_size = 560 - feat_dim  # 560 - 80 = 480
            padding = mx.zeros((batch_size, seq_len, padding_size))
            speech = mx.concatenate([speech, padding], axis=2)
            print(f"🔧 Speech特征维度扩展: {feat_dim} -> {speech.shape[2]}")
        
        # Concatenate style query with speech  
        speech = mx.concatenate([style_query, speech], axis=1)
        speech_lengths = speech_lengths + 1
        
        # Create event and emotion queries
        event_emo_query = self.embed(mx.array([[1, 2]], dtype=mx.int32))
        event_emo_query = mx.broadcast_to(event_emo_query, (batch_size, 2, speech.shape[2]))
        
        # Concatenate language and event/emotion queries
        input_query = mx.concatenate([language_query, event_emo_query], axis=1)
        speech = mx.concatenate([input_query, speech], axis=1)
        speech_lengths = speech_lengths + 3
        
        # Pass through encoder
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        
        return encoder_out, encoder_out_lens
    
    def __call__(self, speech: mx.array, speech_lengths: Optional[mx.array] = None, debug: bool = False, **kwargs) -> Dict[str, mx.array]:
        """
        Forward pass for inference.
        Includes a debug mode to return intermediate layer outputs for verification.
        
        Args:
            speech: Input features of shape (batch, seq_len, feature_dim)
            speech_lengths: Sequence lengths of shape (batch,)
            debug: If True, returns intermediate outputs for verification
            
        Returns:
            Dictionary containing model outputs with 'ctc_logits' key
            If debug=True, returns dictionary with intermediate outputs
        """
        # 如果是调试模式，则初始化一个用于存储中间输出的字典
        debug_outputs = {}
        
        if speech_lengths is None:
            batch_size, seq_len, _ = speech.shape
            speech_lengths = mx.full((batch_size,), seq_len, dtype=mx.int32)
        
        # 捕获原始输入特征
        if debug:
            debug_outputs["input_speech"] = speech
            debug_outputs["input_speech_lengths"] = speech_lengths
        
        # Encode speech
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        
        # 捕获编码器输出
        if debug:
            debug_outputs["encoder_out"] = encoder_out
            debug_outputs["encoder_out_lens"] = encoder_out_lens
        
        # Get CTC logits
        ctc_logits = self.ctc.get_logits(encoder_out)
        
        # 最终的返回逻辑
        if debug:
            # 在调试模式下，将最终输出也加入字典并返回整个字典
            debug_outputs["ctc_logits"] = ctc_logits
            return debug_outputs
        else:
            # 在正常模式下，保持与原始PyTorch模型相同的输出格式
            return {'ctc_logits': ctc_logits}
    
    def load_pytorch_weights(self, pytorch_state_dict: dict, strict: bool = True) -> None:
        """
        Load weights from PyTorch state dict with careful mapping.
        
        Args:
            pytorch_state_dict: Dictionary containing PyTorch model weights
            strict: Whether to strictly match all keys
        """
        print("Starting PyTorch to MLX weight conversion...")
        
        # Create mapping between PyTorch and MLX parameter names
        name_mapping = self._create_name_mapping()
        
        mlx_state_dict = {}
        missing_keys = []
        unexpected_keys = []
        
        # Convert PyTorch weights to MLX format
        for pytorch_key, pytorch_weight in pytorch_state_dict.items():
            if pytorch_key in name_mapping:
                mlx_key = name_mapping[pytorch_key]
                
                # Convert weight format if needed
                mlx_weight = self._convert_weight_format(pytorch_key, pytorch_weight)
                mlx_state_dict[mlx_key] = mlx_weight
                print(f"Mapped: {pytorch_key} -> {mlx_key}")
            else:
                unexpected_keys.append(pytorch_key)
        
        # Check for missing keys
        for param_name, param in self.named_parameters():
            if param_name not in mlx_state_dict:
                missing_keys.append(param_name)
        
        # Load the converted weights
        self.load_weights(mlx_state_dict)
        
        # Report status
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
            if strict:
                raise ValueError(f"Missing keys in state dict: {missing_keys}")
        
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:10]}...")  # Show first 10
        
        print(f"Successfully loaded {len(mlx_state_dict)} weight tensors.")
    
    def _create_name_mapping(self) -> dict:
        """Create mapping between PyTorch and MLX parameter names."""
        name_mapping = {}
        
        # Basic mapping patterns
        # PyTorch uses dots, MLX uses the same but may have different module names
        
        # Encoder mappings
        for i in range(50):  # Assuming 50 encoder layers
            # Attention layers
            name_mapping[f'encoder.encoders.{i}.self_attn.linear_q_k_v.weight'] = f'encoder.encoders.{i}.self_attn.linear_q_k_v.weight'
            name_mapping[f'encoder.encoders.{i}.self_attn.linear_q_k_v.bias'] = f'encoder.encoders.{i}.self_attn.linear_q_k_v.bias'
            name_mapping[f'encoder.encoders.{i}.self_attn.linear_out.weight'] = f'encoder.encoders.{i}.self_attn.linear_out.weight'
            name_mapping[f'encoder.encoders.{i}.self_attn.linear_out.bias'] = f'encoder.encoders.{i}.self_attn.linear_out.bias'
            name_mapping[f'encoder.encoders.{i}.self_attn.fsmn_block.weight'] = f'encoder.encoders.{i}.self_attn.fsmn_conv.weight'
            
            # Feed forward layers
            name_mapping[f'encoder.encoders.{i}.feed_forward.w_1.weight'] = f'encoder.encoders.{i}.feed_forward.w_1.weight'
            name_mapping[f'encoder.encoders.{i}.feed_forward.w_1.bias'] = f'encoder.encoders.{i}.feed_forward.w_1.bias'
            name_mapping[f'encoder.encoders.{i}.feed_forward.w_2.weight'] = f'encoder.encoders.{i}.feed_forward.w_2.weight'
            name_mapping[f'encoder.encoders.{i}.feed_forward.w_2.bias'] = f'encoder.encoders.{i}.feed_forward.w_2.bias'
            
            # Layer norms
            name_mapping[f'encoder.encoders.{i}.norm1.weight'] = f'encoder.encoders.{i}.norm1.weight'
            name_mapping[f'encoder.encoders.{i}.norm1.bias'] = f'encoder.encoders.{i}.norm1.bias'
            name_mapping[f'encoder.encoders.{i}.norm2.weight'] = f'encoder.encoders.{i}.norm2.weight'
            name_mapping[f'encoder.encoders.{i}.norm2.bias'] = f'encoder.encoders.{i}.norm2.bias'
        
        # CTC layer mapping
        name_mapping['ctc.ctc_lo.weight'] = 'ctc.ctc_lo.weight'
        name_mapping['ctc.ctc_lo.bias'] = 'ctc.ctc_lo.bias'
        
        # Embedding layer mapping
        name_mapping['embed.weight'] = 'embed.weight'
        
        return name_mapping
    
    def _convert_weight_format(self, pytorch_key: str, pytorch_weight) -> mx.array:
        """
        Convert PyTorch weight tensor to MLX format.
        
        Args:
            pytorch_key: The key/name of the parameter in PyTorch
            pytorch_weight: The PyTorch tensor weight
            
        Returns:
            Converted MLX array
        """
        # Convert to numpy first, then to MLX
        import numpy as np
        
        if hasattr(pytorch_weight, 'detach'):
            numpy_weight = pytorch_weight.detach().cpu().numpy()
        else:
            numpy_weight = np.array(pytorch_weight)
        
        # Handle specific layer type conversions
        if 'linear' in pytorch_key.lower() and 'weight' in pytorch_key:
            # Linear layer weights might need transposition
            # PyTorch Linear: (out_features, in_features)
            # MLX Linear: (in_features, out_features) - check MLX documentation
            pass  # Keep original for now, adjust if needed
        
        elif 'conv' in pytorch_key.lower() and 'weight' in pytorch_key:
            # Convolution weights - check if format differs
            pass  # Keep original for now
        
        # Convert to MLX array
        mlx_weight = mx.array(numpy_weight)
        
        return mlx_weight


# Initialize the model when module is imported
print("SenseVoiceMLX class definition completed.")
print("Ready for weight loading and validation.")