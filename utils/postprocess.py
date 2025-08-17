# utils/postprocess.py

from funasr import AutoModel
from typing import Optional
import torch


class PunctuationRestorer:
    """
    一个独立的标点恢复模块。
    
    该类封装了 FunASR 的标点恢复模型，提供一个简洁的接口
    来对无标点的文本进行后处理。
    """
    def __init__(
        self,
        model_name_or_path: str = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        初始化标点恢复器。
        
        Args:
            model_name_or_path (str): 要加载的 FunASR 标点模型名称或本地路径。
            device (Optional[str]): 指定运行设备的字符串 ('cpu', 'cuda', 'cuda:0' 等)。
                                    如果为 None，则自动检测。
            verbose (bool): 是否在加载时打印详细信息。
        """
        self.model_name = model_name_or_path
        self.verbose = verbose
        
        # 确定设备。优先使用用户指定的 device，否则自动检测 GPU。
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.verbose:
            print(f"🔧 初始化 PunctuationRestorer...")
            print(f"   - 模型: {self.model_name}")
            print(f"   - 设备: {self.device}")
        
        # 加载模型。这是核心操作。
        try:
            self.model = AutoModel(
                model=self.model_name,
                device=self.device
            )
            if self.verbose:
                print("   ✅ 标点恢复模型加载成功。")
        except Exception as e:
            # 关键的错误处理：如果模型加载失败，系统必须明确失败，而不是静默错误。
            raise RuntimeError(f"无法加载标点恢复模型 '{self.model_name}'。错误: {e}")
    
    def restore(self, text: str) -> str:
        """
        对输入的文本字符串进行标点恢复。
        
        Args:
            text (str): 不含标点的原始文本。
            
        Returns:
            str: 添加了标点的文本。
        """
        # 边缘情况处理：对于空或仅包含空白的输入，直接返回，避免不必要的模型调用。
        if not text or not text.strip():
            return text
        
        try:
            # 调用 FunASR 模型。直接传入文本字符串。
            result = self.model.generate(input=text)
            
            # 从返回结果中提取处理后的文本。
            # FunASR 的标点模型通常返回一个包含 'text' 键的字典列表。
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'text' in result[0]:
                    punctuated_text = result[0]['text']
                else:
                    # 如果格式不同，尝试直接使用结果
                    punctuated_text = str(result[0]) if result else text
            else:
                punctuated_text = str(result) if result else text
            
            return punctuated_text
            
        except Exception as e:
            # 如果推理失败，打印警告并返回原始文本。
            # 这是一种服务降级策略，确保即使标点功能失败，主流程也不会中断。
            if self.verbose:
                print(f"⚠️  标点恢复推理失败: {e}。将返回原始文本。")
            return text