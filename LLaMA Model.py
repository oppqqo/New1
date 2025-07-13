import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional
# 本项目的目的是建立LLaMA模型

@dataclass
class ModelArgs:
    dim: int = 4096 # 模型维度
    n_layers: int = 32 # 层数
    n_heads: int = 32 # 注意力头数
    n_kv_heads: Optional[int] = None # kv注意力头数
    vocab_size: int = -1  # 词表大小，这个数值将会在tokenizer初始化时被设置
    multiple_of: int = 256 # 维度倍数
    ffn_dim_multiplier: Optional[float] = None # ffn维度倍数
    norm_eps: float = 1e-5 # 归一化eps
    # kv cache需要该参数
    max_batch_size: int = 32 # 最大batch大小
    max_seq_len: int = 2048 # 最大序列长度
    
    device: str = None # 设备

# 定义旋转位置编码计算函数
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0) -> torch.Tensor:
    # 根据论文，嵌入的维度必须是偶数
    assert head_dim % 2 == 0, "维度必须是偶数"
    # 根据论文的内容计算theta的参数，theta_i = 10000 ^ (-2 * (i-1) / dim)) for i in [1, 2, ..., head_dim / 2]，但此处从0开始，则为theta_i = 10000 ^ (-2 * i / dim)) for i in [0, 1, ..., head_dim / 2-1]
    # shape : (1, head_dim // 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # 生成位置索引
    # shape : (seq_len)
    m = torch.arange(seq_len, device = device)
    # 使用外积将每一种位置索引分别乘以不同的theta值
    # shape : (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # 组合实部和虚部，计算方法为 c = R * exp(i * theta * m)
    # shape : (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:
    # 首先将输入张量x转换为复数形式
    # shape : (batch_size, seq_len, H, head_dim) -> (batch_size, seq_len, H, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (batch_size, seq_len, H, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (batch_size, seq_len, H, head_dim / 2) * (1, seq_len, 1, head_dim / 2) -> (batch_size, seq_len, H, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # (batch_size, seq_len, H, head_dim / 2) -> (batch_size, seq_len, H, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (batch_size, seq_len, H, head_dim / 2, 2) -> (batch_size, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class TransformerModel(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "请先初始化tokenizer"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim) 
        # 设置编码器层
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        # 输入层RMS归一化
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层线性变换
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        # 旋转位置编码
        self.freq_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2,device=self.args.device)
    
    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        # (batch_size, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "只支持单步解码"
        
        # (batch_size, seq_len, dim)
        h = self.tok_embeddings(tokens)
        # 计算旋转位置编码[start_pos, start_pos + seq_len]
        freqs_complex = self.freq_complex[start_pos:start_pos + seq_len]
        
        # 连续应用所有的编码器层
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        # 输入层RMS归一化
        h = self.norm(h)
        # 输出层线性变换
        output = self.output(h).float()
        return output
        