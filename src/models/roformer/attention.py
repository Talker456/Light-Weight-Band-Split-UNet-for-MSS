import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Rotary Positional Embedding) module."""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device),
        )

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RoPEAttention(nn.Module):
    """Self-Attention module with RoPE application."""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rope(q, seq_len=N)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1).type_as(q)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RoPETransformerBlock(nn.Module):
    """Transformer Block containing RoPE Attention."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class InterleavedRoPEBlock(nn.Module):
    """Interleaved Transformer block that alternates modeling between time and frequency axes."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.time_block = RoPETransformerBlock(dim, num_heads, mlp_ratio, dropout)
        self.freq_block = RoPETransformerBlock(dim, num_heads, mlp_ratio, dropout)

    def forward(self, x):
        B, C, F, T = x.shape
        
        # 1. Time-domain modeling
        x = x.permute(0, 2, 3, 1).reshape(B * F, T, C)
        x = self.time_block(x)
        
        # 2. Frequency-domain modeling
        x = x.view(B, F, T, C).transpose(1, 2).reshape(B * T, F, C)
        x = self.freq_block(x)
        
        # 3. Output Reconstruction
        x = x.view(B, T, F, C).transpose(1, 2).permute(0, 3, 1, 2).contiguous()
        return x
