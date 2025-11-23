import torch
import torch.nn as nn
import math
from typing import Optional
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in_feat, out_feat in_feat -> ... out_feat")


class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5, 
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        
        x = x.to(torch.float32)
        
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        normalized = (x / rms) * self.gain
        
        return normalized.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: Optional[int] = None,
        device=None, 
        dtype=None
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = ((d_ff + 63) // 64) * 64
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = torch.nn.functional.silu(self.w1(x))
        gated = swish * self.w3(x)
        
        return self.w2(gated)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, 
        theta: float, 
        d_k: int, 
        max_seq_len: int, 
        device=None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        position = torch.arange(max_seq_len, device=device).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_k, 2, device=device) * -(math.log(theta) / d_k)
        )
        
        angles = position * div_term
        
        self.register_buffer('cos_cached', torch.cos(angles), persistent=False)
        self.register_buffer('sin_cached', torch.sin(angles), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        rotated = rearrange(rotated, "... d two -> ... (d two)")
        
        return rotated


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    
    return exp_x / sum_exp


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    d_k = query.shape[-1]
    
    scores = einsum(
        query, key, 
        "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k"
    ) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    attn_weights = softmax(scores, dim=-1)
    
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    
    output = einsum(
        attn_weights, value,
        "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v"
    )
    
    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: Optional[RotaryPositionalEmbedding] = None,
        device=None,
        dtype=None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)
        
        self.rope = rope
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_heads)
        
        if self.rope is not None:
            q = rearrange(q, "b s h d -> b h s d")
            k = rearrange(k, "b s h d -> b h s d")
            
            token_positions_expanded = token_positions.unsqueeze(1).expand(-1, self.num_heads, -1)
            
            q = self.rope(q, token_positions_expanded)
            k = self.rope(k, token_positions_expanded)
            
            q = rearrange(q, "b h s d -> b s h d")
            k = rearrange(k, "b h s d -> b s h d")
        
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")
        
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )
        
        attn_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")
        
        output = self.w_o(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        rope: Optional[RotaryPositionalEmbedding] = None,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, rope=rope, device=device, dtype=dtype
        )
        
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        
        x = x + self.ffn(self.ln2(x))
        
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.token_embedding = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=d_k,
            max_seq_len=context_length,
            device=device
        )
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope=self.rope,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(
        self, 
        token_ids: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        
        if token_positions is None:
            token_positions = torch.arange(
                seq_len, device=token_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        x = self.token_embedding(token_ids)
        
        for layer in self.layers:
            x = layer(x, token_positions)
        
        x = self.final_norm(x)
        
        logits = self.lm_head(x)
        
        return logits