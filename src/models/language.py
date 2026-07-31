import torch
import torch.nn as nn
from ..attention import _slow_fast_groups


# Transformer decoder block for causal language modeling.
class Block(nn.Module):
    def __init__(self, d_model, n_head, attention_cls, **kw):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.attn = attention_cls(d_model, n_head, **kw)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        out, aux = self.attn(self.ln1(x), causal=True)
        x = x + out
        return x + self.ffn(self.ln2(x)), aux

    def update_ema(self):
        if hasattr(self.attn, "update_ema"):
            self.attn.update_ema()


# Causal transformer language model with pluggable attention.
class LMTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, block_size, attention_cls, **kw):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.pos    = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, attention_cls, **kw) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(d_model)
        self.fc     = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        aux_sum = x.new_zeros(1)
        for b in self.blocks:
            h, aux = b(h)
            aux_sum = aux_sum + aux
        return self.fc(self.ln_f(h)), aux_sum

    def update_ema(self):
        for b in self.blocks:
            b.update_ema()

    def get_param_groups(self, base_lr):
        return _slow_fast_groups(self, base_lr)
