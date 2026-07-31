import torch
import torch.nn as nn
import torch.nn.functional as F
from ..attention import _slow_fast_groups
from ..config import MAX_PEAKS, VOCAB_WHOLE, VOCAB_FRAC, VOCAB_INT
from ..logger import log


class PeakTokenEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.whole_embed = nn.Embedding(VOCAB_WHOLE, d_model, padding_idx=0)
        self.frac_embed = nn.Embedding(VOCAB_FRAC, d_model, padding_idx=0)
        self.int_embed = nn.Embedding(VOCAB_INT, d_model, padding_idx=0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        emb = (
            self.whole_embed(x[..., 0])
            + self.frac_embed(x[..., 1])
            + self.int_embed(x[..., 2])
        )
        return self.norm(emb)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, attention_cls, attn_kwargs,
                 dropout=0.0):
        super().__init__()
        ff = 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = attention_cls(d_model, n_head, **attn_kwargs)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        h = self.norm1(x)
        out, aux = self.attn(h, h, h, causal=False, padding_mask=padding_mask)
        x = x + out
        return x + self.ffn(self.norm2(x)), aux


class SpectrumEmbeddingTransformer(nn.Module):
    def __init__(self, attention_cls, d_model: int, n_head: int, n_layer: int,
                 attn_kwargs, max_peaks: int = MAX_PEAKS):
        super().__init__()
        self.max_peaks = max_peaks

        self.peak_embed     = PeakTokenEmbedding(d_model)
        self.cls_token      = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed      = nn.Parameter(torch.randn(1, max_peaks + 1, d_model) * 0.02)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_head, attention_cls, attn_kwargs) for _ in range(n_layer)]
        )
        self.norm       = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def encode(self, peaks: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = peaks.size(0)
        lengths = lengths.to(peaks.device).long().clamp(min=0, max=self.max_peaks)

        _dbg = not getattr(self, "_shapes_logged", False)
        if _dbg:
            log.debug("SpectrumEmbeddingTransformer peak-token forward")
            log.debug(f"  input peak tokens={tuple(peaks.shape)}")

        x   = self.peak_embed(peaks)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed[:, :peaks.size(1) + 1]
        positions = torch.arange(peaks.size(1), device=peaks.device).unsqueeze(0)
        peak_mask = positions < lengths.unsqueeze(1)
        padding_mask = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=peaks.device), peak_mask],
            dim=1,
        )

        if _dbg:
            log.debug(f"  tokens x={tuple(x.shape)}")

        aux_sum = torch.zeros(1, device=peaks.device, dtype=x.dtype)
        for i, layer in enumerate(self.encoder_layers):
            x, aux = layer(x, padding_mask=padding_mask)
            if _dbg:
                log.debug(f"  layer {i+1:2d} x={tuple(x.shape)}")
            aux_sum = aux_sum + aux

        emb = F.normalize(self.projection(self.norm(x[:, 0])), dim=-1)
        if _dbg:
            log.debug(f"  output emb={tuple(emb.shape)} (L2-normalised)")
            self._shapes_logged = True
        return emb, aux_sum

    def forward(self, peaks: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(peaks, lengths)

    def get_param_groups(self, base_lr):
        return _slow_fast_groups(self, base_lr)

    @torch.no_grad()
    def update_ema(self):
        for m in self.modules():
            if m is not self and hasattr(m, "update_ema"):
                m.update_ema()
