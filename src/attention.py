import torch
import torch.nn as nn
import torch.nn.functional as F
from linformer import LinformerSelfAttention
from transformers import LongformerConfig
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention


def _attn_mask(padding_mask, dtype):
    if padding_mask is None:
        return None
    pad = padding_mask if padding_mask.dim() == 4 else padding_mask[:, None, None, :]
    return torch.zeros_like(pad, dtype=dtype).masked_fill(~pad, float("-inf"))


def _sdpa_mask(padding_mask, dtype, causal):
    if causal:
        return None
    return _attn_mask(padding_mask, dtype)


def _slow_fast_groups(model, base_lr):
    slow_ids, slow, scale = set(), [], None
    for m in model.modules():
        if hasattr(m, "k_slow") and hasattr(m, "v_slow"):
            scale = getattr(m, "slow_lr_scale", scale)
            for p in list(m.k_slow.parameters()) + list(m.v_slow.parameters()):
                if id(p) not in slow_ids:
                    slow_ids.add(id(p))
                    slow.append(p)
    fast = [p for p in model.parameters() if id(p) not in slow_ids]
    if not slow:
        return [{"params": fast, "lr": base_lr}]
    return [{"params": fast, "lr": base_lr}, {"params": slow, "lr": base_lr * scale}]


class VanillaAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.head_dim, self.d_model = n_head, d_model // n_head, d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key=None, value=None, causal=False, padding_mask=None):
        key = query if key is None else key
        value = key if value is None else value
        B, Tq, _ = query.shape
        Tk = key.shape[1]
        split = lambda x, lin, t: lin(x).view(B, t, self.n_head, self.head_dim).transpose(1, 2)
        q, k, v = split(query, self.q_proj, Tq), split(key, self.k_proj, Tk), split(value, self.v_proj, Tk)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=_sdpa_mask(padding_mask, query.dtype, causal), is_causal=causal)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)), query.new_zeros(1)

    def update_ema(self): pass


class SplitEMAAttention(nn.Module):
    EMA_DECAY = 0.99

    def __init__(self, d_model, n_head, diversity_weight=0.1, diversity_power=1):
        super().__init__()
        assert d_model % n_head == 0 and n_head % 2 == 0
        if diversity_power < 1:
            raise ValueError("diversity_power must be >= 1")
        self.n_head, self.head_dim, self.d_model = n_head, d_model // n_head, d_model
        self.n_grad = self.n_ema = n_head // 2
        self.diversity_weight = diversity_weight
        self.diversity_power = diversity_power
        gw = self.n_grad * self.head_dim
        self.q_grad = nn.Linear(d_model, gw)
        self.k_grad = nn.Linear(d_model, gw)
        self.v_grad = nn.Linear(d_model, gw)
        # Reuse grad Q for the EMA path.
        self.k_ema  = nn.Linear(d_model, gw)
        self.v_ema  = nn.Linear(d_model, gw)
        for p in (list(self.k_ema.parameters()) +
                  list(self.v_ema.parameters())):
            p.requires_grad_(False)
        self.o_proj = nn.Linear(d_model, d_model)

    @torch.no_grad()
    def update_ema(self):
        for el, gl in ((self.k_ema, self.k_grad),
                       (self.v_ema, self.v_grad)):
            for pe, pg in zip(el.parameters(), gl.parameters()):
                pe.data.mul_(self.EMA_DECAY).add_(pg.data, alpha=1 - self.EMA_DECAY)

    def forward(self, query, key=None, value=None, causal=False, padding_mask=None):
        key = query if key is None else key
        value = key if value is None else value
        B, T, _ = query.shape
        Tk, hd, ng = key.shape[1], self.head_dim, self.n_grad
        mask = _sdpa_mask(padding_mask, query.dtype, causal)

        q = self.q_grad(query).view(B, T, ng, hd).transpose(1, 2)

        def _run(k_l, v_l, src_k, src_v):
            k = k_l(src_k).view(B, Tk, ng, hd).transpose(1, 2)
            v = v_l(src_v).view(B, Tk, ng, hd).transpose(1, 2)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal
                   ).transpose(1, 2).contiguous().view(B, T, -1)

        grad_out = _run(self.k_grad, self.v_grad, key, value)
        ema_out  = _run(self.k_ema,  self.v_ema,  key, value)

        if self.training and self.diversity_weight > 0:
            cosine = F.cosine_similarity(grad_out.mean(1), ema_out.mean(1), dim=-1)
            aux = cosine.pow(self.diversity_power).mean() * self.diversity_weight
        else:
            aux = query.new_zeros(1)
        return self.o_proj(torch.cat([grad_out, ema_out], dim=-1)), aux


class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, n_head, group_size=2):
        super().__init__()
        assert d_model % n_head == 0 and n_head % group_size == 0
        self.n_head, self.head_dim = n_head, d_model // n_head
        self.n_kv = n_head // group_size
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.n_kv * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.n_kv * self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key=None, value=None, causal=False, padding_mask=None):
        key = query if key is None else key
        value = key if value is None else value
        B, Tq, _ = query.shape
        Tk, hd = key.shape[1], self.head_dim
        q = self.q_proj(query).view(B, Tq, self.n_head, hd).transpose(1, 2)
        k = self.k_proj(key).view(B, Tk, self.n_kv, hd).transpose(1, 2)
        v = self.v_proj(value).view(B, Tk, self.n_kv, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=_sdpa_mask(padding_mask, query.dtype, causal),
                                             is_causal=causal, enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, Tq, -1)), query.new_zeros(1)

    def update_ema(self): pass


class LongformerAttention(nn.Module):
    def __init__(self, d_model, n_head, attention_window=64):
        super().__init__()
        if attention_window % 2 != 0:
            raise ValueError("Longformer attention_window must be even")

        config = LongformerConfig(
            hidden_size=d_model,
            num_attention_heads=n_head,
            attention_window=[attention_window],
            attention_probs_dropout_prob=0.0,
        )
        self.attn = LongformerSelfAttention(config, layer_id=0)
        self.attention_window = attention_window

    def forward(self, query, key=None, value=None, causal=False, padding_mask=None):
        if causal:
            raise ValueError("LongformerAttention does not support causal attention")
        if key is not None and key is not query:
            raise ValueError("LongformerAttention only supports self-attention")
        if value is not None and value is not query:
            raise ValueError("LongformerAttention only supports self-attention")

        B, T, D = query.shape
        multiple = self.attention_window
        pad_len = (-T) % multiple
        if pad_len:
            query = F.pad(query, (0, 0, 0, pad_len))
            if padding_mask is not None:
                padding_mask = F.pad(padding_mask, (0, pad_len), value=False)

        attention_mask = query.new_zeros(query.shape[:2])
        if padding_mask is not None:
            attention_mask = attention_mask.masked_fill(~padding_mask.to(torch.bool), -10000.0)
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = bool(is_index_global_attn.any().item())

        out = self.attn(
            query,
            attention_mask=attention_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=False,
        )[0]
        return out[:, :T], query.new_zeros(1)

    def update_ema(self): pass


class LinformerAttention(nn.Module):
    def __init__(self, d_model, n_head, seq_len=1024, k=256, one_kv_head=False, share_kv=False):
        super().__init__()
        self.attn = LinformerSelfAttention(
            dim=d_model,
            seq_len=seq_len,
            k=min(k, seq_len),
            heads=n_head,
            one_kv_head=one_kv_head,
            share_kv=share_kv,
            dropout=0.0,
        )

    def forward(self, query, key=None, value=None, causal=False, padding_mask=None):
        if causal:
            raise ValueError("LinformerAttention adapter does not support causal attention")
        if key is not None and key is not query:
            raise ValueError("LinformerAttention adapter only supports self-attention")
        if value is not None and value is not query:
            raise ValueError("LinformerAttention adapter only supports self-attention")
        if padding_mask is not None:
            query = query.masked_fill(~padding_mask.to(torch.bool).unsqueeze(-1), 0)
        return self.attn(query), query.new_zeros(1)

    def update_ema(self): pass
