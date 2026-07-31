import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from ..config import device
from ..logger import log

_AUTOCAST_DTYPE = torch.bfloat16


# Seed Python, NumPy, and Torch for repeatable experiments.
def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


# Create an autocast context for the current compute device.
def _autocast(dev):
    return torch.autocast(
        device_type="cuda" if str(dev).startswith("cuda") else "cpu",
        dtype=_AUTOCAST_DTYPE,
    )


# Convert a display name into a filesystem-safe identifier.
def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)


# Build a warmup-plus-cosine learning-rate schedule.
def make_cosine_lr_lambda(warmup_steps: int, total_steps: int):
    def _lr_lambda(s):
        if s < warmup_steps:
            return 0.1 + 0.9 * (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return _lr_lambda


# Sample text from a trained causal language model.
def generate(model, tokenizer, prompt, length=120, temperature=0.8, block_size=128):
    model.eval()
    idx = torch.tensor(
        tokenizer.encode(prompt).ids, dtype=torch.long, device=device
    ).unsqueeze(0)
    for _ in range(length):
        with torch.no_grad():
            logits, _ = model(idx[:, -block_size:])
        tok = torch.multinomial(F.softmax(logits[0, -1] / temperature, dim=-1), 1)
        idx = torch.cat([idx, tok.unsqueeze(0)], dim=1)
    return tokenizer.decode(idx[0].tolist())


# Log module-level weight and gradient statistics for debugging training.
def _log_weight_stats(model, step):
    log.debug(f"weight stats  step={step}")
    for name, module in model.named_children():
        ws = [p.detach().float() for p in module.parameters() if p.requires_grad]
        if not ws:
            continue
        w  = torch.cat([p.flatten() for p in ws])
        gs = [p.grad.float() for p in module.parameters()
              if p.requires_grad and p.grad is not None]
        line = f"  {name:<25}  mean={w.mean():+.3e}  std={w.std():.3e}  norm={w.norm():.3e}"
        if gs:
            g = torch.cat([p.flatten() for p in gs])
            line += f"  grad_norm={g.norm():.3e}  grad_std={g.std():.3e}"
        log.debug(line)
