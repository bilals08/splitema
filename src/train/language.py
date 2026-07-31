import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from ..config import SEED, device
from ..models import LMTransformer
from .utils import _log_weight_stats, generate, safe_name, set_seed


# Train one language-model attention variant and report validation/test metrics.
def train_and_eval_language(
    name,
    attention_cls,
    attn_kwargs,
    vocab_size,
    get_batch,
    get_val_batch,
    get_test_batch,
    tokenizer,
    cfg,
    steps_per_epoch=1,
    seed=SEED,
    save_dir=None,
    checkpoint_epochs=(100,),
):
    set_seed(seed)
    model = LMTransformer(
        vocab_size, cfg.d_model, cfg.n_head, cfg.n_layer,
        cfg.block_size, attention_cls, **attn_kwargs,
    ).to(device)
    set_seed(seed + 1)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{name}  ({n_train:,} params)")

    opt = torch.optim.AdamW(model.get_param_groups(cfg.lr), weight_decay=cfg.weight_decay)

    def eval_split(batch_fn, n=5, seed_base=2000):
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(n):
                x, y = batch_fn(seed=seed_base + i)
                logits, _ = model(x)
                losses.append(F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item())
        model.train()
        return sum(losses) / n

    steps_per_epoch = max(1, int(steps_per_epoch))
    best_val, best_epoch, stagnant, best_state = float("inf"), -1, 0, None
    t0    = time.time()
    epoch = 0
    step  = 0
    checkpoint_paths = []
    best_checkpoint_path = None
    checkpoint_epochs = set(int(e) for e in checkpoint_epochs)

    for epoch in range(cfg.max_epochs):
        train_losses = []
        for _ in range(steps_per_epoch):
            x, y = get_batch()
            logits, aux = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)) + aux
            train_losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_ema()
            step += 1

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = eval_split(get_val_batch)
        if val_loss < best_val:
            best_val, best_epoch, stagnant = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_dir:
                sd = Path(save_dir)
                sd.mkdir(parents=True, exist_ok=True)
                best_checkpoint_path = sd / f"{safe_name(name)}_best.pt"
                torch.save(
                    {
                        "model": best_state,
                        "optimizer": opt.state_dict(),
                        "epoch": epoch + 1,
                        "step": step,
                        "val_loss": val_loss,
                        "config": {
                            "vocab_size": vocab_size,
                            "d_model": cfg.d_model,
                            "n_head": cfg.n_head,
                            "n_layer": cfg.n_layer,
                            "block_size": cfg.block_size,
                            "seed": seed,
                        },
                    },
                    best_checkpoint_path,
                )
        else:
            stagnant += 1

        if epoch % 1 == 0 or stagnant == 0:
            _log_weight_stats(model, step)
            print(
                f"  ep {epoch + 1:4d}/{cfg.max_epochs}"
                f"  step {step:7d}"
                f"  loss {train_loss:.4f}"
                f"  val {val_loss:.4f}"
                f"  best {best_val:.4f}"
            )
            if stagnant == 0 and best_checkpoint_path:
                print(f"  saved best checkpoint: {best_checkpoint_path}")

        epoch_num = epoch + 1
        if save_dir and epoch_num in checkpoint_epochs:
            sd = Path(save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            path = sd / f"{safe_name(name)}_epoch{epoch_num:03d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch_num,
                    "step": step,
                    "val_loss": val_loss,
                    "config": {
                        "vocab_size": vocab_size,
                        "d_model": cfg.d_model,
                        "n_head": cfg.n_head,
                        "n_layer": cfg.n_layer,
                        "block_size": cfg.block_size,
                        "seed": seed,
                    },
                },
                path,
            )
            checkpoint_paths.append(str(path))
            print(f"  saved checkpoint: {path}")

    if best_state:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0

    def final_eval(batch_fn, seed_base, n=20):
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(n):
                x, y = batch_fn(seed=seed_base + i)
                logits, _ = model(x)
                losses.append(F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item())
        model.train()
        return sum(losses) / n

    val_loss  = final_eval(get_val_batch,  2000)
    test_loss = final_eval(get_test_batch, 3000)
    val_ppl, test_ppl = math.exp(val_loss), math.exp(test_loss)

    print(f"  done {elapsed:.1f}s  val_ppl={val_ppl:.2f}  test_ppl={test_ppl:.2f}")
    print(f"  sample: {generate(model, tokenizer, 'The ', length=80, block_size=cfg.block_size)}")

    return {
        "name":      name,
        "trainable": n_train,
        "total":     sum(p.numel() for p in model.parameters()),
        "epochs":    epoch + 1,
        "steps_per_epoch": steps_per_epoch,
        "train_steps": step,
        "time_s":    round(elapsed, 1),
        "val_loss":  round(val_loss, 4),
        "val_ppl":   round(val_ppl, 2),
        "test_loss": round(test_loss, 4),
        "test_ppl":  round(test_ppl, 2),
        "best_epoch": best_epoch + 1,
        "best_checkpoint_path": str(best_checkpoint_path),
        "checkpoint_paths": ";".join(checkpoint_paths),
    }
