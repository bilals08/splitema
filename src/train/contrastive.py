import math
import time
from itertools import cycle
from pathlib import Path
import torch
import torch.nn.functional as F
from ..config import SEED, device
from ..data import make_contrastive_loader
from ..models import SpectrumEmbeddingTransformer
from .utils import (
    _autocast,
    make_cosine_lr_lambda, safe_name, set_seed,
)


# Compute bidirectional InfoNCE loss with theoretical positives and decoys.
def _infonce(ee, te, de, temperature):
    B      = ee.size(0)
    labels = torch.arange(B, device=ee.device)
    loss_fwd = F.cross_entropy(ee @ torch.cat([te, de], dim=0).T / temperature, labels)
    loss_bwd = F.cross_entropy(te @ torch.cat([ee, de], dim=0).T / temperature, labels)
    return 0.5 * (loss_fwd + loss_bwd)


# Evaluate contrastive retrieval loss and top-1 matching accuracy.
def evaluate_contrastive(model, loader, temperature, eval_batches=None):
    model.eval()
    dev = next(model.parameters()).device
    losses, correct, total = [], 0, 0

    with torch.no_grad(), _autocast(dev):
        for i, (ep, _, el, tp, tl, dp, dl, _) in enumerate(loader):
            if eval_batches is not None and i >= eval_batches:
                break
            ee, _ = model(ep.to(dev), el.to(dev))
            te, _ = model(tp.to(dev), tl.to(dev))
            de, _ = model(dp.to(dev), dl.to(dev))
            labels = torch.arange(ee.size(0), device=dev)
            losses.append(_infonce(ee, te, de, temperature).item())
            pool = torch.cat([te, de], dim=0)
            correct += (ee @ pool.T).argmax(-1).eq(labels).sum().item()
            total   += labels.numel()

    model.train()
    return {
        "loss":               sum(losses) / max(1, len(losses)),
        "retrieval_accuracy": correct / max(1, total),
    }


# Train one or more spectrum embedding models on the contrastive task.
def train_and_eval_contrastive_multi(
    model_specs,
    train_ds,
    val_ds,
    test_ds,
    cfg,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    save_dir=None,
    epochs=None,
    seed: int = SEED,
):
    set_seed(seed)
    train_loader = make_contrastive_loader(
        train_ds, cfg.batch_size, True, num_workers, persistent_workers,
        drop_last=True, seed=seed, prefetch_factor=prefetch_factor,
    )
    val_loader   = make_contrastive_loader(
        val_ds, cfg.batch_size, False, num_workers,
        seed=seed + 1, prefetch_factor=prefetch_factor,
    )
    test_loader  = make_contrastive_loader(
        test_ds, cfg.batch_size, False, num_workers,
        seed=seed + 2, prefetch_factor=prefetch_factor,
    )
    train_iter   = cycle(train_loader)

    total_steps     = math.ceil(len(train_ds) / cfg.batch_size) * epochs if epochs else cfg.train_steps
    steps_per_epoch = max(1, math.ceil(len(train_ds) / cfg.batch_size))

    set_seed(seed)
    models, optimizers, schedulers = [], [], []
    best_val_accs, best_states, use_aux_flags = [], [], []

    for name, attn_cls, attn_kw in model_specs:
        m = SpectrumEmbeddingTransformer(
            attn_cls, cfg.d_model, cfg.n_head, cfg.n_layer, attn_kw
        ).to(device)
        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in m.parameters())
        print(f"  {name}: {n_train:,} trainable / {n_total:,} total")

        opt = torch.optim.AdamW(m.get_param_groups(cfg.lr), weight_decay=cfg.weight_decay)
        sch = (
            torch.optim.lr_scheduler.LambdaLR(opt, make_cosine_lr_lambda(cfg.warmup_steps, total_steps))
            if total_steps > 0 else None
        )
        models.append(m)
        optimizers.append(opt)
        schedulers.append(sch)
        best_val_accs.append(float("-inf"))
        best_states.append(None)
        use_aux_flags.append(any(getattr(sub, "diversity_weight", 0) > 0 for sub in m.modules()))

    dev         = next(models[0].parameters()).device
    n_models    = len(models)
    model_names = [n for n, _, _ in model_specs]
    t0          = time.time()
    step        = 0
    last_losses = [float("nan")] * n_models

    while step < total_steps:
        ep, _, el, tp, tl, dp, dl, _ = next(train_iter)
        ep = ep.to(dev)
        el = el.to(dev)
        tp = tp.to(dev)
        tl = tl.to(dev)
        dp = dp.to(dev)
        dl = dl.to(dev)

        for i in range(n_models):
            m, opt, sch = models[i], optimizers[i], schedulers[i]
            with _autocast(dev):
                ee, aux_e = m(ep, el)
                te, aux_t = m(tp, tl)
                de, aux_d = m(dp, dl)
                loss = _infonce(ee, te, de, cfg.temperature)
                if use_aux_flags[i]:
                    loss = loss + (aux_e + aux_t + aux_d) / 3
            last_losses[i] = loss.item()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if sch:
                sch.step()
            m.update_ema()

        step += 1

        if save_dir and step % steps_per_epoch == 0:
            epoch_num = step // steps_per_epoch
            for i in range(n_models):
                ep_path = Path(save_dir) / f"{safe_name(model_specs[i][0])}_epoch{epoch_num:03d}.pt"
                tmp_path = ep_path.with_suffix(".tmp")
                torch.save({
                    "step":      step,
                    "epoch":     epoch_num,
                    "model":     models[i].state_dict(),
                    "optimizer": optimizers[i].state_dict(),
                    "scheduler": schedulers[i].state_dict() if schedulers[i] else None,
                }, tmp_path)
                tmp_path.replace(ep_path)
                print(f"  [{model_specs[i][0]}] epoch {epoch_num} checkpoint: {ep_path}")

        if step % 100 == 0 or step == 1:
            epoch_now   = step // steps_per_epoch
            for i in range(n_models):
                name = model_names[i]
                vm   = evaluate_contrastive(models[i], val_loader, cfg.temperature, cfg.eval_batches)
                acc  = vm["retrieval_accuracy"]
                print(
                    f"  [{name}] epoch {epoch_now:3d}  step {step:6d}"
                    f"  loss {last_losses[i]:.4f}"
                    f"  val_loss {vm['loss']:.4f}"
                    f"  val_acc {100 * acc:.2f}%"
                )
                if acc > best_val_accs[i]:
                    best_val_accs[i] = acc
                    best_states[i]   = {k: v.detach().cpu().clone() for k, v in models[i].state_dict().items()}

    results = []
    for i, (name, _, _) in enumerate(model_specs):
        m = models[i]
        if best_states[i]:
            m.load_state_dict(best_states[i])
        m.eval()

        val_m  = evaluate_contrastive(m, val_loader,  cfg.temperature, cfg.eval_batches)
        test_m = evaluate_contrastive(m, test_loader, cfg.temperature, cfg.eval_batches)

        weights_path = None
        if save_dir:
            sd = Path(save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            wp = sd / f"{safe_name(name)}.pt"
            torch.save(m.state_dict(), wp)
            weights_path = str(wp)

        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in m.parameters())
        results.append({
            "name":                  name,
            "trainable":             n_train,
            "total":                 n_total,
            "time_s":                round(time.time() - t0, 1),
            "val_loss":              round(val_m["loss"], 4),
            "val_retrieval_acc":     round(val_m["retrieval_accuracy"], 4),
            "val_retrieval_acc_pct": round(100 * val_m["retrieval_accuracy"], 2),
            "test_loss":             round(test_m["loss"], 4),
            "test_retrieval_acc":    round(test_m["retrieval_accuracy"], 4),
            "test_retrieval_acc_pct":round(100 * test_m["retrieval_accuracy"], 2),
            "weights_path":          weights_path,
        })

    return results
