"""优化器与学习率调度：AdamW（bias / LayerNorm 不做 weight decay）+ 线性 warmup / 衰减。"""

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01,
                    betas=(0.9, 0.999), eps: float = 1e-6) -> Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "layer_norm" in name else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=lr, betas=betas, eps=eps)


def build_linear_warmup_scheduler(optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda)
