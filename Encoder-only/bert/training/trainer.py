"""通用 Trainer：只依赖「模型 forward 返回带 .loss 的对象」这一约定，与具体任务解耦。"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..model import BertPreTrainedModel
from .optim import build_linear_warmup_scheduler, build_optimizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    output_dir: str = "outputs"
    num_epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    log_every: int = 50
    save_every_epoch: bool = True
    device: str = "auto"
    use_amp: bool = False

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


class Trainer:
    def __init__(
        self,
        model: BertPreTrainedModel,
        args: TrainingArguments,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        callbacks: Optional[list[Callable[[int, Dict[str, float]], None]]] = None,
    ):
        self.args = args
        self.device = args.resolve_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.callbacks = callbacks or []

        steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        self.total_steps = steps_per_epoch * args.num_epochs
        self.optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
        self.scheduler = build_linear_warmup_scheduler(
            self.optimizer, int(self.total_steps * args.warmup_ratio), self.total_steps
        )
        self.amp = args.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.global_step = 0

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _emit(self, metrics: Dict[str, float]) -> None:
        logger.info("step %d | %s", self.global_step,
                    " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        for cb in self.callbacks:
            cb(self.global_step, metrics)

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def train(self) -> None:
        accum = self.args.gradient_accumulation_steps
        for epoch in range(self.args.num_epochs):
            self.model.train()
            running, count = 0.0, 0
            for i, batch in enumerate(self.train_loader):
                batch = self._to_device(batch)
                with torch.autocast(self.device.type, enabled=self.amp):
                    loss = self.model(**batch).loss / accum
                self.scaler.scale(loss).backward()
                running += loss.item() * accum
                count += 1

                if (i + 1) % accum == 0 or (i + 1) == len(self.train_loader):
                    self._optimizer_step()
                    if self.global_step % self.args.log_every == 0:
                        self._emit({"loss": running / count, "lr": self.scheduler.get_last_lr()[0]})
                        running, count = 0.0, 0

            if self.eval_loader is not None:
                self._emit({"epoch": epoch, **self.evaluate()})
            if self.args.save_every_epoch:
                self.save(Path(self.args.output_dir) / f"epoch-{epoch}")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.eval_loader:
            total += self.model(**self._to_device(batch)).loss.item()
            n += 1
        avg = total / max(1, n)
        return {"eval_loss": avg, "eval_ppl": math.exp(min(avg, 20))}

    def save(self, directory: Path) -> None:
        self.model.save_pretrained(directory)
        logger.info("checkpoint saved to %s", directory)
