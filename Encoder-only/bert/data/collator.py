"""批处理：动态 padding + 在线 MLM 掩码（每个 epoch 掩码位置不同，即 dynamic masking）。"""

from typing import Dict, List

import torch

from .masking import MLMMasker


class PretrainingCollator:
    def __init__(self, masker: MLMMasker, pad_token_id: int = 0):
        self.masker = masker
        self.pad_token_id = pad_token_id

    def _pad(self, seqs: List[List[int]], value: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        return torch.tensor([s + [value] * (max_len - len(s)) for s in seqs], dtype=torch.long)

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        input_ids = self._pad([b["input_ids"] for b in batch], self.pad_token_id)
        masked_ids, mlm_labels = self.masker(input_ids)
        out = {
            "input_ids": masked_ids,
            "token_type_ids": self._pad([b["token_type_ids"] for b in batch], 0),
            "attention_mask": self._pad([b["attention_mask"] for b in batch], 0),
            "mlm_labels": mlm_labels,
        }
        if "nsp_label" in batch[0]:
            out["nsp_labels"] = torch.tensor([b["nsp_label"] for b in batch], dtype=torch.long)
        return out
