"""MLM 掩码策略：选取 15% 的 token，其中 80% -> [MASK]，10% -> 随机词，10% 保持不变。"""

from typing import Iterable, Optional, Tuple

import torch

IGNORE_INDEX = -100


class MLMMasker:
    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        special_token_ids: Iterable[int],
        mlm_probability: float = 0.15,
        mask_ratio: float = 0.8,
        random_ratio: float = 0.1,
    ):
        assert 0 < mlm_probability < 1 and mask_ratio + random_ratio <= 1
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.special_token_ids = torch.tensor(sorted(set(special_token_ids)))
        self.mlm_probability = mlm_probability
        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio

    def __call__(
        self, input_ids: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (masked_input_ids, labels)，未被选中位置的 label 为 IGNORE_INDEX。"""
        input_ids = input_ids.clone()
        labels = input_ids.clone()

        prob = torch.full(input_ids.shape, self.mlm_probability)
        prob[torch.isin(input_ids, self.special_token_ids)] = 0.0
        selected = torch.bernoulli(prob, generator=generator).bool()
        labels[~selected] = IGNORE_INDEX

        rand = torch.rand(input_ids.shape, generator=generator)
        to_mask = selected & (rand < self.mask_ratio)
        to_random = selected & (rand >= self.mask_ratio) & (rand < self.mask_ratio + self.random_ratio)

        input_ids[to_mask] = self.mask_token_id
        random_words = torch.randint(self.vocab_size, input_ids.shape, generator=generator)
        input_ids[to_random] = random_words[to_random]
        return input_ids, labels
