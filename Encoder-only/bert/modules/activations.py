"""激活函数注册表：通过字符串名称解耦配置与实现。"""

import math

import torch
from torch import nn


class GELU(nn.Module):
    """BERT 原始实现使用的 erf 精确版 GELU。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


_ACTIVATIONS = {
    "gelu": GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


def get_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError as e:
        raise ValueError(f"未知激活函数 {name!r}，可选：{list(_ACTIVATIONS)}") from e
