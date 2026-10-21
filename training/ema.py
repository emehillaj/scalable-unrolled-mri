"""Exponential moving average of the model weights (ours is evaluated on the EMA)."""

import copy

import torch
import torch.nn as nn


class ModelEMA:
    """EMA copy of a model: ``ema = decay * ema + (1 - decay) * live``, buffers are copied."""

    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.copy_(d * ema_p + (1.0 - d) * p.detach())
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)

    @torch.no_grad()
    def copy_from(self, model: nn.Module) -> None:
        """Reset the EMA to the live weights."""
        for ema_t, t in zip(self.module.state_dict().values(), model.state_dict().values()):
            ema_t.copy_(t)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.module.load_state_dict(state_dict)
