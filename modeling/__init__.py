"""Model construction for the four methods (ours, e2e, gleam, flat)."""
import torch.nn as nn

from modeling.flat import FlatVarNet
from modeling.gleam import GleamVarNet
from modeling.sampled_varnet import SampledVarNet
from utils.misc import METHODS

__all__ = ["build_model", "SampledVarNet", "GleamVarNet", "FlatVarNet"]


def build_model(cfg: dict) -> nn.Module:
    """Build the model for ``cfg["method"]``.

    ours / e2e -> SampledVarNet (ours samples one cascade per step at training
    time; e2e runs the full chain). gleam -> GleamVarNet, flat -> FlatVarNet.
    Required keys: method, challenge ("singlecoil" or "multicoil"),
    num_cascades, chans. Optional: pools (4), sens_chans (8), sens_pools (4) and,
    for ours / e2e, use_time_embed (False), time_embedded_unet_arch
    ("standard"), bitpack_relu_mask (True), exact_suffix (False),
    activation_checkpointing (False), and for gleam / flat,
    zero_init_final_conv (True).
    """
    method = cfg["method"]
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    common = dict(
        num_cascades=cfg["num_cascades"],
        chans=cfg["chans"],
        pools=cfg.get("pools", 4),
        sens_chans=cfg.get("sens_chans", 8),
        sens_pools=cfg.get("sens_pools", 4),
        singlecoil=(cfg["challenge"] == "singlecoil"),
    )
    if method in ("ours", "e2e"):
        return SampledVarNet(
            use_time_embed=cfg.get("use_time_embed", False),
            time_embedded_unet_arch=cfg.get("time_embedded_unet_arch", "standard"),
            bitpack_relu_mask=cfg.get("bitpack_relu_mask", True),
            exact_suffix=cfg.get("exact_suffix", False),
            activation_checkpointing=cfg.get("activation_checkpointing", False),
            **common,
        )
    if cfg.get("use_time_embed", False):
        raise ValueError(f"use_time_embed is not supported for method {method!r}")
    zero_init = cfg.get("zero_init_final_conv", True)
    if method == "gleam":
        return GleamVarNet(zero_init_final_conv=zero_init, **common)
    return FlatVarNet(zero_init_final_conv=zero_init, **common)
