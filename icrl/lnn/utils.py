import torch


def val_clamp(x, _min: float = 0, _max: float = 1) -> torch.Tensor:
    """gradient-transparent clamping to clamp values between [min, max]"""
    clamp_min = (x.detach() - _min).clamp(max=0)
    clamp_max = (x.detach() - _max).clamp(min=0)
    return x - clamp_max - clamp_min
