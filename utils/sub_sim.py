from typing import Callable
from torch import LongTensor
from torch.tensor import Tensor
import torch


def sub_sim(
    images: Tensor,
    captions: Tensor,
    caps_len: LongTensor,
    a_split_size: int,
    b_split_size: int,
    model: Callable[[Tensor, Tensor, Tensor], Tensor],
    device: str,
) -> Tensor:
    """Split a/b tensor to `a/b_split_size` then compute similarity score."""

    results = torch.zeros((captions.shape[0], images.shape[0]))
    results.to(device)
    a_step = images.shape[0] // a_split_size
    for i in range(a_split_size):
        i_start = i * a_step
        i_end = (i + 1) * a_step
        img_ = images[i_start:i_end]
        img_ = img_.to(device)
        b_step = captions.shape[0] // b_split_size
        for c in range(b_split_size):
            c_start = c * b_step
            c_end = (c + 1) * b_step
            cap_ = captions[c_start:c_end]
            cap_ = cap_.to(device)
            results[c_start:c_end, i_start:i_end] = model(
                img_, cap_, caps_len[c_start:c_end]
            )

    return results
