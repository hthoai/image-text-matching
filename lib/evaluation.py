"""Evaluation"""
from typing import Tuple
import torch

def ComputeMetric(sim_score: torch.Tensor, return_ranks: bool=False) -> Tuple:
    """
    Sim score: (N caption, N image) matrix of similarity score
    """
    n_img = sim_score.size(1)
    idx_ = torch.arange(n_img).repeat_interleave(5)

    args_max = torch.argsort(sim_score, descending=True)
    ranks = torch.where(args_max == idx_.unsqueeze(1))[1]
    top1 = args_max[:, 0]

    # Compute metrics
    r1 = (ranks < 1).sum() / ranks.size(0) * 100
    r5 = (ranks < 5).sum() / ranks.size(0) * 100
    r10 = (ranks < 10).sum() / ranks.size(0) * 100

    medr = ranks.median() + 1
    meanr = (ranks * 1.0).mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)