from typing import List, Tuple
import torch

from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn

from .encoder import ImageEncoder, TextEncoder
from lib.contrastive_loss import ContrastiveLoss


class SCAN(nn.Module):
    """
    Text Image Stacked Cross Attention Network (SCAN)

    Attributes
    ----------
    grad_clip: gradient clipping
    img_enc: image encoder
    txt_enc: text encoder
    criterion: triplet loss
    """

    def __init__(
        self,
        vocab_size: int,
        img_dim: int,
        word_dim: int,
        embed_size: int,
        lambda_softmax: float,
        grad_clip: float,
        margin: float,
    ):
        super().__init__()
        # Build Models
        self.lambda_softmax = lambda_softmax
        self.grad_clip = grad_clip
        self.margin = margin
        self.img_enc = ImageEncoder(img_dim, embed_size)
        self.txt_enc = TextEncoder(vocab_size, word_dim, embed_size)

        # Loss function
        self.criterion = ContrastiveLoss(margin)

    def forward(self, images: Tensor, captions: Tensor, max_seq_len: Tensor) -> float:
        """
        Parameters
        ----------
        images:      (batch, regions, hidden_dim)
        captions:    (batch, cap_lenghts, hidden_dim)
        cap_lengths: (batch)

        Returns
        -------
        sim_scores: (batch, batch)
        """
        # Projecting the image/caption to embedding dimensions
        img_embs = self.img_enc(images)
        cap_embs, cap_lens = self.txt_enc(captions, max_seq_len)
        # -> img_embs: (batch, regions, hidden)
        # -> cap_embs: (batch, max_seq_len, hidden)

        # Compute similarity between each region and each token
        # cos(u, v) = (u * v) / (||u|| * ||v||)
        #           = (u / ||u||) * (v / ||v||)
        img_embs = F.normalize(img_embs, dim=-1)
        cap_embs = F.normalize(cap_embs, dim=-1)

        img_embs = img_embs.unsqueeze(0)
        cap_embs = img_embs.unsqueeze(1)
        # -> img_embs: (1, batch, regions, hidden)
        # -> cap_embs: (batch, 1, max_seq_len, hidden)

        # After normalizing: cos(u, v) = u * v
        sim_region_token = torch.matmul(cap_embs, img_embs.transpose(-1, -2))
        # -> sim_region_token: (batch, batch, max_seq_len, regions)

        alpha = self.lambda_softmax * sim_region_token
        # -> alpha: (batch, batch, max_seq_len, regions)

        # Create mask from caption length
        # torch.arange(max_seq_len_SIZE).expand(batch_SIZE, max_seq_len_SIZE)
        padding_mask = torch.arange(cap_embs.size(2)).expand(
            cap_embs.size(0), cap_embs.size(2)
        ) >= cap_lens.unsqueeze(1)
        padding_mask = padding_mask.unsqueeze(1)
        padding_mask = padding_mask.unsqueeze(-1)
        # -> padding_mask: (batch, 1, max_seq_len, 1)

        # Mask score of padding tokens
        alpha.data.masked_fill_(padding_mask, -float('inf'))
        attention_weights = F.softmax(alpha, dim=-1)
        # -> attention_weights: (batch, batch, max_seq_len, regions)

        # Calculate weighted sum of regions
        attention = torch.matmul(attention_weights, img_embs)
        # -> attention: (batch, batch, max_seq_len, regions)

        # Calculate the importance of each attended image vector
        # w.r.t each sentence's tokens
        r = F.cosine_similarity(cap_embs, attention, dim=-1)
        # -> r: (batch, batch, max_seq_len)

        # Zeros similarity of padding tokens
        padding_mask = padding_mask.squeeze().expand_as(r)
        r[padding_mask] = 0

        # Average pooling
        sim_scores = r.mean(dim=-1)
        # -> sim_score: (batch, batch)

        return sim_scores
