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
        # self.
        self.img_enc = ImageEncoder(img_dim, embed_size)
        self.txt_enc = TextEncoder(vocab_size, word_dim, embed_size)

        # Loss function
        self.criterion = ContrastiveLoss(margin)

    def forward(self, images: Tensor, captions: Tensor, cap_lengths: Tensor) -> float:
        """
        Parameters
        ----------
        images:     (batch, regions, hidden_dim)
        captions:   (batch, max_seq_len, hidden_dim)
        cap_lengths:(batch)
        """
        # Projecting the image/caption to embedding dimensions
        img_embs = self.img_enc(images)
        cap_embs, cap_lens = self.txt_enc(captions, cap_lengths)

        # img_embs: (batch, regions, hidden)
        # cap_embs: (batch, seq len, hidden)

        # Compute similarity between each region and each token
        # cos(u, v) = u*v / ||u|| * ||v||
        # ==> u/||u|| * v/||v||
        img_embs = F.normalize(img_embs, dim=-1)
        cap_embs = F.normalize(cap_embs, dim=-1)
        similarity_matrix = torch.bmm(cap_embs, img_embs.permute(0, 2, 1))
        # similarity_matrix: (batch, seq len, regions)

        alpha = self.lambda_softmax * similarity_matrix
        # create mask from caption length
        padding_mask = torch.arange(cap_embs.size(1)).expand(cap_embs.size(0), cap_embs.size(1)) >= cap_lengths.unsqueeze(1)
        # mask score of padding token
        alpha.data.masked_fill_(padding_mask, -float('inf'))
        att_weight = F.softmax(alpha, dim=2)

        # calculate weighted sum of regions
        att = torch.bmm(att_weight, img_embs.transpose(1,2))
        # att: (batch, seq len, hidden)

        # Calculate token importance w.r.t each image
        r = torch.cosine_similarity(cap_embs, att, dim=2)
        # r: (batch, seq len)

        # zeros similarity of padding token
        r[padding_mask] = 0

        # average along sequence length
        sim_score = r.sum(axis=1) / cap_lengths
        # sim_score: (batch size)

        return sim_score

    def loss(self, img_emb: Tensor, cap_emb: Tensor, cap_len: int, **kwargs) -> float:
        """Compute the loss given pairs of image and caption embeddings"""
        loss = self.criterion(img_emb, cap_emb, cap_len)
        return loss
