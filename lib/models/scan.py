# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""

from typing import Tuple

from torch.functional import Tensor
import torch.nn as nn

from .image_encoder import EncoderImage
from .text_encoder import EncoderText
from lib.contrastive_loss import ContrastiveLoss

# import torch_xla
# import torch_xla.core.xla_model as xm


class SCAN(nn.Module):
    """
    Stacked Cross Attention Network (SCAN) model
    """

    def __init__(
        self,
        vocab_size: int,
        img_dim: int,
        word_dim: int,
        embed_size: int,
        precomp_enc_type: str,
        no_imgnorm: bool,
        num_layers: int,
        use_bi_gru: bool,
        no_txtnorm: bool,
        grad_clip: float,
        cross_attn: str,
        raw_feature_norm: str,
        agg_func: str,
        margin: float,
        lambda_lse: float,
        lambda_softmax: float,
    ):
        super().__init__()
        # Build Models
        self.grad_clip = grad_clip
        self.img_enc = EncoderImage(img_dim, embed_size, precomp_enc_type, no_imgnorm)
        self.txt_enc = EncoderText(
            vocab_size, word_dim, embed_size, num_layers, use_bi_gru, no_txtnorm
        )

        # Loss function
        self.criterion = ContrastiveLoss(
            cross_attn,
            raw_feature_norm,
            agg_func,
            margin,
            lambda_lse,
            lambda_softmax,
        )

    def forward(
        self, images: Tensor, captions: Tensor, lengths: int
    ) -> Tuple[Tensor, Tensor, int]:
        """Compute the image and caption embeddings"""
        img_emb = self.img_enc(images)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens

    def loss(self, img_emb: Tensor, cap_emb: Tensor, cap_len: int, **kwargs) -> float:
        """Compute the loss given pairs of image and caption embeddings"""
        loss = self.criterion(img_emb, cap_emb, cap_len)
        return loss
