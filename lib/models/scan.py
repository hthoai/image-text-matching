# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.nn.utils.clip_grad import clip_grad_norm_

from .image_encoder import EncoderImage
from .text_encoder import EncoderText
from lib.contrastive_loss import ContrastiveLoss

# import torch_xla
# import torch_xla.core.xla_model as xm

device = "cuda"


class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """

    def __init__(
        self,
        img_dim,
        word_dim,
        embed_size,
        precomp_enc_type,
        no_imgnorm,
        num_layers,
        bi_gru,
        no_txtnorm,
        grad_clip,
    ):
        # Build Models
        self.grad_clip = grad_clip
        self.img_enc = EncoderImage(img_dim, embed_size, precomp_enc_type, no_imgnorm)
        self.txt_enc = EncoderText(word_dim, embed_size, num_layers, bi_gru, no_txtnorm)
        # if torch.cuda.is_available():
        self.img_enc.to(device)
        self.txt_enc.to(device)
        # cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(
            self.cross_attn,
            self.raw_feature_norm,
            self.agg_func,
            self.margin,
            self.lambda_lse,
            self.lambda_softmax,
        )
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        # self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings"""
        # # Set mini-batch dataset
        # images = Variable(images, volatile=volatile)
        # captions = Variable(captions, volatile=volatile)
        # if torch.cuda.is_available():
        images = images.to(device)
        captions = captions.to(device)

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens

    def forward_loss(self, img_emb, cap_emb, cap_len, **kwargs):
        """Compute the loss given pairs of image and caption embeddings"""
        loss = self.criterion(img_emb, cap_emb, cap_len)
        self.logger.update("Le", loss.data, img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions."""
        self.Eiters += 1
        self.logger.update("Eit", self.Eiters)
        self.logger.update("lr", self.optimizer.param_groups[0]["lr"])

        # compute the embeddings
        img_emb, cap_emb, cap_lens = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
