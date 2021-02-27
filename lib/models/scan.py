"""SCAN model"""

import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from lib.models.encoder import ImageEncoder, TextEncoder
from lib.contrastive_loss import ContrastiveLoss
device = "cuda"


class SCAN(nn.Module):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self,
                 grad_clip:int,
                 img_size: int,
                 enc_size: int,
                 vocab_size: int,
                 emb_size: int):

        # Build Models
        self.grad_clip = grad_clip
        self.img_enc = ImageEncoder(img_size, enc_size)
        self.txt_enc = TextEncoder(vocab_size, emb_size, enc_size)

        
        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0
    
    def cuda(self, device=None):
        self.img_enc.cuda(device)
        self.txt_enc.cuda(device)
        # cudnn.benchmark = True


    def forward(self, images, captions, lengths):
        """Compute the image and caption embeddings
        """
        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens

    def forward_loss(self, img_emb, cap_emb, cap_len, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, cap_len)
        self.logger.update('Le', loss.data, img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

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
