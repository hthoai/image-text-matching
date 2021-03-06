from typing import Tuple

import math

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ImageEncoder(nn.Module):
    """An image feature wraper"""

    def __init__(self, img_size: int, enc_size: int):
        super(ImageEncoder, self).__init__()
        self.enc_size = enc_size

        self.fc = nn.Linear(img_size, enc_size)

        self.init_weights()

    def init_weights(self):
        """ Xavier initialization for the fully connected layer"""
        r = math.sqrt(6.0) / math.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images: Tensor) -> Tensor:
        """Extract image features and project them onto joint embedding space"""

        img_emb = self.fc(images)

        # img_emb  = [batch size, img region count, enc size]

        # norm in the joint embedding space
        img_emb = F.normalize(img_emb, p=2, dim=-1, eps=1e-8)
        # img_emb  = [batch size, img region count, enc size]

        return img_emb


class TextEncoder(nn.Module):
    """Text encoder model"""

    def __init__(self, vocab_size: int, emb_size: int, enc_size: int):
        super(TextEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.enc_size = enc_size

        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.rnn = nn.GRU(emb_size, enc_size, bidirectional=True, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, cap: Tensor, cap_length: Tensor) -> Tensor:

        # cap [batch size, seq len]
        # cap_len [batch size]

        embedded = self.embedding(cap)
        # embedded = [batch size, seq len, emb dim]
        packed_embedded = pack_padded_sequence(embedded, cap_length, batch_first=True)

        packed_outputs, _ = self.rnn(packed_embedded)
        # packed_outputs is a packed sequence containing all hidden states

        cap_emb, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        # cap_emb: [batch size, seq len, 2 * enc_size]

        batch_size = cap.shape[0]
        seq_length = cap.shape[1]
        if not self.rnn.batch_first:
            batch_size, seq_length = seq_length, batch_size

        # avg pooling over 2 direction
        cap_emb = cap_emb.view(batch_size, seq_length, 2, self.enc_size).mean(axis=-2)
        # cap_emb: [batch size, seq len, enc size]

        # norm in the joint embedding space
        cap_emb = F.normalize(cap_emb, p=2, dim=-1, eps=1e-8)
        # cap_emb: [batch size, seq len, enc size]

        return cap_emb
