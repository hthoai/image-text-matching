from typing import Tuple
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ImageEncoder(nn.Module):
    """An image feature wraper"""

    def __init__(self, img_dim: int, emb_dim: int):
        super(ImageEncoder, self).__init__()
        self.emb_dim = emb_dim

        self.fc = nn.Linear(img_dim, emb_dim)

        self.init_weights()

    def init_weights(self):
        """ Xavier initialization for the fully connected layer"""

        pass

    def forward(self, images: Tensor) -> Tensor:
        """Extract image features and project them onto joint embedding space"""

        img_emb = self.fc(images)

        # img_features  = [batch size, img region count, embed dim]

        # norm in the joint embedding space
        img_emb = F.normalize(img_emb, p=2, dim=-1)

        return img_emb


class TextEncoder(nn.Module):
    """A text feature wraper"""

    def __init__(self, vocab_sz: int, emb_dim: int, enc_dim: int):
        super(TextEncoder, self).__init__()

        self.vocab_sz = vocab_sz
        self.emb_dim = emb_dim
        self.enc_dim = enc_dim

        self.embedding = nn.Embedding(vocab_sz, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_dim, bidirectional=True, batch_first=True)

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, cap: Tensor, cap_len: int) -> Tuple[Tensor, Tensor]:

        # cap [cap len, batch size]
        # cap_len [batch size]

        embedded = self.embedding(cap)
        # embedded = [cap len, batch size, emb dim]

        packed_embedded = pack_padded_sequence(embedded, cap_len, batch_first=True)

        packed_outputs, _ = self.rnn(packed_embedded)
        # packed_outputs is a packed sequence containing all hidden states

        cap_emb, cap_len = pad_packed_sequence(packed_outputs, batch_first=True)

        cap_emb = (
            cap_emb[:, :, : cap_emb.size(2) // 2]
            + cap_emb[:, :, cap_emb.size(2) // 2 :]
        ) / 2

        # norm in the joint embedding space
        cap_emb = F.normalize(cap_emb, p=2, dim=-1)

        return cap_emb, cap_len
