import torch

from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn

from .encoder import ImageEncoder, TextEncoder



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
    ):
        super().__init__()
        # Build Models
        self.lambda_softmax = lambda_softmax
        self.grad_clip = grad_clip
        self.img_enc = ImageEncoder(img_dim, embed_size)
        self.txt_enc = TextEncoder(vocab_size, word_dim, embed_size)

    def forward(self, images: Tensor, captions: Tensor, cap_lens: Tensor) -> Tensor:
        """
        Parameters
        ----------
        images:      (batch, regions, hidden_dim)
        captions:    (batch, cap_lenghts, hidden_dim)
        cap_lengths: (batch)

        Returns
        -------
        sim_scores: (batch cap, batch image)
        """
        # Projecting the image/caption to embedding dimensions
        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, cap_lens)
        # -> img_embs: (batch, regions, hidden)
        # -> cap_embs: (batch, max_seq_len, hidden)

        # Compute similarity between each region and each token
        # cos(u, v) = (u @ v) / (||u|| * ||v||)
        #           = (u / ||u||) @ (v / ||v||)
        img_embs = F.normalize(img_embs, dim=-1)
        cap_embs = F.normalize(cap_embs, dim=-1)

        img_embs.unsqueeze_(0)
        cap_embs.unsqueeze_(1)
        # -> img_embs: (1, batch, regions, hidden)
        # -> cap_embs: (batch, 1, max_seq_len, hidden)

        # After normalizing: cos(u, v) = u @ v
        sim_token_region = torch.matmul(cap_embs, img_embs.transpose(-1, -2))
        # -> sim_token_region: (batch, batch, max_seq_len, regions)

        att_score = self.lambda_softmax * sim_token_region
        # -> att_score: (batch, batch, max_seq_len, regions)

        # Create mask from caption length
        # torch.arange(max_seq_len_SIZE).expand(batch_SIZE, max_seq_len_SIZE)
        padding_mask = torch.arange(cap_embs.size(2)).expand(
            cap_embs.size(0), cap_embs.size(2)
        ) >= cap_lens.unsqueeze(1)
        # padding_mask: (batch, max_seq_len)
        
        padding_mask = padding_mask.unsqueeze(-1).unsqueeze(1).to(cap_embs.device)
        # -> padding_mask: (batch, 1, max_seq_len, 1)

        # mask score of padding tokens
        att_score.data.masked_fill_(padding_mask, -float('inf'))

        # softmax along regions axis
        attention_weights = F.softmax(att_score, dim=-1)
        # -> attention_weights: (batch, batch, max_seq_len, regions)

        # Calculate weighted sum of regions -> attended image vectors
        attention = torch.matmul(attention_weights, img_embs)
        # -> attention: (batch, batch, max_seq_len, hidden_dim)

        # Calculate the importance of each attended image vector
        # w.r.t each token of sentence
        r = F.cosine_similarity(cap_embs, attention, dim=-1)
        # -> r: (batch, batch, max_seq_len)

        # Zeros similarity of padding tokens
        r[r.isnan()] = 0

        # Calculate similarity of each caption and each image by averaging all tokens in a caption.
        sim_scores = r.sum(dim=-1) / cap_lens.view(-1, 1).to(cap_embs.device)
        # -> sim_score: (batch, batch)

        return sim_scores
