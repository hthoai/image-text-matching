from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor

from utils.norm import l2norm
from utils.cosine_similarity import cosine_similarity


def func_attention(
    query: Tensor, context: Tensor, params: dict, smooth: float
) -> Tuple[Tensor, Tensor]:
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    # queryT = torch.transpose(query, 1, 2)

    # # (batch, sourceL, d)(batch, d, queryL)
    # # --> (batch, sourceL, queryL)
    # attn = torch.bmm(context, queryT)
    q_norm = F.normalize(query, dim=-1)  # q_norm: (batch, queryL, d)
    c_norm = F.normalize(context, dim=-1)  # c_norm: (batch, contextL, d)
    q_transposed = q_norm.transpose(-1, -2)  # q_transposed (batch, d, queryL)\
    attn = torch.matmul(c_norm, q_transposed)

    if params["raw_feature_norm"] == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif params["raw_feature_norm"] == "l2norm":
        attn = l2norm(attn, 2)
    elif params["raw_feature_norm"] == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif params["raw_feature_norm"] == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif params["raw_feature_norm"] == "no_norm":
        pass
    else:
        raise ValueError("Unknown first norm type:", params["raw_feature_norm"])
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


def xattn_score_t2i(
    images: Tensor, captions: Tensor, cap_lens: int, params: dict
) -> Tensor:
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(
            cap_i_expand, images, params, smooth=params["lambda_softmax"]
        )
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if params["agg_func"] == "LogSumExp":
            row_sim.mul_(params["lambda_lse"]).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / params["lambda_lse"]
        elif params["agg_func"] == "Max":
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif params["agg_func"] == "Sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif params["agg_func"] == "Mean":
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(params["agg_func"]))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


def xattn_score_i2t(
    images: Tensor, captions: Tensor, cap_lens: int, params: dict
) -> Tensor:
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(
            images, cap_i_expand, params, smooth=params["lambda_softmax"]
        )
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if params["agg_func"] == "LogSumExp":
            row_sim.mul_(params["lambda_lse"]).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / params["lambda_lse"]
        elif params["agg_func"] == "Max":
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif params["agg_func"] == "Sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif params["agg_func"] == "Mean":
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(params["agg_func"]))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities
