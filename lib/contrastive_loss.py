import torch
import torch.nn as nn
from utils.norm import l1norm, l2norm


device = "cuda"


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(
        self,
        cross_attn,
        raw_feature_norm,
        agg_func,
        margin,
        lambda_lse,
        lambda_softmax,
        max_violation=False,
    ):
        super(ContrastiveLoss, self).__init__()
        self.cross_attn = cross_attn
        self.raw_feature_norm = raw_feature_norm
        self.agg_func = agg_func
        self.margin = margin
        self.lambda_lse = lambda_lse
        self.lambda_softmax = lambda_softmax
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == "t2i":
            scores = self.xattn_score_t2i(im, s, s_l, self.opt)
        elif self.opt.cross_attn == "i2t":
            scores = self.xattn_score_i2t(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:", self.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        # I = Variable(mask)
        # if torch.cuda.is_available():
        I = mask.to(device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

    def xattn_score_t2i(self, images, captions, cap_lens):
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
            weiContext, attn = self.func_attention(
                cap_i_expand, images, smooth=self.lambda_softmax
            )
            cap_i_expand = cap_i_expand.contiguous()
            weiContext = weiContext.contiguous()
            # (n_image, n_word)
            row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
            if self.agg_func == "LogSumExp":
                row_sim.mul_(self.lambda_lse).exp_()
                row_sim = row_sim.sum(dim=1, keepdim=True)
                row_sim = torch.log(row_sim) / self.lambda_lse
            elif self.agg_func == "Max":
                row_sim = row_sim.max(dim=1, keepdim=True)[0]
            elif self.agg_func == "Sum":
                row_sim = row_sim.sum(dim=1, keepdim=True)
            elif self.agg_func == "Mean":
                row_sim = row_sim.mean(dim=1, keepdim=True)
            else:
                raise ValueError("unknown aggfunc: {}".format(self.agg_func))
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)

        return similarities

    def xattn_score_i2t(self, images, captions, cap_lens):
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
            weiContext, attn = self.func_attention(
                images, cap_i_expand, smooth=self.lambda_softmax
            )
            # (n_image, n_region)
            row_sim = cosine_similarity(images, weiContext, dim=2)
            if self.agg_func == "LogSumExp":
                row_sim.mul_(self.lambda_lse).exp_()
                row_sim = row_sim.sum(dim=1, keepdim=True)
                row_sim = torch.log(row_sim) / self.lambda_lse
            elif self.agg_func == "Max":
                row_sim = row_sim.max(dim=1, keepdim=True)[0]
            elif self.agg_func == "Sum":
                row_sim = row_sim.sum(dim=1, keepdim=True)
            elif self.agg_func == "Mean":
                row_sim = row_sim.mean(dim=1, keepdim=True)
            else:
                raise ValueError("unknown aggfunc: {}".format(self.agg_func))
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)
        return similarities

    def func_attention(self, query, context, smooth, eps=1e-8):
        """
        query: (n_context, queryL, d)
        context: (n_context, sourceL, d)
        """
        batch_size_q, queryL = query.size(0), query.size(1)
        batch_size, sourceL = context.size(0), context.size(1)

        # Get attention
        # --> (batch, d, queryL)
        queryT = torch.transpose(query, 1, 2)

        # (batch, sourceL, d)(batch, d, queryL)
        # --> (batch, sourceL, queryL)
        attn = torch.bmm(context, queryT)
        if self.raw_feature_norm == "softmax":
            # --> (batch*sourceL, queryL)
            attn = attn.view(batch_size * sourceL, queryL)
            attn = nn.Softmax()(attn)
            # --> (batch, sourceL, queryL)
            attn = attn.view(batch_size, sourceL, queryL)
        elif self.raw_feature_norm == "l2norm":
            attn = l2norm(attn, 2)
        elif self.raw_feature_norm == "clipped_l2norm":
            attn = nn.LeakyReLU(0.1)(attn)
            attn = l2norm(attn, 2)
        # elif opt.raw_feature_norm == "l1norm":
        #     attn = l1norm_d(attn, 2)
        # elif opt.raw_feature_norm == "clipped_l1norm":
        #     attn = nn.LeakyReLU(0.1)(attn)
        #     attn = l1norm_d(attn, 2)
        elif self.raw_feature_norm == "clipped":
            attn = nn.LeakyReLU(0.1)(attn)
        elif self.raw_feature_norm == "no_norm":
            pass
        else:
            raise ValueError("unknown first norm type:", self.raw_feature_norm)
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


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
