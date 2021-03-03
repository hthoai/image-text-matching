from torch import Tensor
import torch.nn as nn

class TripletLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(
        self,
        margin: float
    ):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(
        self,
        scores: Tensor
    ) -> float:
        # scores: (No. cap, No. img)

        # get positive score of caption with image
        positive_scores = scores.diag()

        # remove pos scores
        scores.fill_diagonal_(0)

        # get hardest negatives
        hardest_neq_images = scores.max(dim=1)
        hardest_neq_captions = scores.max(dim=0)

        loss_captions = (self.margin - positive_scores + hardest_neq_captions).clamp(min=0)
        loss_images = (self.margin - positive_scores + hardest_neq_images).clamp(min=0)

        return loss_captions.sum() + loss_images.sum()